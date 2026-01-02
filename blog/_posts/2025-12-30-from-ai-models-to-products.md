---
layout: post
title_en: "From Models to Products: Notes on Building AI Systems in Practice (TBD)"
subtitle_en: "How modeling assumptions, system constraints, and user feedback shape real AI features"
date: 2026-1-1
categories: [Tech]
---


<div class="lang-en" markdown="1">

We are at a stage where the benefits of AI are widely visible.

AI progress is often framed around models: larger architectures, better objectives, and stronger benchmarks. Yet many of the most consequential challenges in practice arise outside the model itself. Assumptions made during training quietly break during inference; systems optimized for offline metrics behave unpredictably under real workloads; and AI product success depends on asymmetry feedback signals that models were never designed to observe.

This post reflects on what it takes to move from models to products, treating models, systems, and user-facing features as equally important components of the same pipeline. Rather than focusing on algorithms in isolation, tries to examine the assumptions, constraints, and failure modes that emerge when AI systems are deployed at scale, and how these shape real AI features.

> Products enable iterative learning through deployment, while great products and
> research strengthen each other. Products keep us grounded in reality and guide us
> to solve the most impactful problems. 
> ---- [Thinking Machines Lab](https://thinkingmachines.ai/)

If we want AI to genuinely benefit people, understanding this end-to-end transition matters as much as improving model quality itself.


## Model Primitives
Data mixtures, computation infrastructure, and training algorithms form the core foundations of strong AI models.

### Data Determinism

It is widely recognized that data quality dominates model performance, motivating extensive effort in cleaning, labeling, sampling, and synthesis. What is far less discussed, however, is data determinism. As models scale and training pipelines grow more complex, data ceases to be a static input, it is continuously generated, filtered, mixed, and reshaped across ETL jobs, online sampling logic, distributed training, and iterative retraining. Without careful design, small sources of nondeterminism accumulate, making training runs difficult or impossible to reproduce, especially when data mixtures are constructed online rather than frozen offline.

In practice, nondeterminism often enters through seemingly reasonable choices: using large models for data synthesis with stochastic decoding, constructing online data mixtures whose sampling depends on mutable state or worker order, or modifying curricula without a stable notion of what data the model has already seen.

Most modern frameworks provide strong controls for randomness inside model execution. JAX exposes explicit PRNG keys that scale cleanly across devices, while PyTorch relies on global and worker-level seeds. These mechanisms are effective, but they operate at a different layer. Once randomness enters through data generation or distributed input pipelines, purely **stateful** control becomes difficult to reason about or replay at system scale.

The goal is not to eliminate randomness, but to make it reproducible. In practice, this
means that data pipelines must be able to answer a simple question:
*Given a global step, a seed, and a configuration, how to deterministically recover the exact data the model was trained on?*

Achieving this requires treating data sampling as a **pure, stateless function** of
explicit inputs rather than an emergent property of mutable pipeline state. Randomness
should be derived from explicit seeds and global coordinates; data sources,
preprocessing steps, and mixture weights should be versioned and auditable; and any
curriculum changes should be intentional and traceable. For streaming or continual
learning, reproducibility shifts from exact replay to controlled, well-documented
evolution.

A minimal illustration is:
```python
def sample_index(seed, epoch, global_step, dataset_size):
    h = hash(f"{seed}-{epoch}-{global_step}")
    return h % dataset_size
```

While any hashing-based scheme has limitations, the essential property is that sampling
depends only on explicit inputs, not mutable runtime state. This **stateless control**, making resuming, debugging, and comparing runs feasible even under large-scale distributed training.


### Computation Infrastructure


> The biggest lesson that can be read from 70 years of AI research 
> is that general methods that leverage computation are ultimately the most effective, 
> and by a large margin.  ---- [Richard Sutton](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf)

As many practitioners have emphasized, computation is not just a resource but also a design constraint.

Choices around hardware, parallelism strategies, and system architecture determine
what is possible downstream. Once these foundations are set, changing them becomes
expensive—and sometimes infeasible.

Crucially, parallelism decisions are not merely about efficiency. They determine whether training runs can be debugged and reasoned about; How compute scales across training and inference; Whether rollouts and evaluation pipelines can scale with the model; Whether reinforcement learning loops remain tractable at all.

In practice, computation infrastructure defines the *shape* of the system long
before it defines its speed.

**Parallelism Starts from Communication**

Conceptually, any parallel system can be decomposed into three concepts:
- **Devices**: independent compute units (CPU cores, GPUs, TPU cores)
- **Channels**: point-to-point communication paths between devices
- **Communicators**: semantic layers that define collective operations
(e.g., all-reduce, all-gather, scatter)

All familiar parallelism strategies such as DDP, FSDP, tensor parallelism, pipeline
parallelism, and export parallelism are built by composing these primitives in different ways.

The following minimal example models devices and channels explicitly, making collective
communication visible.
<img
  src="{{ '/assets/images/ai-models-products/com.png' | relative_url }}"
  alt="Demo image"
  class="zoomable"
/>


<details>
<strong>Toy demo: devices, channels, and collective communication</strong>

```python
  # device.py
  class Device:
      def __init__(self, device_id, data):
          self.id = device_id
          self.data = data

  # channel.py
  from queue import Queue

  class Channel:
      def __init__(self):
          self._queue = Queue()

      def send(self, x):
          self._queue.put(x)

      def receive(self):
          return self._queue.get()

  # communicator.py
  class Communicator:
      """
      A minimal abstraction over collective communication semantics.

      The goal is to make parallelism decisions explicit and inspectable.
      """
      def __init__(self, devices):
          self.devices = devices

      def allreduce(self):
          total = sum(d.data for d in self.devices)
          for d in self.devices:
              d.data = total

  # run_demo.py
  devices = [Device(i, i + 1) for i in range(4)]
  comm = Communicator(devices)

  comm.allreduce()
  print([d.data for d in devices])  # [10, 10, 10, 10]
```

</details>

**From Primitives to Parallelism Strategies**

Once these abstractions are in place, common parallelism strategies (DDP, FSDP, TP, Pipeline Parallelism, Export Parallelism, and hybrid solutions) naturally occupy the layer above. Many excellent resources already provide detailed explanations of data, tensor, and pipeline parallelism (e.g., [Weng, 2021](https://lilianweng.github.io/posts/2021-09-25-train-large/) and [Stanford CS336](https://stanford-cs336.github.io/spring2025/)). At this level, parallelism is no longer a collection of tricks. It is a structured mapping from tensors to devices, with communication patterns determined by that mapping. 

While strategies can become arbitrarily complex, starting from simpler choices tend to make scaling more smoother. A small script to visualize a tensor parallelism strategy correspond to its sharding layout below.

<details>
<strong>Visualization demo: tensor parallelism via sharding</strong>

```python
# Simulate multiple devices on CPU
import os
import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

print(f"Available devices: {jax.local_devices()}")

# --------------------------------------------------
# Tensor Parallelism
# --------------------------------------------------
# Intuition:
#   - Linear layer weights are partitioned across devices
#   - Communication is required to combine partial results
#
# This mirrors the Megatron-style column / row sharding pattern.

mesh = jax.make_mesh((8,), ("tp",))

# First linear layer: column-wise sharding
w1 = jax.random.normal(jax.random.key(0), (88, 88))
jax.debug.visualize_array_sharding(
    jax.device_put(w1, NamedSharding(mesh, P(None, "tp")))
)

# Second linear layer: row-wise sharding
w2 = jax.random.normal(jax.random.key(1), (88, 88))
jax.debug.visualize_array_sharding(
    jax.device_put(w2, NamedSharding(mesh, P("tp", None)))
)

# Alternative: fully sharded tensor (both dims)
w3 = jax.random.normal(jax.random.key(2), (88, 88))
jax.debug.visualize_array_sharding(
    jax.device_put(w3, NamedSharding(mesh, P("tp")))
)
```

</details>


### Training Architecture

## From AI Models to AI Systems

Models do not fail when they are wrong, They fail when they are deployed.

Once a model leaves training and faces real users, the center of gravity shifts, from architecture to systems.
Failures may not any where from numerics, latency, memory, and feedback loops, system-level constraints that quietly dominate outcomes.

### Inference Optimization


### Agentic Models

### Reinforcement Learning for Agents

### Numerical Stability

### Distillation

## Building AI Features

</div>


<footer class="mt-16 pt-6 border-t border-gray-200 dark:border-gray-700
               text-sm text-black-500 dark:text-black-400">
  <p>
    Notes: Post is actively revised. AI helps with English editing. Credits goes to the broader tech community for the learning and inspiration behind these notes. Notes are written for memorization and reflection, not for tracking SOTA. Please feel free to reach out with corrections or citation suggestions.
  </p>
</footer>

