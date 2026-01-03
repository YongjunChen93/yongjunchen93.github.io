---
layout: post
title_en: "From Model Primitives to Production Features: Notes on Dissecting AI Systems in Practice"
subtitle_en: "How modeling assumptions, system constraints, and user feedback shape real AI features from an AI practionaoer  view"
date: 2026-1-1
categories: [Tech]
---


<div class="lang-en" markdown="1">

We are at a stage where the benefits of AI are widely visible.

AI progress is often framed around models: larger architectures, better objectives, and stronger benchmarks. Yet many of the most consequential challenges in practice arise outside the model itself. Assumptions made during training quietly break during inference; systems optimized for offline metrics behave unpredictably under real workloads; and AI product success depends on asymmetry feedback signals that models were never designed to observe.

This post reflects on what it takes to move from models to products, considering **models, systems, and user-facing features as equally important components** of the same pipeline. Rather than focusing on algorithms in isolation, tries to examine the assumptions, constraints, and failure modes that emerge when AI systems are deployed at scale, and how these shape real AI features.

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

As many pioneers have emphasized, computation is not just a resource but also a design constraint.

Choices around hardware, parallelism strategies, and system architecture determine
what is possible downstream. Once these foundations are set, changing them becomes
expensive—and sometimes infeasible.

Crucially, parallelism decisions are not merely about efficiency. They determine whether training runs can be debugged and reasoned about; How compute scales across training and inference; Whether rollouts and evaluation pipelines can scale with the model; Whether reinforcement learning loops remain tractable at all.

In practice, computation infrastructure defines the *shape* of the system long
before it defines its speed.

***Parallelism Starts from Communication***


Conceptually, any parallel system can be decomposed into three atoms:
- **Devices**: independent compute units (CPU cores, GPUs, TPU cores)
- **Channels**: point-to-point communication paths between devices
- **Communicators**: semantic layers that define collective operations
(e.g., all-reduce, all-gather, scatter)

<img
  src="{{ '/assets/images/ai-models-products/com.png' | relative_url }}"
  alt="Demo image"
  class="zoomable"
/>

An minimal atom relationships is illustrated in above figure.

<details>
<strong>toy demo: devices, channels, and collective communication</strong>

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

***From Communication Atoms to Parallelism Strategies***

Once these abstractions are in place,  parallelism strategies (such as DDP, FSDP, TP, Pipeline Parallelism, Export Parallelism, and hybrid solutions) naturally occupy the layer above. Many excellent resources provide detailed explanations of data, tensor, and pipeline parallelism (e.g., [Weng, 2021](https://lilianweng.github.io/posts/2021-09-25-train-large/) and [Stanford CS336](https://stanford-cs336.github.io/spring2025/)). At this level, parallelism is no longer a collection of tricks. It is a structured mapping from tensors to devices, with communication patterns determined by that mapping. 

While strategies can become arbitrarily complex, starting from simpler designs tends to make scaling smoother. As a starting point, a small script illustrating tensor parallelism is shown below.
<details>
<strong>tensor parallelism visualization</strong>

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

While computation infrastructure defines the shape of a system, training architecture determines how that shape evolves. Seemingly local architectural choices often have systemic consequences, governing parameter movement, stability, and their interaction with infrastructure constraints.

In practice, a few design patterns repeatedly become the standard choices.

***Low-Rank Adaptation (LoRA)***: Tuning a base model is often costly and difficult to scale when serving many clients. LoRA introduces trainable low-rank adapters alongside frozen base parameters ([Hu et al. 2021](https://arxiv.org/abs/2106.09685)). Rather than a shortcut to full fine-tuning, LoRA functions as an *auxiliary structure* attached to the model. Viewed this way, LoRA restricts learning to a small set of adapter parameters, keeping changes localized and the base model stable (e.g., [LoRA as programs](https://kevinlu.ai/loras-as-programs)).
This framing becomes increasingly important as models are reused, composed, and
adapted across tasks and products.

<details>
<strong>side notes on LoRA:</strong> In LoRA, the original weight matrix $W \in \real^{d_{out}\times d_{in}}$ is kept frozen, and updates are expressed through a low-rank decomposition:

<div class="lang-en" markdown="0"> 

$$ \begin{aligned} 
W' &= W + \Delta W \\ 
\Delta W &= B A \\ 
A &\in \mathbb{R}^{r \times d_{\text{in}}}, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}, \quad r \ll \min(d_{\text{in}}, d_{\text{out}}) \end{aligned} $$

</div>

</details>

***Memory-Aware Attention***： Out-of-memory (OOM) issues are often the first constraint encountered when running AI models. FlashAttention ([Dao et al. 2022](https://arxiv.org/abs/2205.14135)) and its successors ([v2](https://arxiv.org/abs/2307.08691), [v3](https://arxiv.org/abs/2407.08608)) are frequently framed as performance optimizations. Their deeper impact is architectural. Online softmax computation, originally motivated by numerical stability, combined with careful tiling and backward recomputation, reshapes the memory–compute trade-off. These techniques together enable attention patterns and context lengths that were previously
infeasible under conventional memory layouts.

<details>
<strong>abstraction</strong>

The key insight is not faster kernels, but a different memory model: attention is computed block by block, normalized online, and recomputed in backward passes instead of being materialized.

```python
# Forward pass: block-wise attention with online softmax
for each query_block:
    running_max = -inf
    running_sum = 0
    output_accumulator = 0

    for each key_value_block:
        scores = compute_attention_scores(query_block, key_value_block)

        # online softmax update
        new_max = max(running_max, max(scores))
        rescale = exp(running_max - new_max)

        probs = exp(scores - new_max)
        running_sum = running_sum * rescale + sum(probs)
        output_accumulator = (
            output_accumulator * rescale
            + probs @ value_block
        )

        running_max = new_max

    output = output_accumulator / running_sum
    save_minimal_stats(running_max, running_sum)

# Backward pass: recompute attention blocks instead of storing them
for each key_value_block:
    for each query_block:
        load_minimal_stats(query_block)

        scores = recompute_attention_scores(query_block, key_value_block)
        probs = exp(scores - running_max) / running_sum

        accumulate_gradients_for_QKV(probs)

```
</details>

***Mixture of Experts (MoE)***: 
As Transformer models scale, most computation becomes dominated by feed-forward networks (FFNs), which motivated Mixture-of-Experts (MoE) as a way to scale capacity without paying the full dense compute cost. MoE architectures ([Du et al. 2021](https://arxiv.org/abs/2112.06905), [Zoph et al. 2022](https://arxiv.org/abs/2202.08906)) are attractive primarily for scaling: by activating only a subset of experts per token, they allow model capacity to grow while keeping per-token compute roughly constant.In practice, however, MoE is less a local modeling trick than a system-level choice, routing, load balancing, and expert utilization tightly couple data, optimization, and infrastructure behavior. Imbalances formed during training often surface later as unstable rollouts, poor utilization, or degraded reinforcement learning performance, making MoE success depend on careful co-design across data, model, and infrastructure rather than architecture alone.

<details>
<strong>side notes: common techniques for stabilizing MoE</strong>

**Load balancing loss.**  
Introduced in [GLaM](https://arxiv.org/pdf/2006.16668), load balancing losses encourage uniform expert utilization by
penalizing skewed routing decisions across experts:

<div class="lang-en" markdown="0">
$$
\begin{aligned}
\mathcal{L}_{\text{balance}}
&= N \sum_{i=1}^{N} f_i \cdot p_i
\end{aligned}
$$
</div>

where $N$ is the number of experts, $f\_i$ denotes the fraction of tokens routed to
expert $i$, and $p\_i$ is the average routing probability assigned to that expert.
This term discourages expert collapse and promotes balanced utilization during training.

**Z-loss for router stability.**  
Introduced in [ST-MoE](https://arxiv.org/pdf/2202.08906), the z-loss penalizes large router logits to improve numerical
stability and softmax behavior:

<div class="lang-en" markdown="0">
$$
\begin{aligned}
\mathcal{L}_{\text{z}}
&= \log \sum_{j} \exp(z_j)
\end{aligned}
$$
</div>

where $z\_j$ are the pre-softmax router logits. Compared to simple $L2$ penalties, z-loss
regularizes the *joint scale* of logits, which is particularly effective for stabilizing
softmax-based routing under low-precision training.

</details>

## From AI Models to AI Systems

We don’t call a model failed because it performs poorly offline; we call it failed when it is deployed. After deployment, failures often appear around inference constraints, agentic
interactions, learning dynamics, and numerical stability. These are not exhaustive,
but they provide concrete lenses for understanding how models become systems.

### Inference Optimization

In autoregressive generation, attention is often the first place where inference hits real limits. While KV caching reduces decoding complexity from quadratic to linear in sequence length, naive KV allocation quickly runs into memory fragmentation and poor utilization under high concurrency. [**PagedAttention**](https://arxiv.org/abs/2309.06180) addresses this by managing KV cache as a virtualized memory system, decoupling logical sequence length from physical memory layout and enabling efficient sharing and eviction across requests. This abstraction underpins [**vLLM**](https://docs.vllm.ai/), which treats inference not as isolated forward passes, but as a continuous scheduling problem over many concurrent sequences, optimizing both throughput and time-to-first-token (TTFT). In longer-lived agentic sessions, KV reuse is often extended into a hierarchical cache, sometimes backed by a distributed KV store (**Redis Attention**), to avoid repeated prefilling across turns.


Beyond memory, decoding itself becomes a systems problem. **Speculative decoding** ([Chen et al. 2023](https://arxiv.org/abs/2302.01318), [Leviathan et al. 2023](https://arxiv.org/pdf/2211.17192)) reduces end-to-end latency by letting cheap draft models propose multiple tokens that
expensive target models verify in parallel, without changing output distributions. Systems such as [DeepSeek V3](https://arxiv.org/abs/2412.19437) use multi-token prediction to produce stronger drafts for speculative decoding, while newer variants move toward **speculative
editing** (e.g., [EfficientEdit](https://arxiv.org/pdf/2506.02780)), where only edited spans are regenerated and unchanged context is reused. At the extreme, even an **n-gram** model can serve as the draft, reducing the problem to validating local edits rather than regenerating the full sequence. A toy example is shown below.

<details>
<strong>illustration of speculative editing</strong>

```python
# prompt: original context
# draft: cheap proposal (e.g., n-gram or small model, fast to generate)
# target: large LLM

# 1. Propose edits with a cheap draft
draft = propose(prompt)

# 2. Verify draft token-by-token with the target model
logits = target(prompt + draft)
accepted = []
for i, token in enumerate(draft):
    if argmax(logits[i]) == token:
        accepted.append(token)
    else:
        break

# 3. Commit accepted prefix
text = prompt + decode(accepted)

# 4. Continue with normal decoding if needed
text += decode_with_target(target, text)
```
</details>


### Agentic Systems

While inference focuses on generating a response for a single request, agentic models
extend model behavior across steps and interactions. With the emergence of [Chain-of-Thought](https://arxiv.org/abs/2201.11903), language models began to exhibit explicit reasoning behavior rather than acting as purely passive sequence predictors. The [ReAct](https://arxiv.org/abs/2210.03629) (Reasoning and Acting) was one of the early works to formalize this shift by interleaving reasoning traces with action decisions, allowing models to reason and act in a loop instead of producing a single text continuation.

This marked a paradigm shift in how models are used. Rather than simply responding
to prompts, models began deciding what to do next—invoking tools, calling APIs, and
interacting with environments as part of their output. In agentic settings, a model
becomes an autonomous component that reasons, plans, and acts across multiple steps,
often over long horizons, with memory and feedback loops.

Compared to traditional inference, agentic models introduce a tight reasoning, planning, and action loop, persistent state across interactions, and deep integration with external tools
such as search, calculators, and execution engines. These properties make them suitable
for real-world workflows involving planning, automation, and multi-step decision
making—but they also introduce new challenges in reliability, controllability, and
long-horizon behavior.


### Reinforcement Learning for Agents

In agentic settings, supervised learning and preference modeling quickly reach their
limits: feedback is delayed, data is sparse, and good behavior depends on long-horizon
trade-offs rather than local likelihood. Once models act and interact with an
environment, learning signals become outcomes of decision sequences over time, making
reinforcement learning the minimal framework for aligning agentic behavior.

At its core, **REINFORCE** provides the policy-gradient foundation by linking actions to
delayed rewards through sampled trajectories. As systems scale, purely on-policy
learning becomes impractical, and **importance sampling**, typically with respect to a
reference or behavior policy, enables reuse of experience while correcting for
distribution mismatch. Building on these foundations, policy optimization methods for
agentic LLMs primarily differ in how they constrain policy updates and normalize learning signals at scale. **PPO** stabilizes training by explicitly restricting policy
movement through ratio clipping or KL penalties relative to a reference policy, together
with a learned value function for variance reduction, but remains sensitive to absolute
reward scale and reward-model drift; **DPO** removes explicit environment interaction
and value estimation by optimizing directly against preference comparisons under a
fixed reference, trading adaptability for simplicity when feedback is static; and
**GRPO**, in contrast, eliminates the value function and replaces absolute rewards with
within-group relative comparisons, making updates less sensitive to reward magnitude
and prompt-dependent variance—particularly effective in multi-sample,
reasoning-heavy settings where only comparative signals are reliable. Following the success of [DeepSeek-R1](https://arxiv.org/abs/2501.12948), which validated GRPO-style optimization at scale, a growing body of follow-up work has built on the same core idea ([Liu et al. 2025](https://arxiv.org/pdf/2503.20783)).

In fully agentic settings, with environment interaction and long-horizon feedback, these control issues become first-order concerns rather than implementation details. Recent work uch as [**rStar2-Agent**](https://arxiv.org/abs/2508.20722) makes this explicit: ***progress depends less on new reinforcement learning objectives than on engineering mechanisms that make
interaction, rollout, and learning stable and scalable***. In practice, reinforcement learning for agentic LLMs is best understood as a set of tools for managing variance, normalization, and forgetting, rather than as a single unified algorithm. The rapid maturation of this direction is further reflected in a recent survey by [Zhang et al. 2025](https://arxiv.org/pdf/2509.02547), which reviews over five hundred works on agentic reinforcement learning for LLMs.


### Numerical Stability


### Distillation

## AI Features

</div>


<footer class="mt-16 pt-6 border-t border-gray-200 dark:border-gray-700
               text-sm text-black-500 dark:text-black-400">
  <p>
    Notes: Post is actively revised. AI helps with English editing. Credits goes to the broader tech community for the learning and inspiration behind these notes. Notes are written for memorization and reflection, not for tracking SOTA. Please feel free to reach out with corrections or citation suggestions.
  </p>
</footer>
