---
layout: post
title: "A few Interesting Patterns of Conway's Game of Life"
title_en: "A few Interesting Patterns of Conway's Game of Life"
subtitle_en: "Simple rules, emergent complexity"
date: 2026-01-31
categories: [Tech]
preview_type: canvas_animation
---



<div markdown="0">
<style>
#lifeContainer { margin: 40px auto; max-width: 700px; text-align: center; }
#lifeCanvas { background: #fff; display: block; margin: 0 auto; }
.life-controls { display: flex; gap: 20px; justify-content: center; align-items: center; margin-top: 20px; }
.life-btn { padding: 8px 16px; font-size: 13px; border: none; background: none; cursor: pointer; font-family: monospace; transition: opacity 0.2s; }
.life-btn:hover { opacity: 0.6; }
.life-btn:active { transform: scale(0.95); }
.life-info { font-family: monospace; font-size: 14px; color: #333; }
</style>

<div id="lifeContainer">
<canvas id="lifeCanvas" width="600" height="400"></canvas>
<div class="life-controls">
<button class="life-btn" id="prevBtn">← Prev</button>
<span class="life-info" id="patternName">Gosper Gun</span>
<button class="life-btn" id="nextBtn">Next →</button>
</div>
</div>

<script>
const canvas = document.getElementById('lifeCanvas');
const ctx = canvas.getContext('2d');
const CELL_SIZE = 8;
const COLS = Math.floor(canvas.width / CELL_SIZE);
const ROWS = Math.floor(canvas.height / CELL_SIZE);

let grid = Array(ROWS).fill(null).map(() => Array(COLS).fill(0));
let generation = 0;
let running = true;
let currentPattern = '-';

// Patterns
const patternList = [
  {
    name: 'Gosper Gun',
    cells: [
      [5,1],[5,2],[6,1],[6,2],[5,11],[6,11],[7,11],[4,12],[3,13],[3,14],[8,12],[9,13],[9,14],
      [6,15],[4,16],[5,17],[6,17],[7,17],[6,18],[8,16],[3,21],[4,21],[5,21],[3,22],[4,22],
      [5,22],[2,23],[6,23],[1,25],[2,25],[6,25],[7,25],[3,35],[4,35],[3,36],[4,36]
    ],
    offsetX: 5,
    offsetY: 5
  },
  {
    name: 'Acorn',
    cells: [[0,1],[1,3],[2,0],[2,1],[2,4],[2,5],[2,6]],
    offsetX: 20,
    offsetY: 15
  },
  {
    name: 'R-pentomino',
    cells: [[10,11],[10,12],[11,10],[11,11],[12,11]],
    offsetX: 30,
    offsetY: 20
  }
];

let currentPatternIndex = 0;

function loadPattern(index) {
  currentPatternIndex = index;
  const pattern = patternList[index];
  grid = Array(ROWS).fill(null).map(() => Array(COLS).fill(0));
  generation = 0;

  pattern.cells.forEach(([r, c]) => {
    if (r + pattern.offsetY < ROWS && c + pattern.offsetX < COLS) {
      grid[r + pattern.offsetY][c + pattern.offsetX] = 1;
    }
  });

  document.getElementById('patternName').textContent = pattern.name;
  draw();
}

function nextPattern() {
  currentPatternIndex = (currentPatternIndex + 1) % patternList.length;
  loadPattern(currentPatternIndex);
}

function prevPattern() {
  currentPatternIndex = (currentPatternIndex - 1 + patternList.length) % patternList.length;
  loadPattern(currentPatternIndex);
}

function countNeighbors(r, c) {
  let count = 0;
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const nr = r + dr;
      const nc = c + dc;
      // Fixed boundary - cells outside are dead
      if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
        count += grid[nr][nc];
      }
    }
  }
  return count;
}

function update() {
  const newGrid = Array(ROWS).fill(null).map(() => Array(COLS).fill(0));
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const neighbors = countNeighbors(r, c);
      if (grid[r][c] === 1) {
        newGrid[r][c] = (neighbors === 2 || neighbors === 3) ? 1 : 0;
      } else {
        newGrid[r][c] = (neighbors === 3) ? 1 : 0;
      }
    }
  }
  grid = newGrid;
  generation++;
}

function draw() {
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#000';
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (grid[r][c] === 1) {
        ctx.fillRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1);
      }
    }
  }
}

function gameLoop() {
  if (running) {
    update();
  }
  draw();
  setTimeout(gameLoop, 100);
}

// Event listeners
document.getElementById('prevBtn').addEventListener('click', prevPattern);
document.getElementById('nextBtn').addEventListener('click', nextPattern);

// Start with first pattern
loadPattern(0);
gameLoop();
</script>
</div>

<div class="lang-en" markdown="1">

[Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) is a cellular automaton with simple rules. It is Turing-complete, demonstrating that universal computation emerges from simple local rules rather than being explicitly encoded in them.


<footer style="font-size: 0.85em; color: #666; margin-top: 2em; padding-top: 1em; border-top: 1px solid #ddd;">
  <p><strong>Conway's Game of Life</strong> was created by mathematician John Horton Conway in 1970. Published in <em>Scientific American</em>, it became one of the most famous examples of cellular automata.</p>
  <p>Code generated with AI coding assistant.</p>
</footer>

</div>

