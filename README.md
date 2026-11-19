# 🐿️ Squirrel Simulator

A behavioral simulation of squirrel foraging ecology on a 2D integer grid, written in **pure Python (stdlib only — zero external dependencies)**. A neural agent learns to cache nuts during summer and retrieve them during winter under strict movement and energy budgets, operating under a **strictly blind, stateless** constraint.

---

## 📋 Features & Constraints

- **2D Discrete Grid:** Unbounded integer lattice $\mathbb{Z}^2$ with the home burrow at origin $(0, 0)$.
- **Strictly Blind Sensing:** The squirrel only senses whether a nut exists on the exact cell $(x, y)$ it occupies—no remote scent, vision, or radar.
- **Stateless Across Seasons:** No episodic memory map of cache coordinates. Survival relies purely on co-adapting summer caching routes and winter retrieval routes via a 12-dimensional local observation vector (including cyclic seasonal clock features $\sin/\cos(2\pi \cdot d / 30)$).
- **Action Economy:**
  - Cardinal movement (`North`, `South`, `East`, `West`) costs 1 move; `Stay` at home ends the day early.
  - Burying (summer) and digging/retrieving (winter) are free actions (0 moves).
  - Single-nut carrying capacity in summer ($P_{\text{find}} = 20\%$).
  - **Conserving harvest:** In winter, the squirrel stops digging once its daily quota $K$ is met and returns home via the satiated homing reflex.
- **Survival Objective:**
  - Daily budget: $M$ moves (must return to $(0, 0)$ every night or die stranded).
  - Summer: 30 days to forage and bury nuts.
  - Winter: 30 days to dig up at least $K$ nuts per day or starve.

---

## 🧠 Model & Training

- **Policy Network (`PolicyMLP`):** Pure-Python feed-forward MLP `(12 → 20 → 20 → 5)` with `tanh` hidden activations, serialized to JSON.
- **Neuroevolution (`GeneticTrainer`):** Tournament selection, uniform crossover, Gaussian weight mutation, and elitism optimizing winter survival days, nuts retrieved, summer caching/exploration coverage, and lost-bury-opportunity penalties.
- **Heuristic Baselines:**
  - `RandomSquirrel`: Random walk with homing awareness.
  - `SectorPatrolSquirrel`: Rotating 4-quadrant disjoint 2D boustrophedon (lawnmower) sweep.

---

## 📊 Difficulty Tiers & Benchmark Results

All benchmarks use the exact same `(12, 20, 20, 5)` MLP across 5 unseen evaluation seeds (`[42, 100, 2024, 7, 999]`):

| Tier | Config ($M$, $K$, $P$) | Reachable Cells | Winter Demand | Random | SectorPatrol | Neural MLP (5-seed avg / max) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard** | `M=30, K=1, P=0.20` | 480 | 30 nuts | ~18 / 30 | 30.0 / 30 | **30.0 / 30** (max 30) |
| **Hard (`--hard`)** | `M=22, K=2, P=0.20` | 264 | 60 nuts | 11.4 / 30 | 30.0 / 30 | **28.6 / 30** (max 30) |
| **Extreme** | `M=16, K=2, P=0.20` | 144 | 60 nuts | 9.6 / 30 | 20.2 / 30 | **19.2 / 30** (max 26, solved train Gen 79) |
| **Nightmare** | `M=22, K=3, P=0.20` | 264 | 90 nuts | 0.0 / 30 | 19.0 / 30 | **26.4 / 30** (max 29) |

---

## 🚀 Quickstart & Usage

Run commands from the parent directory of `squirrel_sim/` (or ensure it is on `PYTHONPATH`):

### 1. Run Heuristic Baselines
```bash
# Standard mode (M=30, K=1)
python3 squirrel_sim/main.py --mode baseline

# Hard mode (M=22, K=2)
python3 squirrel_sim/main.py --mode baseline --hard
```

### 2. Train the Neural Policy
```bash
# Standard mode
python3 squirrel_sim/main.py --mode train --generations 30 --pop-size 60 --save-path trained_squirrel.json

# Hard mode (M=22, K=2)
python3 squirrel_sim/main.py --mode train --hard --generations 30 --pop-size 60 --save-path trained_squirrel_hard.json

# Custom parameters (e.g. M=22, K=3)
python3 squirrel_sim/main.py --mode train --move-budget 22 --winter-quota 3 --p-find 0.20 --generations 80 --pop-size 80 --save-path trained_m22_k3.json
```

### 3. Generate an Interactive HTML Replay Viewer
Exports a **single self-contained `.html` file** (no server required) with dynamic viewport sizing, day/step scrubbers, playback speed, config table, clickable **Nuts in Ground per Day** line graph, per-day event log, and keyboard controls (`←`, `→`, `Space`):

```bash
# Replay a trained MLP checkpoint
python3 squirrel_sim/main.py --mode replay --hard \
  --load-path squirrel_sim/checkpoints/trained_squirrel_hard.json \
  --seed 2024 --html-out squirrel_replay.html

# Replay the M=22, K=3 model (29/30 on seed 999)
python3 squirrel_sim/main.py --mode replay --move-budget 22 --winter-quota 3 \
  --load-path squirrel_sim/checkpoints/trained_squirrel_m22_k3_v4.json \
  --seed 999 --html-out squirrel_replay_m22_k3.html

# Replay heuristic baselines (Sector Patrol or Random Walk)
python3 squirrel_sim/main.py --mode replay --hard --agent sector --seed 2024 --html-out sector_replay.html
python3 squirrel_sim/main.py --mode replay --hard --agent random --seed 2024 --html-out random_replay.html
```

### 4. Terminal ASCII Demo
```bash
python3 squirrel_sim/main.py --mode demo --hard \
  --load-path squirrel_sim/checkpoints/trained_squirrel_hard.json --seed 2024
```

---

## 📁 Repository Structure

```text
squirrel_sim/
├── main.py                         # CLI entry point (baseline, train, demo, replay)
├── squirrel_replay_template.html   # Self-contained HTML replay viewer template
├── env/
│   ├── grid_world.py               # Sparse 2D integer lattice cache store
│   └── squirrel_env.py             # Seasonal state machine, 12D obs, homing reflex
├── models/
│   ├── mlp.py                      # Pure-Python PolicyMLP with JSON serialization
│   └── heuristics.py               # RandomSquirrel & disjoint 4-quadrant SectorPatrol
├── train/
│   └── genetic_algo.py             # Neuroevolution trainer & history logger
├── viz/
│   └── terminal_viz.py             # ANSI/ASCII terminal grid renderer
└── checkpoints/                    # Pre-trained policy weights (Standard, Hard, M16/K2, M22/K3)
```
