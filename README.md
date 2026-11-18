# 🐿️ Squirrel Simulator

A behavioral simulation of squirrel foraging ecology on a 2D integer grid. The squirrel learns to cache nuts during the summer and retrieve them during the winter under strict movement and energy budgets, operating under a **strictly blind, stateless** constraint.

---

## 📋 Features & Constraints

- **2D Discrete Grid:** Infinite integer lattice $\mathbb{Z}^2$ with burrow/home at origin $(0, 0)$.
- **Strictly Blind Sensing:** The squirrel only senses whether a nut exists on the exact cell $(x, y)$ it occupies. No remote scent or vision.
- **Stateless Across Seasons:** The squirrel maintains no episodic memory of cache locations. Survival relies purely on the co-adaptation of its summer spatial dispersion policy and winter search trajectory.
- **Action Economy:**
  - Cardinal movement (North, South, East, West) costs 1 move.
  - Burying (summer) and digging/retrieving (winter) are free actions (0 moves).
  - Single-nut carrying capacity in summer ($P_{\text{find}} = 20\%$).
  - Conserving harvest: Squirrel leaves caches in the ground once daily quota is satisfied.
- **Survival Objective:**
  - Daily budget: $M$ moves (must return to $(0,0)$ every night).
  - Summer: 30 days to forage and cache nuts.
  - Winter: 30 days where the squirrel must retrieve at least $K$ nuts daily or starve.

---

## ⚙️ Difficulty Tiers (Existing Parameters Only)

| Parameter | Standard / Baseline | Hard Mode (`--hard`) | Notes |
| :--- | :--- | :--- | :--- |
| **Move Budget ($M$)** | 30 moves | **22 moves** | Halves reachable area ($R_{\text{max}} = 11$). |
| **Winter Quota ($K$)** | 1 nut / day | **2 nuts / day** | Doubles winter food demand (60 nuts required). |
| **Nut Discovery ($P_{\text{find}}$)** | 20% | 20% | Chance per unvisited cell in summer. |
| **Days (Summer / Winter)** | 30 / 30 | 30 / 30 | 60 days total lifecycle. |

---

## 🚀 Quickstart & Usage

### 1. Run Heuristic Baselines
Compare Random Walker against Sector Patrol (boustrophedon sweep):
```bash
# Standard mode
python3 squirrel_sim/main.py --mode baseline

# Hard mode (M=22, K=2)
python3 squirrel_sim/main.py --mode baseline --hard
```

### 2. Train the Base Squirrel Model
Train a neural policy using Neuroevolution / Genetic Algorithm directly optimizing winter survival:
```bash
# Standard mode
python3 squirrel_sim/main.py --mode train --generations 25 --pop-size 60 --save-path trained_squirrel.json

# Hard mode (M=22, K=2)
python3 squirrel_sim/main.py --mode train --hard --generations 25 --pop-size 60 --save-path trained_squirrel_hard.json
```

### 3. Visualize Daily Expeditions in ASCII
Replay animated journeys with trails, caches, and burrow:
```bash
# Replay standard squirrel
python3 squirrel_sim/main.py --mode demo --load-path trained_squirrel.json --seed 42

# Replay Hard Mode squirrel
python3 squirrel_sim/main.py --mode demo --hard --load-path trained_squirrel_hard.json --seed 2024
```
