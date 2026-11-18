"""Heuristic baseline agents for comparison with learned neural policies.

Provides:
1. RandomWalker: Wanders randomly with homing awareness.
2. SectorPatrolAgent: Explores systematically by sectors and returns home once satisfied.
"""
import random
from typing import List, Tuple
from squirrel_sim.env.squirrel_env import SquirrelEnv


def get_homing_action(x: int, y: int) -> int:
    """Return cardinal step toward (0,0)."""
    if abs(x) > abs(y):
        return SquirrelEnv.ACTION_WEST if x > 0 else SquirrelEnv.ACTION_EAST
    elif y != 0:
        return SquirrelEnv.ACTION_SOUTH if y > 0 else SquirrelEnv.ACTION_NORTH
    elif x != 0:
        return SquirrelEnv.ACTION_WEST if x > 0 else SquirrelEnv.ACTION_EAST
    return SquirrelEnv.ACTION_STAY


class RandomSquirrel:
    """Randomly wanders; heads directly home when safety margin is low."""

    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)

    def select_action(self, obs: List[float], x: int, y: int, margin: int) -> int:
        # If already at home in winter and ate, stay
        quota_met = (obs[8] >= 1.0 - 1e-6)
        is_winter = (obs[7] > 0.5)

        if is_winter and quota_met:
            if (x, y) == (0, 0):
                return SquirrelEnv.ACTION_STAY
            return get_homing_action(x, y)

        # If margin is low, must head home
        if margin <= 1 and (x, y) != (0, 0):
            return get_homing_action(x, y)

        # Otherwise pick random move
        return self.rng.choice([
            SquirrelEnv.ACTION_NORTH,
            SquirrelEnv.ACTION_SOUTH,
            SquirrelEnv.ACTION_EAST,
            SquirrelEnv.ACTION_WEST,
        ])


class SectorPatrolSquirrel:
    """Systematic heuristic baseline that explores in rotating quadrant sweeps.

    Instead of marching along a single 1D diagonal ray, it executes a 2D boustrophedon
    (lawnmower) raster sweep back and forth across the assigned quadrant, maximizing
    2D surface area coverage (25+ unique cells visited per day) before heading home.

    - In summer: Sweeps the quadrant back and forth to disperse nuts across a 2D patch.
    - In winter: Sweeps the quadrant back and forth until 1 nut is eaten, then takes
      the shortest path home and rests.
    """

    def __init__(self, sweep_width: int = 4):
        self.sweep_width = sweep_width
        # 4 strictly disjoint 90°-rotated quadrants partitioning Z^2 \ {(0,0)}:
        # Each quadrant owns (p >= 1, q >= 0) where p is its primary outward axis
        # and q is its secondary axis, so each axis (+x, +y, -x, -y) is owned by
        # exactly one quadrant with zero double-walking.
        # Format: (p_dir_vec, q_dir_vec, action_+p, action_-p, action_+q)
        self.quadrants = [
            ((1, 0),  (0, 1),  SquirrelEnv.ACTION_EAST,  SquirrelEnv.ACTION_WEST,  SquirrelEnv.ACTION_NORTH),  # Q0: x>0, y>=0
            ((0, 1),  (-1, 0), SquirrelEnv.ACTION_NORTH, SquirrelEnv.ACTION_SOUTH, SquirrelEnv.ACTION_WEST),   # Q1: x<=0, y>0
            ((-1, 0), (0, -1), SquirrelEnv.ACTION_WEST,  SquirrelEnv.ACTION_EAST,  SquirrelEnv.ACTION_SOUTH),  # Q2: x<0, y<=0
            ((0, -1), (1, 0),  SquirrelEnv.ACTION_SOUTH, SquirrelEnv.ACTION_NORTH, SquirrelEnv.ACTION_EAST),   # Q3: x>=0, y<0
        ]

    def select_action(self, obs: List[float], x: int, y: int, margin: int, day: int) -> int:
        is_winter = (obs[7] > 0.5)
        quota_met = (obs[8] >= 1.0 - 1e-6)

        # Winter rule: If quota met, immediately return home and stay
        if is_winter and quota_met:
            if (x, y) == (0, 0):
                return SquirrelEnv.ACTION_STAY
            return get_homing_action(x, y)

        # Safety rule: If remaining moves only suffice to return home, head home
        if margin <= 0:
            if (x, y) == (0, 0):
                return SquirrelEnv.ACTION_STAY
            return get_homing_action(x, y)

        # Active 90°-rotated disjoint quadrant for today
        (px, py), (qx, qy), act_pos_p, act_neg_p, act_pos_q = self.quadrants[day % 4]

        # Local quadrant coordinates (p >= 1, q >= 0 inside quadrant; (0,0) at home)
        p = x * px + y * py
        q = x * qx + y * qy

        # Boustrophedon 2D raster sweep over p in [1..sweep_width], q = 0, 1, 2...
        # - At (0, 0): p=0, q=0 -> steps +p onto (1, 0) (the quadrant's owned axis)
        # - Even rows (q = 0, 2...): sweep outward along +p until sweep_width, then step +q
        # - Odd rows  (q = 1, 3...): sweep inward along -p until p == 1, then step +q
        if q % 2 == 0:
            if p < self.sweep_width:
                return act_pos_p
            return act_pos_q
        else:
            if p > 1:
                return act_neg_p
            return act_pos_q
