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
        # Signs for the 4 quadrants: (East/North, East/South, West/South, West/North)
        self.quadrant_signs = [
            (1, 1),
            (1, -1),
            (-1, -1),
            (-1, 1),
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

        # Determine active quadrant for today
        quadrant = day % 4
        sx, sy = self.quadrant_signs[quadrant]

        # Convert to local quadrant coordinates (u >= 0, v >= 0)
        u = x * sx
        v = y * sy

        # Boustrophedon 2D raster sweep starting on row v = 0 (y = 0):
        # - On even rows (v = 0, 2, 4...): sweep outward along +u until u == sweep_width, then step +v
        # - On odd rows  (v = 1, 3, 5...): sweep inward along -u until u == 0, then step +v
        if v % 2 == 0:
            if u < self.sweep_width:
                return SquirrelEnv.ACTION_EAST if sx > 0 else SquirrelEnv.ACTION_WEST
            else:
                return SquirrelEnv.ACTION_NORTH if sy > 0 else SquirrelEnv.ACTION_SOUTH
        else:
            if u > 0:
                return SquirrelEnv.ACTION_WEST if sx > 0 else SquirrelEnv.ACTION_EAST
            else:
                return SquirrelEnv.ACTION_NORTH if sy > 0 else SquirrelEnv.ACTION_SOUTH
