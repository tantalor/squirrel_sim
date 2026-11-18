"""Squirrel Simulation Environment.

Simulates the summer caching and winter retrieval phases with daily movement
budgets, energy invariants, and survival checks under strictly blind perception.
"""
import random
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from squirrel_sim.env.grid_world import GridWorld


@dataclass
class SimConfig:
    move_budget: int = 30
    summer_days: int = 30
    winter_days: int = 30
    p_find_summer: float = 0.20
    daily_winter_quota: int = 1
    homing_reflex: bool = True  # Emergency homing reflex when remaining moves == distance to home


class SquirrelEnv:
    """Environment simulating seasonal squirrel foraging on a 2D integer grid."""

    # Cardinal actions
    ACTION_NORTH = 0
    ACTION_SOUTH = 1
    ACTION_EAST = 2
    ACTION_WEST = 3
    ACTION_STAY = 4

    ACTION_DELTAS = {
        0: (0, 1),   # North
        1: (0, -1),  # South
        2: (1, 0),   # East
        3: (-1, 0),  # West
        4: (0, 0),   # Stay / Rest
    }

    def __init__(self, config: Optional[SimConfig] = None, seed: Optional[int] = None):
        self.config = config or SimConfig()
        self.rng = random.Random(seed)
        self.world = GridWorld(home_pos=(0, 0))

        # Max safe radius from home: floor(M / 2)
        self.max_reach = self.config.move_budget // 2

        # Seasonal state
        self.season = "summer"  # "summer" or "winter"
        self.current_day = 1
        self.day_step = 0
        self.x = 0
        self.y = 0

        # Day inventory & progress
        self.carrying_nut = False
        self.nuts_retrieved_today = 0
        self.winter_days_survived = 0

        # Daily trajectory recording for visualization
        self.daily_trail: List[Tuple[int, int]] = [(0, 0)]

        # Simulation lifecycle status
        # status: "running", "survived_winter", "stranded", "starved"
        self.status = "running"
        # self.done indicates whether the ENTIRE seasonal simulation has terminated
        # (e.g. the squirrel died of starvation/stranding, or completed all winter days).
        # It is NOT done for the day; a daily sunset simply transitions to the next day.
        self.done = False

    def seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng.seed(seed)

    def reset(self, seed: Optional[int] = None) -> List[float]:
        """Reset environment for a new full season simulation."""
        if seed is not None:
            self.seed(seed)

        self.world.reset()
        self.season = "summer"
        self.current_day = 1
        self.day_step = 0
        self.x = 0
        self.y = 0
        self.carrying_nut = False
        self.nuts_retrieved_today = 0
        self.winter_days_survived = 0
        self.daily_trail = [(0, 0)]
        self.status = "running"
        # Episode-level done flag (True only when simulation ends in death or winter victory)
        self.done = False

        return self.get_observation()

    def get_observation(self) -> List[float]:
        """Construct the normalized 12-dimensional observation vector for the agent.

        WHAT THIS REPRESENTS:
            The squirrel's momentary, local snapshot of the world and its own internal
            physiological status at this exact time step t. Because the squirrel is
            strictly blind and stateless across seasons, it has NO remote vision, NO scent
            gradient, and NO persistent map of cache coordinates. It only perceives:
            - Its current relative location and distance to its home burrow (0, 0).
            - Its remaining energy budget today and its safe return margin.
            - What is directly under its paws right now (contact nut detection).
            - Its physical inventory (is it carrying a nut in its paws?).
            - Its biological calendar and hunger state (season, daily quota met, and
              internal circadian/seasonal day progression).

        WHAT THIS IS FOR:
            This 12-element floating-point vector is the DIRECT INPUT to the neural
            network policy (PolicyMLP) at every step. The model computes a forward pass
            over this vector to output logits that select the next movement action
            (North, South, East, West, or Stay).

        FEATURE BREAKDOWN:
            1. x / max_reach: Normalized horizontal offset from home burrow [-1.0, 1.0].
            2. y / max_reach: Normalized vertical offset from home burrow [-1.0, 1.0].
            3. (M - day_step) / M: Fraction of daily movement budget remaining [0.0, 1.0].
            4. (|x| + |y|) / max_reach: Normalized Manhattan distance to home burrow [0.0, 1.0].
            5. Safety Margin: [(M - day_step) - (|x| + |y|)] / M. The buffer of spare moves
               available before the squirrel is forced to turn around and head home.
            6. Nut on Cell: Contact sensor (1.0 if a nut is buried directly here, 0.0 otherwise).
            7. Carrying Nut: 1.0 if currently holding a nut in paws (summer only), 0.0 otherwise.
            8. Is Winter: 1.0 during winter retrieval phase, 0.0 during summer caching phase.
            9. Quota Met: 1.0 if daily food requirement has been satisfied, 0.0 if still hungry.
            10. Day Progress: Linear fraction through season (current_day - 1) / total_days [0.0, 1.0].
            11. sin(2*pi * day_progress): Cyclic circadian day indicator (Y-component on calendar clock).
            12. cos(2*pi * day_progress): Cyclic circadian day indicator (X-component on calendar clock).
        """
        remaining_moves = self.config.move_budget - self.day_step
        dist_home = abs(self.x) + abs(self.y)
        safety_margin = remaining_moves - dist_home
        nut_on_cell = 1.0 if self.world.has_nut(self.x, self.y) else 0.0
        quota_progress = min(1.0, self.nuts_retrieved_today / max(1, self.config.daily_winter_quota))

        total_days = self.config.winter_days if self.season == "winter" else self.config.summer_days
        day_prog = (self.current_day - 1) / max(1, total_days)
        angle = 2.0 * math.pi * day_prog

        return [
            self.x / max(1, self.max_reach),
            self.y / max(1, self.max_reach),
            remaining_moves / self.config.move_budget,
            dist_home / max(1, self.max_reach),
            safety_margin / self.config.move_budget,
            nut_on_cell,
            1.0 if self.carrying_nut else 0.0,
            1.0 if self.season == "winter" else 0.0,
            quota_progress,
            day_prog,
            math.sin(angle),
            math.cos(angle),
        ]

    def _apply_free_actions(self):
        """Perform automatic free actions (nut discovery, burying, retrieving)."""
        curr_pos = (self.x, self.y)

        if self.season == "summer":
            # If hands are empty and not at home burrow, roll for surface discovery
            if not self.carrying_nut and curr_pos != (0, 0):
                if self.rng.random() < self.config.p_find_summer:
                    self.carrying_nut = True

            # If carrying a nut and current cell is empty (and not home), bury it
            if self.carrying_nut and curr_pos != (0, 0):
                if not self.world.has_nut(self.x, self.y):
                    self.world.bury_nut(self.x, self.y)
                    self.carrying_nut = False

        elif self.season == "winter":
            # If current cell has a buried nut and daily quota is not yet met, dig it up
            if self.nuts_retrieved_today < self.config.daily_winter_quota:
                if self.world.has_nut(self.x, self.y):
                    self.world.retrieve_nut(self.x, self.y)
                    self.nuts_retrieved_today += 1

    def _emergency_homing_action(self) -> int:
        """Instinctive step toward (0, 0) when margin is critical."""
        if self.x > 0:
            return self.ACTION_WEST
        elif self.x < 0:
            return self.ACTION_EAST
        elif self.y > 0:
            return self.ACTION_SOUTH
        elif self.y < 0:
            return self.ACTION_NORTH
        return self.ACTION_STAY

    def step(self, action: int) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        """Advance simulation by one step.

        Returns:
            obs: Observation vector
            reward: Scalar reward (optional for RL; GA uses fitness)
            done: Whether the episode has terminated
            info: Metadata dictionary
        """
        if self.done:
            return self.get_observation(), 0.0, True, {"status": self.status}

        dist_home = abs(self.x) + abs(self.y)
        remaining_moves = self.config.move_budget - self.day_step

        # Biological homing reflex:
        # 1. Emergency return when remaining moves only suffice to walk home.
        # 2. Satiation return: once daily winter food quota is met, return home to rest.
        if self.config.homing_reflex:
            is_full_in_winter = (
                self.season == "winter"
                and self.nuts_retrieved_today >= self.config.daily_winter_quota
            )
            is_time_up = (remaining_moves <= dist_home)

            if (is_time_up or is_full_in_winter) and (self.x, self.y) != (0, 0):
                action = self._emergency_homing_action()
            elif is_full_in_winter and (self.x, self.y) == (0, 0):
                action = self.ACTION_STAY

        # Handle STAY / Rest action
        if action == self.ACTION_STAY:
            if (self.x, self.y) == (0, 0):
                # Squirrel is safely at home and decides to conclude the day early
                self.day_step = self.config.move_budget
            else:
                # Resting outside consumes 1 move of time
                self.day_step += 1
        else:
            dx, dy = self.ACTION_DELTAS.get(action, (0, 0))
            self.x += dx
            self.y += dy
            self.day_step += 1
            self.daily_trail.append((self.x, self.y))

            # Trigger free actions (discovery, burying, eating)
            self._apply_free_actions()

        # Check return invariant: can squirrel still reach (0, 0)?
        dist_home = abs(self.x) + abs(self.y)
        remaining_moves = self.config.move_budget - self.day_step

        if dist_home > remaining_moves:
            # Cannot reach home before sunset - stranded outside!
            self.status = "stranded"
            self.done = True
            return self.get_observation(), -10.0, True, self._make_info()

        # Check if day has ended
        if self.day_step >= self.config.move_budget:
            self._end_of_day_transition()

        return self.get_observation(), 0.0, self.done, self._make_info()

    def _end_of_day_transition(self):
        """Evaluate day conclusion and transition to next day or season."""
        # Must be at home burrow
        if (self.x, self.y) != (0, 0):
            self.status = "stranded"
            self.done = True
            return

        if self.season == "summer":
            # Summer day completed successfully
            if self.current_day >= self.config.summer_days:
                # Summer ends, winter begins!
                self.season = "winter"
                self.current_day = 1
            else:
                self.current_day += 1

            # Prepare for next day
            self.day_step = 0
            self.carrying_nut = False
            self.daily_trail = [(0, 0)]

        elif self.season == "winter":
            # Check winter daily survival quota
            if self.nuts_retrieved_today >= self.config.daily_winter_quota:
                self.winter_days_survived += 1
                if self.winter_days_survived >= self.config.winter_days:
                    # Survived the whole winter!
                    self.status = "survived_winter"
                    self.done = True
                    return
                else:
                    self.current_day += 1
                    self.day_step = 0
                    self.nuts_retrieved_today = 0
                    self.daily_trail = [(0, 0)]
            else:
                # Failed to meet daily quota - starvation
                self.status = "starved"
                self.done = True

    def _make_info(self) -> Dict[str, Any]:
        """Construct diagnostic metadata dictionary."""
        return {
            "status": self.status,
            "season": self.season,
            "current_day": self.current_day,
            "day_step": self.day_step,
            "winter_days_survived": self.winter_days_survived,
            "cached_nuts_remaining": self.world.nut_count,
            "total_buried_summer": self.world.total_buried_summer,
            "total_retrieved_winter": self.world.total_retrieved_winter,
            "pos": (self.x, self.y),
        }
