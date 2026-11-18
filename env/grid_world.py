"""Grid world representation for the Squirrel Simulator.

Manages the 2D integer lattice, cache occupancy, and spatial queries.
"""
from typing import Set, Tuple, Optional, Dict, Any


class GridWorld:
    """Represents the 2D discrete landscape where nuts can be cached."""

    def __init__(self, home_pos: Tuple[int, int] = (0, 0)):
        self.home_pos = home_pos
        # Set of (x, y) coordinates holding a buried nut
        self.cached_nuts: Set[Tuple[int, int]] = set()
        # History of caches created for analytics
        self.total_buried_summer = 0
        self.total_retrieved_winter = 0

    def reset(self):
        """Reset world to initial state with no buried nuts."""
        self.cached_nuts.clear()
        self.total_buried_summer = 0
        self.total_retrieved_winter = 0

    def has_nut(self, x: int, y: int) -> bool:
        """Check if cell (x, y) contains a buried nut."""
        return (x, y) in self.cached_nuts

    def bury_nut(self, x: int, y: int) -> bool:
        """Bury a nut at (x, y).

        Returns True if successful, False if cell is already occupied or is home.
        """
        # Home cell cannot hold a buried nut to keep the burrow clear
        if (x, y) == self.home_pos:
            return False
        if (x, y) in self.cached_nuts:
            return False
        self.cached_nuts.add((x, y))
        self.total_buried_summer += 1
        return True

    def retrieve_nut(self, x: int, y: int) -> bool:
        """Dig up and retrieve a nut from (x, y).

        Returns True if a nut was retrieved, False if empty.
        """
        if (x, y) in self.cached_nuts:
            self.cached_nuts.remove((x, y))
            self.total_retrieved_winter += 1
            return True
        return False

    @property
    def nut_count(self) -> int:
        """Number of active buried nuts currently in the ground."""
        return len(self.cached_nuts)

    @staticmethod
    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int] = (0, 0)) -> int:
        """Calculate L1 Manhattan distance between two coordinates."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def get_summary(self) -> Dict[str, Any]:
        """Return stats on the current world state."""
        return {
            "current_buried_nuts": len(self.cached_nuts),
            "total_buried_summer": self.total_buried_summer,
            "total_retrieved_winter": self.total_retrieved_winter,
        }
