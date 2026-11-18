"""Terminal ASCII / ANSI visualizer for the Squirrel Simulator.

Renders grid trails, cache distributions, and daily activity metrics.
"""
from typing import List, Tuple, Set
from squirrel_sim.env.squirrel_env import SquirrelEnv


class TerminalVisualizer:
    """Renders the 2D grid landscape and daily squirrel journeys in ASCII."""

    # ANSI Colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    RED = "\033[31m"

    def __init__(self, view_radius: int = 8):
        self.view_radius = view_radius

    def render_day(
        self,
        env: SquirrelEnv,
        trail: List[Tuple[int, int]],
        title: str = "Squirrel Daily Expedition",
    ) -> str:
        """Render a full day's movement trail and world state."""
        R = self.view_radius
        trail_set = set(trail)
        curr_pos = (env.x, env.y)

        lines = []
        lines.append(f"\n{self.BOLD}{self.CYAN}=== {title} ==={self.RESET}")
        season_color = self.YELLOW if env.season == "summer" else self.CYAN
        lines.append(
            f"Season: {season_color}{env.season.upper()}{self.RESET} | "
            f"Day: {env.current_day} | "
            f"Steps Used: {env.day_step}/{env.config.move_budget} | "
            f"Caches Remaining: {env.world.nut_count}"
        )
        if env.season == "winter":
            quota_status = (
                f"{self.GREEN}MET (Ate {env.nuts_retrieved_today}){self.RESET}"
                if env.nuts_retrieved_today >= env.config.daily_winter_quota
                else f"{self.RED}STARVING ({env.nuts_retrieved_today}/{env.config.daily_winter_quota}){self.RESET}"
            )
            lines.append(f"Daily Quota: {quota_status} | Winter Days Survived: {env.winter_days_survived}/{env.config.winter_days}")

        lines.append("+" + "-" * (2 * R + 1) * 2 + "+")

        for y in range(R, -R - 1, -1):
            row_chars = []
            for x in range(-R, R + 1):
                pos = (x, y)
                if pos == curr_pos:
                    # Current squirrel location
                    row_chars.append(f"{self.BOLD}{self.RED}S {self.RESET}")
                elif pos == (0, 0):
                    # Home burrow
                    row_chars.append(f"{self.BOLD}{self.BLUE}H {self.RESET}")
                elif pos in env.world.cached_nuts:
                    # Buried nut
                    row_chars.append(f"{self.YELLOW}*{self.RESET} ")
                elif pos in trail_set:
                    # Trail taken today
                    row_chars.append(f"{self.GREEN}·{self.RESET} ")
                else:
                    # Empty ground
                    row_chars.append(". ")
            lines.append("| " + "".join(row_chars) + "|")

        lines.append("+" + "-" * (2 * R + 1) * 2 + "+")
        lines.append(
            f"Legend: {self.BOLD}{self.BLUE}H{self.RESET}: Home | "
            f"{self.BOLD}{self.RED}S{self.RESET}: Squirrel | "
            f"{self.YELLOW}*{self.RESET}: Buried Nut | "
            f"{self.GREEN}·{self.RESET}: Day Trail | "
            f".: Empty Cell\n"
        )
        return "\n".join(lines)
