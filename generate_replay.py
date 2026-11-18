"""Generates a complete step-by-step replay log of every action of every day

Records the entire sensory state, action chosen, events (find, bury, dig, homing),
and active cache locations at every discrete step.
"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from squirrel_sim.env.squirrel_env import SquirrelEnv, SimConfig
from squirrel_sim.models.mlp import PolicyMLP


def record_full_replay(model_path: str = "trained_squirrel_hard.json", seed: int = 2024, output_path: str = "replay_trajectory.json"):
    cfg = SimConfig(move_budget=22, daily_winter_quota=2, p_find_summer=0.20, homing_reflex=True)
    env = SquirrelEnv(config=cfg, seed=seed)
    policy = PolicyMLP.load_json(model_path)

    obs = env.reset()
    action_names = {0: "North", 1: "South", 2: "East", 3: "West", 4: "Stay"}

    days_data = []
    current_day_steps = []
    cached_nuts_set = set()

    step_global = 0

    while not env.done:
        season = env.season
        day = env.current_day
        step_in_day = env.day_step
        x_before, y_before = env.x, env.y
        carrying_before = env.carrying_nut
        eaten_before = env.nuts_retrieved_today
        nuts_in_ground_before = set(env.world.cached_nuts)

        # Decide action
        action = policy.predict(obs, deterministic=True)
        act_name = action_names.get(action, "Stay")

        # Step environment
        obs, _, done, info = env.step(action)
        step_global += 1

        x_after, y_after = env.x, env.y
        carrying_after = env.carrying_nut
        eaten_after = env.nuts_retrieved_today
        nuts_in_ground_after = set(env.world.cached_nuts)

        # Determine event
        events = []
        if season == "summer":
            if not carrying_before and carrying_after:
                events.append("Found surface nut! (Picked up)")
            if carrying_before and not carrying_after:
                events.append(f"Buried nut at ({x_after}, {y_after})")
        elif season == "winter":
            if eaten_after > eaten_before:
                events.append(f"Dug up & ate cached nut at ({x_after}, {y_after})! ({eaten_after}/{cfg.daily_winter_quota})")

        if (x_after, y_after) == (0, 0) and step_in_day > 0:
            if season == "winter" and eaten_after >= cfg.daily_winter_quota:
                events.append("Returned safely to burrow (Quota Met)!")
            elif season == "summer":
                events.append("Returned safely to burrow!")

        step_record = {
            "step_num": step_in_day,
            "pos_from": [x_before, y_before],
            "pos_to": [x_after, y_after],
            "action": act_name,
            "action_id": action,
            "carrying_nut": carrying_after,
            "nuts_eaten_today": eaten_after,
            "quota_needed": cfg.daily_winter_quota,
            "moves_remaining": cfg.move_budget - env.day_step,
            "dist_home": abs(x_after) + abs(y_after),
            "safety_margin": (cfg.move_budget - env.day_step) - (abs(x_after) + abs(y_after)),
            "events": events,
            "active_caches": list(nuts_in_ground_after),
        }
        current_day_steps.append(step_record)

        # Check if day transitioned or simulation finished
        if env.day_step == 0 or env.done or env.current_day != day or env.season != season:
            days_data.append({
                "season": season,
                "day": day,
                "total_steps": len(current_day_steps),
                "nuts_buried_end_of_day": len(nuts_in_ground_after),
                "nuts_eaten_end_of_day": eaten_after,
                "survived_day": True if not env.done or env.status == "survived_winter" else False,
                "steps": current_day_steps
            })
            current_day_steps = []

    print(f"Recorded full replay: {len(days_data)} days, {step_global} total steps!")
    print(f"Final status: {env.status.upper()}, Winter Days Survived: {env.winter_days_survived}")

    output_payload = {
        "config": {
            "move_budget": cfg.move_budget,
            "daily_winter_quota": cfg.daily_winter_quota,
            "summer_days": cfg.summer_days,
            "winter_days": cfg.winter_days,
            "seed": seed,
            "final_status": env.status,
            "total_summer_buried": env.world.total_buried_summer,
            "total_winter_retrieved": env.world.total_retrieved_winter,
        },
        "days": days_data
    }

    with open(output_path, "w") as f:
        json.dump(output_payload, f)
    print(f"Saved replay trajectory to {output_path}")
    return output_payload


if __name__ == "__main__":
    record_full_replay()
