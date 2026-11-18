"""Main entry point for Squirrel Simulator.

Supports:
1. Baseline benchmark (Random Walker vs. Sector Patrol)
2. Neuroevolution training of the Base Squirrel Model
3. Demo / visualization of a trained squirrel across summer and winter
4. Replay: record full step-by-step trajectory and export a self-contained HTML viewer

Usage examples:
  python3 squirrel_sim/main.py --mode baseline
  python3 squirrel_sim/main.py --mode train --hard --save-path trained_squirrel_hard.json
  python3 squirrel_sim/main.py --mode demo --hard --load-path trained_squirrel_hard.json --seed 2024
  python3 squirrel_sim/main.py --mode replay --hard --load-path trained_squirrel_hard.json --seed 2024 --html-out squirrel_replay.html
"""
import sys
import os

# Ensure package root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import time

from squirrel_sim.env.squirrel_env import SquirrelEnv, SimConfig
from squirrel_sim.models.mlp import PolicyMLP
from squirrel_sim.models.heuristics import RandomSquirrel, SectorPatrolSquirrel
from squirrel_sim.train.genetic_algo import GeneticTrainer, GAConfig
from squirrel_sim.viz.terminal_viz import TerminalVisualizer


def run_heuristic_baseline(sim_config: SimConfig = None, env_seed: int = 42):
    """Evaluate heuristic agents to establish benchmark performance."""
    config = sim_config or SimConfig(move_budget=30, summer_days=30, winter_days=30, p_find_summer=0.20)
    print(f"\n{'='*65}")
    print(f"🌲 RUNNING HEURISTIC BASELINES (M={config.move_budget}, Quota={config.daily_winter_quota}, P={config.p_find_summer:.2f})")
    print(f"{'='*65}")

    viz = TerminalVisualizer(view_radius=8)

    # 1. Random Walker
    print("\n--- 1. Random Walker ---")
    env = SquirrelEnv(config=config, seed=env_seed)
    random_agent = RandomSquirrel(seed=env_seed)
    obs = env.reset()

    while not env.done:
        margin = config.move_budget - env.day_step - (abs(env.x) + abs(env.y))
        action = random_agent.select_action(obs, env.x, env.y, margin)
        obs, _, done, info = env.step(action)

    print(f"Status: {env.status.upper()}")
    print(f"Summer Nuts Buried: {env.world.total_buried_summer}")
    print(f"Winter Days Survived: {env.winter_days_survived}/{config.winter_days}")

    # 2. Sector Patrol Agent
    print("\n--- 2. Sector Patrol Agent (Heuristic Forager) ---")
    env = SquirrelEnv(config=config, seed=env_seed)
    sector_agent = SectorPatrolSquirrel()
    obs = env.reset()

    while not env.done:
        margin = config.move_budget - env.day_step - (abs(env.x) + abs(env.y))
        action = sector_agent.select_action(obs, env.x, env.y, margin, env.current_day)
        obs, _, done, info = env.step(action)

    print(f"Status: {env.status.upper()}")
    print(f"Summer Nuts Buried: {env.world.total_buried_summer}")
    print(f"Winter Days Survived: {env.winter_days_survived}/{config.winter_days}")
    print(f"Winter Nuts Retrieved: {env.world.total_retrieved_winter}")


def run_training(
    sim_config: SimConfig = None,
    generations: int = 40,
    pop_size: int = 60,
    save_path: str = "trained_squirrel.json"
) -> PolicyMLP:
    """Train the base squirrel model using Neuroevolution / GA."""
    sim_cfg = sim_config or SimConfig(
        move_budget=30,
        summer_days=30,
        winter_days=30,
        p_find_summer=0.20,
        daily_winter_quota=1,
        homing_reflex=True,
    )
    ga_cfg = GAConfig(
        pop_size=pop_size,
        generations=generations,
        elite_count=max(2, pop_size // 10),
        mutation_rate=0.15,
        mutation_sigma=0.08,
        crossover_rate=0.5,
        eval_seeds=2,
    )

    trainer = GeneticTrainer(sim_config=sim_cfg, ga_config=ga_cfg)
    best_model = trainer.train()

    if best_model:
        best_model.save_json(save_path)
        print(f"\n💾 Saved best squirrel model to {save_path}")

    return best_model


def run_demo(
    sim_config: SimConfig = None,
    model_path: str = "trained_squirrel.json",
    seed: int = 42,
    max_rendered_days: int = 5
):
    """Visualize a squirrel model running day by day with ASCII animation."""
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found! Run with --train first.")
        return

    policy = PolicyMLP.load_json(model_path)
    config = sim_config or SimConfig(move_budget=30, summer_days=30, winter_days=30, p_find_summer=0.20)
    env = SquirrelEnv(config=config, seed=seed)
    viz = TerminalVisualizer(view_radius=8)

    print(f"\n{'='*65}")
    print(f"🐿️  VISUALIZING SQUIRREL EXPEDITIONS (Seed {seed})")
    print(f"{'='*65}\n")

    obs = env.reset()
    days_shown = 0

    while not env.done:
        curr_day = env.current_day
        curr_season = env.season

        while not env.done and env.current_day == curr_day and env.season == curr_season:
            action = policy.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)

        if days_shown < max_rendered_days or env.done:
            title = f"{curr_season.capitalize()} Day {curr_day}"
            print(viz.render_day(env, env.daily_trail, title=title))
            days_shown += 1

    print(f"\n--- Expedition Result ---")
    print(f"Final Status: {env.status.upper()}")
    print(f"Summer Caches Created: {env.world.total_buried_summer}")
    print(f"Winter Days Survived: {env.winter_days_survived}/{config.winter_days}")
    print(f"Winter Nuts Retrieved: {env.world.total_retrieved_winter}")


def run_replay(
    sim_config: SimConfig = None,
    model_path: str = "trained_squirrel.json",
    seed: int = 42,
    html_out: str = None,
    html_template: str = None,
    agent_type: str = "mlp",
):
    """Record a full step-by-step trajectory and bake it into a self-contained HTML viewer.

    The output is a single .html file with all replay data inlined — open it
    in any browser with no server required.
    """
    config = sim_config or SimConfig(
        move_budget=30, daily_winter_quota=1, p_find_summer=0.20, homing_reflex=True
    )

    if agent_type == "mlp":
        if not os.path.exists(model_path):
            print(f"Model file '{model_path}' not found! Run with --mode train first.")
            return
        agent = PolicyMLP.load_json(model_path)
        label = model_path
    elif agent_type == "sector":
        agent = SectorPatrolSquirrel()
        label = "SectorPatrol"
    elif agent_type == "random":
        agent = RandomSquirrel(seed=seed)
        label = "RandomWalker"
    else:
        print(f"Unknown agent_type: {agent_type}")
        return

    env = SquirrelEnv(config=config, seed=seed)
    obs = env.reset()

    action_letters = {0: "N", 1: "S", 2: "E", 3: "W", 4: "St"}

    days_data = []
    current_day_steps = []
    # Track the day/season at the START of each step so we know when a boundary occurs
    prev_day = env.current_day
    prev_season = env.season

    print(f"\n🎥 Recording replay (agent={label}, seed={seed})...")

    while not env.done:
        step_in_day = env.day_step
        x_before, y_before = env.x, env.y
        carrying_before = env.carrying_nut
        eaten_before = env.nuts_retrieved_today
        nuts_before = set(env.world.cached_nuts)

        margin = config.move_budget - env.day_step - (abs(env.x) + abs(env.y))
        if agent_type == "mlp":
            action = agent.predict(obs, deterministic=True)
        elif agent_type == "sector":
            action = agent.select_action(obs, env.x, env.y, margin, env.current_day)
        else:
            action = agent.select_action(obs, env.x, env.y, margin)
        obs, _, done, info = env.step(action)

        x_after, y_after = env.x, env.y
        carrying_after = env.carrying_nut
        eaten_after = env.nuts_retrieved_today
        nuts_after = set(env.world.cached_nuts)

        added = [list(c) for c in nuts_after - nuts_before]
        removed = [list(c) for c in nuts_before - nuts_after]

        # Compute actual eaten count on this step (immune to end-of-day reset)
        eaten_this_step = eaten_before + len(removed) if prev_season == "winter" else eaten_after

        events = []
        if prev_season == "summer":
            if added:
                for bx, by in added:
                    if not carrying_before:
                        events.append(f"Found & buried nut at ({bx}, {by})")
                    else:
                        events.append(f"Buried nut at ({bx}, {by})")
            elif not carrying_before and carrying_after:
                events.append("Found surface nut! (Picked up)")
        elif prev_season == "winter":
            if removed:
                for rx, ry in removed:
                    events.append(
                        f"Dug up & ate cached nut at ({rx}, {ry})!"
                        f" ({eaten_this_step}/{config.daily_winter_quota})"
                    )
        if (x_after, y_after) == (0, 0) and (x_before, y_before) != (0, 0):
            if prev_season == "winter" and eaten_this_step >= config.daily_winter_quota:
                events.append("Returned safely to burrow (Quota Met)!")
            elif prev_season == "summer":
                events.append("Returned safely to burrow!")
        if env.done:
            if env.status == "starved":
                events.append(f"Starved :( ({eaten_this_step}/{config.daily_winter_quota} nuts)")
            elif env.status == "stranded":
                events.append("Stranded outside burrow :(")
            elif env.status == "survived_winter":
                events.append("Survived the entire winter!")

        current_day_steps.append({
            "n": step_in_day,
            "p": [x_after, y_after],
            "a": action_letters[action],
            "c": int(carrying_after),
            "e": eaten_this_step,
            "m": config.move_budget - env.day_step,
            "ev": events,
            "ca": added,
            "cr": removed,
        })

        # Detect end of a day: day or season changed, or simulation finished
        new_day = env.current_day
        new_season = env.season
        if new_day != prev_day or new_season != prev_season or env.done:
            days_data.append({
                "s": prev_season[0],       # "s" or "w"
                "d": prev_day,
                "nc": env.world.nut_count,
                "ne": eaten_after,
                "st": current_day_steps,
            })
            current_day_steps = []

        prev_day = new_day
        prev_season = new_season

    replay_data = {
        "config": {
            "move_budget": config.move_budget,
            "daily_winter_quota": config.daily_winter_quota,
            "p_find_summer": config.p_find_summer,
            "summer_days": config.summer_days,
            "winter_days": config.winter_days,
            "seed": seed,
            "final_status": env.status,
            "total_summer_buried": env.world.total_buried_summer,
            "total_winter_retrieved": env.world.total_retrieved_winter,
        },
        "days": days_data,
    }

    total_steps = sum(len(d["st"]) for d in days_data)
    print(f"✅ Recorded {len(days_data)} days, {total_steps} total steps")
    print(f"   Status: {env.status.upper()}, Winter Days Survived: {env.winter_days_survived}")

    # --- Determine output HTML path ---
    if html_out is None:
        html_out = model_path.replace(".json", "_replay.html")

    # --- Find the HTML template ---
    # Search order: explicit arg → squirrel_sim/ package dir → artifact dir (fallback)
    template_path = html_template
    if template_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "squirrel_replay_template.html"),
        ]
        for c in candidates:
            if os.path.exists(c):
                template_path = c
                break

    if template_path is None or not os.path.exists(template_path):
        json_out = html_out.replace(".html", ".json")
        print(f"⚠️  No HTML template found at {template_path!r}.")
        print(f"   Saving raw JSON to {json_out} instead.")
        with open(json_out, "w") as f:
            json.dump(replay_data, f, separators=(",", ":"))
        return

    with open(template_path) as f:
        html = f.read()

    json_str = json.dumps(replay_data, separators=(",", ":"))

    # The template contains a line like:  RAW = {...};\ninitMinimap();
    # We replace everything from "RAW = {" up to the closing ";\n" before "initMinimap".
    START = "RAW = {"
    AFTER = ";\ninitMinimap();"
    if START in html and AFTER in html:
        start_idx = html.index(START)
        end_idx = html.index(AFTER, start_idx)       # keep the trailing ";\ninitMinimap();"
        html = html[:start_idx] + "RAW = " + json_str + html[end_idx:]
    else:
        print("⚠️  Could not find data-injection markers in template HTML.")
        print("   The template may need to be regenerated from the artifact directory.")
        return

    with open(html_out, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(html_out) // 1024
    print(f"🌐 Saved self-contained HTML viewer → {html_out}  ({size_kb} KB)")
    print(f"   Open it in any browser — no server required.")


def main():
    parser = argparse.ArgumentParser(description="Squirrel Simulator")
    parser.add_argument(
        "--mode",
        choices=["baseline", "train", "demo", "replay"],
        default="baseline",
        help="baseline | train | demo | replay",
    )
    parser.add_argument("--hard", action="store_true", help="Enable Hard Mode (M=22 moves, K=2 nuts/day, P=0.20)")
    parser.add_argument("--move-budget", type=int, default=None, help="Daily Manhattan move budget M")
    parser.add_argument("--winter-quota", type=int, default=None, help="Daily nuts needed to survive K")
    parser.add_argument("--p-find", type=float, default=None, help="Summer surface nut discovery chance")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--pop-size", type=int, default=60)
    parser.add_argument("--save-path", type=str, default="trained_squirrel.json")
    parser.add_argument("--load-path", type=str, default="trained_squirrel.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent", choices=["mlp", "sector", "random"], default="mlp", help="Agent for replay mode")
    parser.add_argument("--html-out", type=str, default=None, help="Output path for self-contained HTML replay viewer")

    args = parser.parse_args()

    # Determine configuration
    if args.hard:
        m = 22 if args.move_budget is None else args.move_budget
        k = 2 if args.winter_quota is None else args.winter_quota
        p = 0.20 if args.p_find is None else args.p_find
    else:
        m = 30 if args.move_budget is None else args.move_budget
        k = 1 if args.winter_quota is None else args.winter_quota
        p = 0.20 if args.p_find is None else args.p_find

    sim_cfg = SimConfig(
        move_budget=m,
        summer_days=30,
        winter_days=30,
        p_find_summer=p,
        daily_winter_quota=k,
        homing_reflex=True,
    )

    if args.mode == "baseline":
        run_heuristic_baseline(sim_config=sim_cfg, env_seed=args.seed)
    elif args.mode == "train":
        run_training(
            sim_config=sim_cfg,
            generations=args.generations,
            pop_size=args.pop_size,
            save_path=args.save_path
        )
    elif args.mode == "demo":
        run_demo(sim_config=sim_cfg, model_path=args.load_path, seed=args.seed)
    elif args.mode == "replay":
        run_replay(
            sim_config=sim_cfg,
            model_path=args.load_path,
            seed=args.seed,
            html_out=args.html_out,
            agent_type=args.agent,
        )


if __name__ == "__main__":
    main()
