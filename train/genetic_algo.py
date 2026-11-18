"""Genetic Algorithm / Neuroevolution for Squirrel Survival Optimization.

Evolves policy MLP weights using tournament selection, uniform crossover,
and Gaussian mutation based on winter survival days.
"""
import random
import time
import json
import csv
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from squirrel_sim.env.squirrel_env import SquirrelEnv, SimConfig
from squirrel_sim.models.mlp import PolicyMLP


@dataclass
class GAConfig:
    pop_size: int = 60
    generations: int = 50
    elite_count: int = 6
    tournament_size: int = 3
    mutation_rate: float = 0.15      # Probability of mutating a given weight
    mutation_sigma: float = 0.08     # Standard deviation of Gaussian mutation
    crossover_rate: float = 0.50     # Probability of performing crossover
    eval_seeds: int = 2              # Seeds per evaluation to reduce variance
    max_steps_per_day: int = 30


class GeneticTrainer:
    """Manages the evolutionary training loop for squirrel survival."""

    def __init__(self, sim_config: Optional[SimConfig] = None, ga_config: Optional[GAConfig] = None):
        self.sim_config = sim_config or SimConfig()
        self.ga_config = ga_config or GAConfig()

        # Initialize population of PolicyMLPs
        self.population: List[PolicyMLP] = [
            PolicyMLP(layer_sizes=(12, 20, 20, 5)) for _ in range(self.ga_config.pop_size)
        ]

        self.best_individual: Optional[PolicyMLP] = None
        self.best_fitness: float = -1e9
        self.best_survived_days: int = 0
        self.history: List[Dict[str, Any]] = []

    def evaluate_policy(self, policy: PolicyMLP, seed_base: int = 1000) -> Tuple[float, int, int]:
        """Evaluate a single policy across multiple random seeds.

        Returns:
            avg_fitness: Combined fitness score
            avg_survived: Average winter days survived
            avg_buried: Average summer nuts buried
        """
        total_fitness = 0.0
        total_survived = 0
        total_buried = 0
        total_retrieved = 0

        for s in range(self.ga_config.eval_seeds):
            seed = seed_base + s * 1337
            env = SquirrelEnv(config=self.sim_config, seed=seed)
            obs = env.reset()

            step_count = 0
            max_total_steps = (self.sim_config.summer_days + self.sim_config.winter_days) * (self.sim_config.move_budget + 2)
            visited_summer_cells = set()

            while not env.done and step_count < max_total_steps:
                was_summer = (env.season == "summer")
                action = policy.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)
                if was_summer and (env.x, env.y) != (0, 0):
                    visited_summer_cells.add((env.x, env.y))
                step_count += 1

            survived = env.winter_days_survived
            buried = env.world.total_buried_summer
            retrieved = env.world.total_retrieved_winter
            is_stranded = (env.status == "stranded")

            # Fitness calculation:
            # - Primary: Winter days survived (up to 30) * 2.0
            # - Intermediate progress: Winter nuts eaten * 1.0
            # - Summer caching bonus: 0.25 per nut buried, capped at 1.25x winter demand
            # - Territory exploration bonus: 0.15 per unique cell visited in summer,
            #   capped at 1.5x target_caches to encourage multi-quadrant foraging
            # - Penalty: Exposure outside home burrow
            target_caches = int(self.sim_config.winter_days * self.sim_config.daily_winter_quota * 1.25)
            target_cells = int(target_caches * 1.5)
            fit = (float(survived) * 2.0) + (float(retrieved) * 1.0)
            fit += 0.25 * min(buried, target_caches)
            fit += 0.15 * min(len(visited_summer_cells), target_cells)
            if is_stranded:
                fit -= 0.5

            total_fitness += fit
            total_survived += survived
            total_buried += buried
            total_retrieved += retrieved

        n = self.ga_config.eval_seeds
        return (total_fitness / n), int(total_survived / n), int(total_buried / n), int(total_retrieved / n)

    def _tournament_select(self, scored_pop: List[Tuple[float, PolicyMLP]]) -> PolicyMLP:
        """Select best individual among a random sample of candidates."""
        candidates = random.sample(scored_pop, self.ga_config.tournament_size)
        best = max(candidates, key=lambda item: item[0])
        return best[1]

    def _crossover(self, p1: PolicyMLP, p2: PolicyMLP) -> Tuple[List[float], List[float]]:
        """Uniform crossover between two parent weight vectors."""
        w1, w2 = p1.weights, p2.weights
        c1, c2 = [], []
        for a, b in zip(w1, w2):
            if random.random() < 0.5:
                c1.append(a)
                c2.append(b)
            else:
                c1.append(b)
                c2.append(a)
        return c1, c2

    def _mutate(self, weights: List[float]) -> List[float]:
        """Apply Gaussian perturbation to individual weights."""
        mutated = []
        for w in weights:
            if random.random() < self.ga_config.mutation_rate:
                w += random.gauss(0.0, self.ga_config.mutation_sigma)
            mutated.append(w)
        return mutated

    def train_generation(self, gen: int) -> Dict[str, Any]:
        """Run one generation of evaluation, selection, crossover, and mutation."""
        t0 = time.time()
        scored_pop: List[Tuple[float, int, int, int, PolicyMLP]] = []

        # Evaluate population
        for ind in self.population:
            fit, survived, buried, retrieved = self.evaluate_policy(ind, seed_base=gen * 100)
            scored_pop.append((fit, survived, buried, retrieved, ind))

        # Sort descending by fitness
        scored_pop.sort(key=lambda x: x[0], reverse=True)

        # Track elite
        gen_best_fit = scored_pop[0][0]
        gen_best_survived = scored_pop[0][1]
        gen_best_buried = scored_pop[0][2]
        gen_best_retrieved = scored_pop[0][3]
        gen_avg_fit = sum(x[0] for x in scored_pop) / len(scored_pop)
        gen_avg_survived = sum(x[1] for x in scored_pop) / len(scored_pop)
        gen_avg_buried = sum(x[2] for x in scored_pop) / len(scored_pop)
        gen_avg_retrieved = sum(x[3] for x in scored_pop) / len(scored_pop)

        if gen_best_fit > self.best_fitness:
            self.best_fitness = gen_best_fit
            self.best_survived_days = gen_best_survived
            # Clone best individual
            self.best_individual = PolicyMLP(
                layer_sizes=scored_pop[0][4].layer_sizes,
                weights=list(scored_pop[0][4].weights)
            )

        # Build next generation
        next_population: List[PolicyMLP] = []

        # 1. Elitism: preserve top performers intact
        for i in range(self.ga_config.elite_count):
            elite_orig = scored_pop[i][4]
            next_population.append(PolicyMLP(
                layer_sizes=elite_orig.layer_sizes,
                weights=list(elite_orig.weights)
            ))

        # 2. Reproduction via crossover and mutation
        parent_pool = [(x[0], x[4]) for x in scored_pop]
        while len(next_population) < self.ga_config.pop_size:
            p1 = self._tournament_select(parent_pool)
            p2 = self._tournament_select(parent_pool)

            if random.random() < self.ga_config.crossover_rate:
                w_child1, w_child2 = self._crossover(p1, p2)
            else:
                w_child1 = list(p1.weights)
                w_child2 = list(p2.weights)

            w_child1 = self._mutate(w_child1)
            next_population.append(PolicyMLP(layer_sizes=p1.layer_sizes, weights=w_child1))

            if len(next_population) < self.ga_config.pop_size:
                w_child2 = self._mutate(w_child2)
                next_population.append(PolicyMLP(layer_sizes=p2.layer_sizes, weights=w_child2))

        self.population = next_population
        elapsed = time.time() - t0

        stats = {
            "gen": gen,
            "best_fitness": round(gen_best_fit, 2),
            "best_survived_days": gen_best_survived,
            "best_buried": gen_best_buried,
            "best_retrieved": gen_best_retrieved,
            "avg_fitness": round(gen_avg_fit, 2),
            "avg_survived_days": round(gen_avg_survived, 2),
            "avg_buried": round(gen_avg_buried, 2),
            "avg_retrieved": round(gen_avg_retrieved, 2),
            "all_time_best_survived": self.best_survived_days,
            "elapsed_sec": round(elapsed, 2),
        }
        self.history.append(stats)
        return stats

    def save_history(self, json_path: str = "training_history.json", csv_path: str = "training_history.csv"):
        """Export generation-by-generation training logs for statistical analysis."""
        if not self.history:
            return

        # 1. Save JSON
        with open(json_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # 2. Save CSV
        fieldnames = list(self.history[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

    def train(self, num_generations: Optional[int] = None, callback=None) -> PolicyMLP:
        """Execute the full evolutionary training process."""
        gens = num_generations or self.ga_config.generations

        print(f"\n{'='*65}")
        print(f"🐿️  SQUIRREL SIMULATOR: NEUROEVOLUTION TRAINING")
        print(f"Population: {self.ga_config.pop_size} | Generations: {gens} | Target: 30 Winter Days")
        print(f"{'='*65}\n")

        for g in range(1, gens + 1):
            stats = self.train_generation(g)
            print(
                f"Gen {g:3d}/{gens:3d} | "
                f"Best Fit: {stats['best_fitness']:6.2f} | "
                f"Survived: {stats['best_survived_days']:2d}/30 days | "
                f"Buried: {stats['best_buried']:2d} nuts | "
                f"Avg Survived: {stats['avg_survived_days']:4.1f} | "
                f"Time: {stats['elapsed_sec']:.2f}s"
            )

            if callback:
                callback(stats)

            # Early stop if top squirrel already survives full 30 days of winter
            if self.best_survived_days >= self.sim_config.winter_days:
                print(f"\n🎉 Goal Reached! A squirrel survived all {self.sim_config.winter_days} winter days!")
                break

        # Save training history
        self.save_history()

        return self.best_individual
