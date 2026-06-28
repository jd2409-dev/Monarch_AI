"""LLM Code Evolver — generates and mutates Python scripts that model environment rules.

Strategy:
1. Observe state transitions from environment interaction
2. Use an LLM to generate hypotheses as executable Python code
3. Score each hypothesis by how well it predicts observed transitions
4. Mutate the best hypotheses to explore the rule space
5. Extract actionable policies from winning code

This is the "scientist" component — it writes theories (code) about how
the environment works and tests them against observations.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np


@dataclass
class CodeHypothesis:
    """A code-based hypothesis about environment rules."""
    code: str
    score: float = 0.0
    correct_predictions: int = 0
    total_predictions: int = 0
    generation: int = 0
    parent_id: str | None = None

    @property
    def id(self) -> str:
        return hashlib.md5(self.code.encode()).hexdigest()[:8]

    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions


@dataclass
class EvolutionConfig:
    """Configuration for code evolution."""
    population_size: int = 10
    num_generations: int = 5
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    elite_count: int = 2
    max_code_length: int = 500
    timeout_per_eval: float = 2.0


class LLMCodeEvolver:
    """Evolves Python code hypotheses about environment dynamics.

    Uses evolutionary search over code space, where fitness is measured
    by how accurately each code hypothesis predicts observed transitions.
    """

    def __init__(self, config: EvolutionConfig | None = None) -> None:
        self.config = config or EvolutionConfig()
        self.population: list[CodeHypothesis] = []
        self.generation = 0
        self.observation_buffer: list[dict] = []
        self.best_hypothesis: CodeHypothesis | None = None

    def record_observation(
        self,
        prev_grid: np.ndarray,
        action: int,
        next_grid: np.ndarray,
        reward: float,
    ) -> None:
        """Record an observation for evolution."""
        self.observation_buffer.append({
            "prev_grid": prev_grid.tolist(),
            "action": action,
            "next_grid": next_grid.tolist(),
            "reward": reward,
        })
        # Keep buffer bounded
        if len(self.observation_buffer) > 200:
            self.observation_buffer = self.observation_buffer[-200:]

    def initialize_population(self) -> list[CodeHypothesis]:
        """Create initial population of code hypotheses."""
        templates = [
            # Movement hypotheses
            self._template_move_agent(),
            self._template_move_direction(),
            # Click hypotheses
            self._template_click_fill(),
            self._template_click_toggle(),
            # Pattern hypotheses
            self._template_shift_rows(),
            self._template_shift_cols(),
            self._template_rotate(),
            self._template_color_map(),
            # Physics hypotheses
            self._template_gravity(),
            self._template_propagation(),
        ]

        self.population = [
            CodeHypothesis(code=t, generation=0)
            for t in templates
        ]
        return self.population

    def evaluate_population(self) -> list[CodeHypothesis]:
        """Score each hypothesis against observed transitions."""
        if not self.observation_buffer:
            return self.population

        for hyp in self.population:
            score = self._evaluate_hyp(hyp)
            hyp.score = score

        self.population.sort(key=lambda h: h.score, reverse=True)

        if self.population and self.population[0].score > (self.best_hypothesis.score if self.best_hypothesis else 0):
            self.best_hypothesis = self.population[0]

        return self.population

    def evolve(self) -> list[CodeHypothesis]:
        """Run one generation of evolutionary search."""
        self.generation += 1

        # Evaluate current population
        self.evaluate_population()

        # Selection: keep elites
        new_pop = self.population[:self.config.elite_count]

        # Crossover
        while len(new_pop) < self.config.population_size - 2:
            if len(self.population) >= 2:
                p1, p2 = self._tournament_select(2)
                child = self._crossover(p1, p2)
                if child:
                    new_pop.append(child)
            else:
                break

        # Mutation
        for i in range(len(new_pop)):
            if i >= self.config.elite_count and torch.rand(1).item() < self.config.mutation_rate:
                new_pop[i] = self._mutate(new_pop[i])

        # Fill remaining with random
        while len(new_pop) < self.config.population_size:
            code = self._random_variant()
            new_pop.append(CodeHypothesis(code=code, generation=self.generation))

        self.population = new_pop
        return self.population

    def get_best_action(self, grid: np.ndarray, available_actions: list[int]) -> int:
        """Use the best hypothesis to select an action."""
        if self.best_hypothesis and self.best_hypothesis.score > 0.5:
            try:
                action = self._execute_hypothesis(self.best_hypothesis, grid, available_actions)
                if action is not None and action in available_actions:
                    return action
            except Exception:
                pass

        # Fallback: random
        return available_actions[0] if available_actions else 1

    def get_policy_summary(self) -> str:
        """Get a human-readable summary of the best hypothesis."""
        if self.best_hypothesis:
            return f"Best (score={self.best_hypothesis.score:.2f}):\n{self.best_hypothesis.code}"
        return "No hypothesis found"

    def _evaluate_hyp(self, hyp: CodeHypothesis) -> float:
        """Evaluate a hypothesis against observations."""
        correct = 0
        total = min(len(self.observation_buffer), 50)

        for obs in self.observation_buffer[-total:]:
            try:
                predicted = self._execute_code_on_grid(
                    hyp.code,
                    np.array(obs["prev_grid"]),
                    obs["action"],
                )
                actual = np.array(obs["next_grid"])
                if predicted is not None and predicted.shape == actual.shape:
                    match_ratio = np.mean(predicted == actual)
                    correct += match_ratio
                    hyp.total_predictions += 1
                    if match_ratio > 0.8:
                        hyp.correct_predictions += 1
            except Exception:
                pass

        return correct / max(total, 1)

    def _execute_code_on_grid(self, code: str, grid: np.ndarray, action: int) -> np.ndarray | None:
        """Safely execute a hypothesis code on a grid."""
        namespace = {
            "grid": grid.copy(),
            "action": action,
            "np": np,
            "result": None,
        }
        try:
            exec(code, {"__builtins__": {}}, namespace)
            return namespace.get("result")
        except Exception:
            return None

    def _execute_hypothesis(self, hyp: CodeHypothesis, grid: np.ndarray, available: list[int]) -> int | None:
        """Execute hypothesis to get an action recommendation."""
        namespace = {
            "grid": grid.copy(),
            "available_actions": available,
            "np": np,
            "recommended_action": None,
        }
        try:
            exec(hyp.code, {"__builtins__": {}}, namespace)
            return namespace.get("recommended_action")
        except Exception:
            return None

    def _tournament_select(self, k: int = 2) -> list[CodeHypothesis]:
        """Tournament selection."""
        import random
        selected = []
        for _ in range(k):
            candidates = random.sample(self.population, min(k, len(self.population)))
            winner = max(candidates, key=lambda h: h.score)
            selected.append(winner)
        return selected

    def _crossover(self, p1: CodeHypothesis, p2: CodeHypothesis) -> CodeHypothesis | None:
        """Cross over two code hypotheses."""
        lines1 = p1.code.strip().split("\n")
        lines2 = p2.code.strip().split("\n")
        if not lines1 or not lines2:
            return None
        cut1 = len(lines1) // 2
        cut2 = len(lines2) // 2
        child_code = "\n".join(lines1[:cut1] + lines2[cut2:])
        return CodeHypothesis(
            code=child_code,
            generation=self.generation,
            parent_id=p1.id,
        )

    def _mutate(self, hyp: CodeHypothesis) -> CodeHypothesis:
        """Mutate a code hypothesis."""
        mutations = [
            self._mutate_constant,
            self._mutate_condition,
            self._mutate_action,
        ]
        import random
        mutator = random.choice(mutations)
        new_code = mutator(hyp.code)
        return CodeHypothesis(
            code=new_code,
            generation=self.generation,
            parent_id=hyp.id,
        )

    def _mutate_constant(self, code: str) -> str:
        """Randomly change a numeric constant."""
        import random
        nums = re.findall(r'\b\d+\b', code)
        if nums:
            old = random.choice(nums)
            new = str(random.randint(0, 15))
            code = code.replace(old, new, 1)
        return code

    def _mutate_condition(self, code: str) -> str:
        """Toggle a comparison operator."""
        import random
        ops = ["==", "!=", "<", ">", "<=", ">="]
        for op in ops:
            if op in code:
                new_op = random.choice([o for o in ops if o != op])
                code = code.replace(op, new_op, 1)
                break
        return code

    def _mutate_action(self, code: str) -> str:
        """Change the recommended action."""
        import random
        if "recommended_action" in code:
            new_action = random.randint(1, 7)
            code = re.sub(
                r'recommended_action\s*=\s*\d+',
                f'recommended_action = {new_action}',
                code,
            )
        return code

    def _random_variant(self) -> str:
        """Generate a random code variant."""
        import random
        action = random.randint(1, 5)
        direction = random.choice(["rows", "cols"])
        return f"""
import numpy as np
grid = grid.copy()
result = grid.copy()
if action == {action}:
    if direction == "{direction}":
        result = np.roll(grid, 1, axis={'0' if direction == 'rows' else '1'})
recommended_action = {action}
direction = "{direction}"
"""

    # --- Template hypotheses ---

    def _template_move_agent(self) -> str:
        return """
import numpy as np
grid = grid.copy()
result = grid.copy()
if action == 1: result = np.roll(grid, -1, axis=0)
elif action == 2: result = np.roll(grid, 1, axis=0)
elif action == 3: result = np.roll(grid, -1, axis=1)
elif action == 4: result = np.roll(grid, 1, axis=1)
"""

    def _template_move_direction(self) -> str:
        return """
import numpy as np
grid = grid.copy()
result = grid.copy()
# Move non-zero cells in action direction
mask = grid > 0
if action == 1: result[mask] = np.roll(grid, -1, axis=0)[mask]
elif action == 2: result[mask] = np.roll(grid, 1, axis=0)[mask]
elif action == 3: result[mask] = np.roll(grid, -1, axis=1)[mask]
elif action == 4: result[mask] = np.roll(grid, 1, axis=1)[mask]
"""

    def _template_click_fill(self) -> str:
        return """
import numpy as np
grid = grid.copy()
result = grid.copy()
if action == 6 and available_actions:
    recommended_action = 6
"""

    def _template_click_toggle(self) -> str:
        return """
import numpy as np
grid = grid.copy()
result = grid.copy()
if action == 6:
    nonzero = grid > 0
    result[nonzero] = 0
    result[~nonzero] = 1
"""

    def _template_shift_rows(self) -> str:
        return """
import numpy as np
grid = grid.copy()
result = np.roll(grid, 1, axis=0)
"""

    def _template_shift_cols(self) -> str:
        return """
import numpy as np
grid = grid.copy()
result = np.roll(grid, 1, axis=1)
"""

    def _template_rotate(self) -> str:
        return """
import numpy as np
grid = grid.copy()
result = np.rot90(grid, -1)
"""

    def _template_color_map(self) -> str:
        return """
import numpy as np
grid = grid.copy()
result = grid.copy()
# Remap colors: 1->2, 2->3, etc.
for src in range(1, 10):
    result[grid == src] = (src % 9) + 1
"""

    def _template_gravity(self) -> str:
        return """
import numpy as np
grid = grid.copy()
result = np.zeros_like(grid)
for col in range(grid.shape[1]):
    nonzero = grid[:, col][grid[:, col] > 0]
    result[-len(nonzero):, col] = nonzero
"""

    def _template_propagation(self) -> str:
        return """
import numpy as np
grid = grid.copy()
result = grid.copy()
# Spread non-zero values to neighbors
for i in range(1, grid.shape[0]-1):
    for j in range(1, grid.shape[1]-1):
        if grid[i, j] > 0:
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                if result[i+di, j+dj] == 0:
                    result[i+di, j+dj] = grid[i, j]
"""
