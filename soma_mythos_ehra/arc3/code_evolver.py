"""LLM Code Evolver v2 — real LLM-driven hypothesis generation.

Uses OpenAI API (or fallback heuristic) to:
1. Generate Python code hypotheses from observed transitions
2. Mutate winning hypotheses via LLM-guided editing
3. Cross over hypotheses via LLM combination
4. Score all hypotheses against observations

When no API key is set, falls back to template-based evolution.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

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
    source: str = "template"  # "llm", "template", "mutant", "crossover"

    @property
    def id(self) -> str:
        import hashlib
        return hashlib.md5(self.code.encode()).hexdigest()[:8]

    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions


@dataclass
class EvolutionConfig:
    population_size: int = 12
    num_generations: int = 8
    mutation_rate: float = 0.4
    crossover_rate: float = 0.2
    elite_count: int = 3
    max_code_length: int = 600
    timeout_per_eval: float = 2.0
    use_llm: bool = True
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.8
    llm_max_retries: int = 2


class LLMCodeEvolver:
    """Evolves Python code hypotheses using LLM + evolutionary search."""

    def __init__(self, config: EvolutionConfig | None = None) -> None:
        self.config = config or EvolutionConfig()
        self.population: list[CodeHypothesis] = []
        self.generation = 0
        self.observation_buffer: list[dict] = []
        self.best_hypothesis: CodeHypothesis | None = None
        self.llm_client = None
        self._init_llm()

    def _init_llm(self) -> None:
        """Initialize OpenAI client if API key available."""
        if not self.config.use_llm:
            return
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                import openai
                self.llm_client = openai.OpenAI(api_key=api_key)
                print("  LLM: OpenAI client initialized")
            except Exception as e:
                print(f"  LLM: Failed to init OpenAI: {e}")
        else:
            print("  LLM: No API key, using heuristic mode")

    @property
    def has_llm(self) -> bool:
        return self.llm_client is not None

    def record_observation(
        self, prev_grid: np.ndarray, action: int,
        next_grid: np.ndarray, reward: float,
    ) -> None:
        self.observation_buffer.append({
            "prev_grid": prev_grid.tolist(),
            "action": action,
            "next_grid": next_grid.tolist(),
            "reward": reward,
        })
        if len(self.observation_buffer) > 300:
            self.observation_buffer = self.observation_buffer[-300:]

    def initialize_population(self) -> list[CodeHypothesis]:
        """Create initial population — LLM-generated if available."""
        if self.has_llm:
            self.population = self._llm_generate_initial()
        else:
            self.population = self._template_initial()
        return self.population

    def evolve(self) -> list[CodeHypothesis]:
        """Run one generation of evolutionary search."""
        self.generation += 1
        self.evaluate_population()

        new_pop = self.population[:self.config.elite_count]

        # LLM or template crossover + mutation
        while len(new_pop) < self.config.population_size:
            if self.has_llm and torch.rand(1).item() < 0.6:
                child = self._llm_crossover_or_mutate()
            else:
                child = self._template_crossover_or_mutate()
            if child:
                new_pop.append(child)

        self.population = new_pop
        self.evaluate_population()
        return self.population

    def evaluate_population(self) -> list[CodeHypothesis]:
        for hyp in self.population:
            hyp.score = self._evaluate_hyp(hyp)
        self.population.sort(key=lambda h: h.score, reverse=True)
        if self.population and (
            self.best_hypothesis is None or
            self.population[0].score > self.best_hypothesis.score
        ):
            self.best_hypothesis = self.population[0]
        return self.population

    def get_best_action(self, grid: np.ndarray, available_actions: list[int]) -> int | None:
        if self.best_hypothesis and self.best_hypothesis.score > 0.3:
            try:
                action = self._execute_hypothesis(self.best_hypothesis, grid, available_actions)
                if action is not None and action in available_actions:
                    return action
            except Exception:
                pass
        return None

    # ── LLM methods ──

    def _llm_generate_initial(self) -> list[CodeHypothesis]:
        """Use LLM to generate diverse hypotheses from observations."""
        obs_summary = self._summarize_observations()
        prompt = f"""You are an AI scientist studying a turn-based grid game (64x64, values 0-15).
The agent takes actions 1-7 (1=up, 2=down, 3=left, 4=right, 5=interact, 6=click at x,y, 7=undo).
After each action, the grid changes. The goal is to find the winning condition.

Here are observed transitions (prev_grid → action → next_grid, reward):
{obs_summary}

Generate {self.config.population_size} Python code hypotheses that predict how the grid changes.
Each hypothesis must:
1. Take `grid` (numpy array), `action` (int), `available_actions` (list) as inputs
2. Set `result` to the predicted next grid
3. Optionally set `recommended_action` for the agent's next move

Return ONLY a JSON array of code strings. No explanation.
Example: ["result = grid.copy()", "import numpy as np\\nresult = np.roll(grid, 1, axis=0)"]
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature,
                max_tokens=2000,
            )
            text = response.choices[0].message.content.strip()
            # Extract JSON array
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                codes = json.loads(match.group())
                return [
                    CodeHypothesis(code=c, generation=0, source="llm")
                    for c in codes[:self.config.population_size]
                ]
        except Exception as e:
            print(f"  LLM generate failed: {e}")
        return self._template_initial()

    def _llm_crossover_or_mutate(self) -> CodeHypothesis | None:
        """Use LLM to crossover or mutate existing hypotheses."""
        if len(self.population) < 2:
            return None

        p1 = self.population[0]
        p2 = self.population[1] if len(self.population) > 1 else self.population[0]

        prompt = f"""You are evolving code hypotheses for a grid game.

Parent 1 (score={p1.score:.2f}):
```python
{p1.code}
```

Parent 2 (score={p2.score:.2f}):
```python
{p2.code}
```

Observations: {self._summarize_observations(max_obs=5)}

Generate ONE improved child hypothesis by combining or mutating the parents.
Return ONLY the Python code string. No explanation.
The code should use `grid`, `action`, `available_actions` as inputs and set `result`.
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature,
                max_tokens=800,
            )
            code = response.choices[0].message.content.strip()
            code = self._extract_code(code)
            if code:
                return CodeHypothesis(
                    code=code, generation=self.generation,
                    parent_id=p1.id, source="llm_mutant",
                )
        except Exception as e:
            print(f"  LLM mutate failed: {e}")
        return self._template_crossover_or_mutate()

    # ── Template fallback methods ──

    def _template_initial(self) -> list[CodeHypothesis]:
        templates = [
            "import numpy as np\nresult = np.roll(grid, -1, axis=0)",
            "import numpy as np\nresult = np.roll(grid, 1, axis=0)",
            "import numpy as np\nresult = np.roll(grid, -1, axis=1)",
            "import numpy as np\nresult = np.roll(grid, 1, axis=1)",
            "import numpy as np\nresult = np.rot90(grid, -1)",
            "import numpy as np\nresult = np.rot90(grid, 1)",
            "import numpy as np\nresult = grid.copy()\nresult[grid > 0] = np.roll(grid, 1, axis=0)[grid > 0]",
            "import numpy as np\nresult = np.zeros_like(grid)\nfor col in range(grid.shape[1]):\n    nz = grid[:, col][grid[:, col] > 0]\n    result[-len(nz):, col] = nz",
            "import numpy as np\nresult = grid.copy()\nfor src in range(1, 10):\n    result[grid == src] = (src % 9) + 1",
            "import numpy as np\nresult = grid.copy()\nif action == 1: result = np.roll(grid, -1, axis=0)\nelif action == 2: result = np.roll(grid, 1, axis=0)\nelif action == 3: result = np.roll(grid, -1, axis=1)\nelif action == 4: result = np.roll(grid, 1, axis=1)",
            "import numpy as np\nmask = grid > 0\nresult = grid.copy()\nif action in [1,2,3,4]:\n    shifts = {1: (-1,0), 2: (1,0), 3: (0,-1), 4: (0,1)}\n    dy, dx = shifts[action]\n    result[mask] = np.roll(grid, (dy, dx), axis=(0,1))[mask]",
            "import numpy as np\nresult = grid.copy()\nif action == 5:\n    nz = np.count_nonzero(grid)\n    result[grid == 0] = 1",
        ]
        return [CodeHypothesis(code=t, generation=0, source="template") for t in templates]

    def _template_crossover_or_mutate(self) -> CodeHypothesis | None:
        if len(self.population) < 2:
            return None
        import random
        p1 = random.choice(self.population[:max(3, len(self.population))])
        p2 = random.choice(self.population[:max(3, len(self.population))])
        lines1 = p1.code.strip().split("\n")
        lines2 = p2.code.strip().split("\n")
        cut1 = len(lines1) // 2
        cut2 = len(lines2) // 2
        child_code = "\n".join(lines1[:cut1] + lines2[cut2:])
        return CodeHypothesis(
            code=child_code, generation=self.generation,
            parent_id=p1.id, source="template_crossover",
        )

    # ── Shared methods ──

    def _evaluate_hyp(self, hyp: CodeHypothesis) -> float:
        correct = 0
        total = min(len(self.observation_buffer), 50)
        for obs in self.observation_buffer[-total:]:
            try:
                predicted = self._execute_code_on_grid(
                    hyp.code, np.array(obs["prev_grid"]), obs["action"],
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
        namespace = {"grid": grid.copy(), "action": action, "np": np, "result": None}
        try:
            exec(code, {"__builtins__": {}}, namespace)
            return namespace.get("result")
        except Exception:
            return None

    def _execute_hypothesis(self, hyp: CodeHypothesis, grid: np.ndarray, available: list[int]) -> int | None:
        namespace = {
            "grid": grid.copy(), "available_actions": available,
            "np": np, "recommended_action": None,
        }
        try:
            exec(hyp.code, {"__builtins__": {}}, namespace)
            return namespace.get("recommended_action")
        except Exception:
            return None

    def _summarize_observations(self, max_obs: int = 10) -> str:
        """Create a text summary of observations for LLM prompt."""
        recent = self.observation_buffer[-max_obs:]
        lines = []
        for i, obs in enumerate(recent):
            prev = np.array(obs["prev_grid"])
            next_g = np.array(obs["next_grid"])
            # Summarize grid as shape + unique values + nonzero count
            prev_summary = f"shape={prev.shape}, values={sorted(prev.unique().tolist())[:8]}, nonzero={np.count_nonzero(prev)}"
            next_summary = f"shape={next_g.shape}, values={sorted(next_g.unique().tolist())[:8]}, nonzero={np.count_nonzero(next_g)}"
            changed = np.mean(prev == next_g)
            lines.append(f"  [{i}] {prev_summary} → action={obs['action']} → {next_summary} (unchanged={changed:.1%}), reward={obs['reward']}")
        return "\n".join(lines) if lines else "  (no observations yet)"

    def _extract_code(self, text: str) -> str | None:
        """Extract Python code from LLM response."""
        # Try to find code block
        match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r'```\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Try bare code
        lines = [l for l in text.strip().split("\n") if l.strip() and not l.startswith("#")]
        if lines:
            return "\n".join(lines)
        return None


# Need torch for random in crossover
import torch
