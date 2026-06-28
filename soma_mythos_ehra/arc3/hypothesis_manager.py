"""Hypothesis Manager — Bayesian belief tracking over environment rules.

Maintains a distribution over possible environment dynamics and updates
beliefs based on observed state transitions. Each hypothesis represents
a theory about how the environment responds to actions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import numpy as np


@dataclass
class Hypothesis:
    """A single hypothesis about environment dynamics."""
    name: str
    log_prob: float = 0.0  # log prior probability
    features: dict = field(default_factory=dict)
    correct_predictions: int = 0
    total_predictions: int = 0

    @property
    def probability(self) -> float:
        return math.exp(self.log_prob)

    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.5
        return self.correct_predictions / self.total_predictions


class HypothesisManager:
    """Manages a belief distribution over environment hypotheses.

    Each hypothesis encodes a theory about the environment:
    - Movement: agent can move in grid
    - Click: clicking cells triggers effects
    - Sequence: specific action sequences unlock progress
    - Physics: objects follow physical rules
    - Puzzle: logical puzzle rules apply
    """

    def __init__(self) -> None:
        self.hypotheses: list[Hypothesis] = self._create_default_hypotheses()
        self.transition_buffer: list[tuple[torch.Tensor, int, torch.Tensor]] = []
        self.max_buffer = 100

    def _create_default_hypotheses(self) -> list[Hypothesis]:
        """Create initial hypothesis set with uniform priors."""
        n = 10
        log_prior = -math.log(n)
        return [
            Hypothesis("movement_grid", log_prior, {"type": "movement", "space": "grid"}),
            Hypothesis("movement_directional", log_prior, {"type": "movement", "space": "directional"}),
            Hypothesis("click_interaction", log_prior, {"type": "click", "target": "cell"}),
            Hypothesis("click_region", log_prior, {"type": "click", "target": "region"}),
            Hypothesis("sequence_lock", log_prior, {"type": "sequence", "pattern": "ordered"}),
            Hypothesis("sequence_combo", log_prior, {"type": "sequence", "pattern": "combo"}),
            Hypothesis("physics_gravity", log_prior, {"type": "physics", "rule": "gravity"}),
            Hypothesis("physics_collision", log_prior, {"type": "physics", "rule": "collision"}),
            Hypothesis("puzzle_logic", log_prior, {"type": "puzzle", "rule": "logic"}),
            Hypothesis("puzzle_pattern", log_prior, {"type": "puzzle", "rule": "pattern"}),
        ]

    def update(self, prev_grid: torch.Tensor, action: int, next_grid: torch.Tensor) -> None:
        """Update beliefs based on observed transition.

        Compares actual transition against each hypothesis's prediction
        and updates posterior probabilities via Bayes' rule.
        """
        self.transition_buffer.append((prev_grid.clone(), action, next_grid.clone()))
        if len(self.transition_buffer) > self.max_buffer:
            self.transition_buffer.pop(0)

        # Compute likelihood for each hypothesis
        log_likelihoods = []
        for hyp in self.hypotheses:
            ll = self._compute_likelihood(hyp, prev_grid, action, next_grid)
            log_likelihoods.append(ll)

        # Update posteriors: log_prob += log_likelihood
        max_ll = max(log_likelihoods) if log_likelihoods else 0
        for hyp, ll in zip(self.hypotheses, log_likelihoods):
            hyp.log_prob += ll - max_ll  # numerical stability
            hyp.total_predictions += 1
            if ll > -0.5:  # threshold for "correct" prediction
                hyp.correct_predictions += 1

        # Normalize
        self._normalize()

    def _compute_likelihood(
        self,
        hyp: Hypothesis,
        prev_grid: torch.Tensor,
        action: int,
        next_grid: torch.Tensor,
    ) -> float:
        """Compute how well a hypothesis predicts the observed transition."""
        htype = hyp.features.get("type", "")

        # Grid changed at all?
        changed = not torch.equal(prev_grid, next_grid)
        num_changed = (prev_grid != next_grid).sum().item()

        if htype == "movement":
            # Movement hypotheses expect small, localized changes
            if changed and num_changed < prev_grid.numel() * 0.3:
                return 0.5  # consistent with movement
            elif not changed and action in [1, 2, 3, 4]:
                return -0.2  # movement action but no change
            return -0.5

        elif htype == "click":
            # Click hypotheses expect targeted changes
            if action == 6:
                if changed and num_changed < prev_grid.numel() * 0.2:
                    return 0.8  # click caused localized change
                return 0.2
            return -0.3  # non-click action

        elif htype == "sequence":
            # Sequence hypotheses need multiple steps to evaluate
            if len(self.transition_buffer) >= 3:
                recent_changes = sum(
                    1 for p, a, n in self.transition_buffer[-3:]
                    if not torch.equal(p, n)
                )
                if recent_changes > 0:
                    return 0.3
            return 0.0

        elif htype == "physics":
            # Physics hypotheses expect smooth, consistent changes
            if changed:
                return 0.2 if num_changed < prev_grid.numel() * 0.5 else -0.3
            return 0.0

        elif htype == "puzzle":
            # Puzzle hypotheses expect logical, structured changes
            if changed:
                # Check if change preserves structure (same number of distinct values)
                prev_vals = len(prev_grid.unique())
                next_vals = len(next_grid.unique())
                if abs(prev_vals - next_vals) <= 1:
                    return 0.4
            return 0.0

        return 0.0

    def _normalize(self) -> None:
        """Normalize log probabilities to sum to 1."""
        log_sum = math.log(sum(h.probability for h in self.hypotheses))
        for hyp in self.hypotheses:
            hyp.log_prob -= log_sum

    def get_top_hypotheses(self, k: int = 3) -> list[Hypothesis]:
        """Get the k most probable hypotheses."""
        return sorted(self.hypotheses, key=lambda h: h.log_prob, reverse=True)[:k]

    def get_belief_vector(self) -> torch.Tensor:
        """Get the full belief distribution as a tensor."""
        return torch.tensor([h.probability for h in self.hypotheses])

    def suggest_action_bias(self, available_actions: list[int]) -> dict[int, float]:
        """Suggest action biases based on top hypotheses.

        Returns action -> bonus score mapping.
        """
        bonuses = {a: 0.0 for a in available_actions}
        top = self.get_top_hypotheses(3)

        for hyp in top:
            htype = hyp.features.get("type", "")
            weight = hyp.probability

            if htype == "click":
                if 6 in bonuses:
                    bonuses[6] += weight * 2.0
            elif htype == "movement":
                for a in [1, 2, 3, 4]:
                    if a in bonuses:
                        bonuses[a] += weight * 1.0
            elif htype == "sequence":
                # Encourage trying different actions
                for a in bonuses:
                    if self._action_count(a) < 3:
                        bonuses[a] += weight * 0.5

        return bonuses

    def _action_count(self, action: int) -> int:
        return sum(1 for _, a, _ in self.transition_buffer if a == action)

    def reset(self) -> None:
        """Reset beliefs for new environment."""
        self.hypotheses = self._create_default_hypotheses()
        self.transition_buffer.clear()
