"""InfoMax Explorer — curiosity-driven action selection for ARC-AGI-3.

Selects actions that maximize information gain about environment dynamics.
Uses the hypothesis ensemble's uncertainty to identify which actions will
teach the agent the most about the environment's hidden rules.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from soma_mythos_ehra.arc3.active_world_model import HypothesisEnsemble


@dataclass
class ActionPlan:
    """A planned action with its expected information gain."""
    action: int
    info_gain: float
    predicted_reward: float
    uncertainty: float
    x: int | None = None
    y: int | None = None


class InfoMaxExplorer:
    """Selects actions that maximize information gain about environment rules.

    Strategy:
    1. Early exploration: random/probing actions to gather baseline data
    2. Uncertainty-driven: choose actions where ensemble disagrees most
    3. Goal-directed: when reward predictions are confident, exploit
    """

    def __init__(
        self,
        ensemble: HypothesisEnsemble,
        exploration_rate: float = 0.3,
        temperature: float = 1.0,
    ) -> None:
        self.ensemble = ensemble
        self.exploration_rate = exploration_rate
        self.temperature = temperature
        self.step_count = 0
        self.action_history: list[int] = []
        self.observation_history: list[torch.Tensor] = []

    def select_action(
        self,
        current_grid: torch.Tensor,
        available_actions: list[int],
        use_click: bool = False,
        grid_shape: tuple[int, int] = (64, 64),
    ) -> ActionPlan:
        """Select the best action given current state.

        Args:
            current_grid: (H, W) current grid state
            available_actions: list of valid action numbers
            use_click: whether ACTION6 (click) is available
            grid_shape: shape of the grid for click coordinate generation
        Returns:
            ActionPlan with chosen action and metadata
        """
        self.step_count += 1
        self.observation_history.append(current_grid.clone())

        # Early exploration: try each action at least once
        if len(self.action_history) < len(available_actions):
            untried = [a for a in available_actions if a not in self.action_history]
            if untried:
                action = untried[0]
                self.action_history.append(action)
                return ActionPlan(
                    action=action, info_gain=1.0,
                    predicted_reward=0.0, uncertainty=1.0,
                )

        # Random exploration with decreasing probability
        if random.random() < self.exploration_rate / (1 + self.step_count * 0.01):
            action = random.choice(available_actions)
            self.action_history.append(action)
            return ActionPlan(
                action=action, info_gain=0.5,
                predicted_reward=0.0, uncertainty=0.5,
            )

        # Ensemble uncertainty-driven selection
        latent = self.ensemble.encode(current_grid.unsqueeze(0))
        plans = []

        for action in available_actions:
            action_tensor = torch.tensor([action])

            # Get ensemble predictions
            uncertainty = self.ensemble.uncertainty(latent, action_tensor).item()
            reward_pred = self.ensemble.predict_reward_ensemble(
                latent, action_tensor
            ).mean(dim=0).item()

            # Info gain = uncertainty + novelty bonus
            action_count = self.action_history.count(action)
            novelty = 1.0 / (1 + action_count)
            info_gain = uncertainty + 0.3 * novelty

            plan = ActionPlan(
                action=action,
                info_gain=info_gain,
                predicted_reward=reward_pred,
                uncertainty=uncertainty,
            )

            # For ACTION6, also consider click coordinates
            if action == 6 and use_click:
                best_x, best_y = self._select_click_target(
                    current_grid, latent, grid_shape
                )
                plan.x = best_x
                plan.y = best_y

            plans.append(plan)

        # Softmax selection over info gain
        if plans:
            gains = torch.tensor([p.info_gain for p in plans])
            probs = F.softmax(gains / self.temperature, dim=0)
            idx = torch.multinomial(probs, 1).item()
            chosen = plans[idx]
        else:
            chosen = ActionPlan(
                action=available_actions[0], info_gain=0,
                predicted_reward=0, uncertainty=0,
            )

        self.action_history.append(chosen.action)
        return chosen

    def _select_click_target(
        self,
        grid: torch.Tensor,
        latent: torch.Tensor,
        grid_shape: tuple[int, int],
    ) -> tuple[int, int]:
        """Select the most informative click coordinates.

        Strategy: click on cells with high entropy (many possible values)
        or at boundaries between different cell types.
        """
        H, W = grid_shape
        best_x, best_y = W // 2, H // 2
        max_score = -1.0

        # Sample candidate positions
        candidates = [(random.randint(0, W-1), random.randint(0, H-1))
                      for _ in range(20)]

        for x, y in candidates:
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                # Score based on local variation
                patch = grid[max(0,y-1):y+2, max(0,x-1):x+2]
                score = float(patch.std())
                # Bonus for boundary cells
                if y > 0 and y < grid.shape[0]-1:
                    if grid[y-1, x] != grid[y+1, x]:
                        score += 1.0
                if x > 0 and x < grid.shape[1]-1:
                    if grid[y, x-1] != grid[y, x+1]:
                        score += 1.0
                if score > max_score:
                    max_score = score
                    best_x, best_y = x, y

        return best_x, best_y

    def reset(self) -> None:
        """Reset explorer state for new episode."""
        self.step_count = 0
        self.action_history.clear()
        self.observation_history.clear()
