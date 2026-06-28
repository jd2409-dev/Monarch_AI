"""Experience Replay Buffer — stores and samples transitions for online learning.

Supports:
- Standard uniform sampling
- Prioritized experience replay (PER) based on TD error
- Reservoir sampling for memory-bounded streaming
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import NamedTuple

import torch
import numpy as np


class Transition(NamedTuple):
    """A single environment transition."""
    prev_grid: torch.Tensor
    action: int
    next_grid: torch.Tensor
    reward: float
    done: bool
    available_actions: list[int]
    level: int


@dataclass
class PERStats:
    """Prioritized experience replay statistics."""
    td_error: float = 0.0
    priority: float = 1.0


class ExperienceReplayBuffer:
    """Memory-bounded experience replay with optional prioritization.

    Stores (grid, action, next_grid, reward, done) tuples and samples
    batches for world model training.
    """

    def __init__(self, capacity: int = 50000, alpha: float = 0.6, beta: float = 0.4) -> None:
        self.capacity = capacity
        self.alpha = alpha  # prioritization exponent
        self.beta = beta    # importance sampling exponent
        self.buffer: list[Transition] = []
        self.priorities: list[float] = []
        self.position = 0
        self.total_added = 0

    def add(
        self,
        prev_grid: torch.Tensor,
        action: int,
        next_grid: torch.Tensor,
        reward: float,
        done: bool,
        available_actions: list[int],
        level: int = 0,
    ) -> None:
        """Add a transition to the buffer."""
        transition = Transition(
            prev_grid=prev_grid.clone(),
            action=action,
            next_grid=next_grid.clone(),
            reward=reward,
            done=done,
            available_actions=list(available_actions),
            level=level,
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(1.0)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = 1.0

        self.position = (self.position + 1) % self.capacity
        self.total_added += 1

    def add_episode(
        self,
        episode_transitions: list[dict],
    ) -> int:
        """Add a full episode's transitions. Returns count added."""
        count = 0
        for t in episode_transitions:
            self.add(
                prev_grid=t["prev_grid"],
                action=t["action"],
                next_grid=t["next_grid"],
                reward=t["reward"],
                done=t["done"],
                available_actions=t.get("available_actions", []),
                level=t.get("level", 0),
            )
            count += 1
        return count

    def sample(self, batch_size: int = 32) -> tuple[Transition, torch.Tensor]:
        """Sample a batch of transitions.

        Returns:
            Batch of transitions, importance sampling weights
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        if sum(self.priorities[:len(self.buffer)]) > 0:
            # Prioritized sampling
            probs = np.array(self.priorities[:len(self.buffer)], dtype=np.float64)
            probs = probs ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
            # Importance sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            weights = torch.tensor(weights, dtype=torch.float32)
        else:
            # Uniform sampling
            indices = random.sample(range(len(self.buffer)), batch_size)
            weights = torch.ones(batch_size)

        batch = [self.buffer[i] for i in indices]
        return batch, weights

    def update_priorities(self, indices: list[int], td_errors: list[float]) -> None:
        """Update priorities based on TD errors."""
        for idx, td in zip(indices, td_errors):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = abs(td) + 1e-6

    def can_sample(self, batch_size: int = 32) -> bool:
        """Check if buffer has enough transitions for a batch."""
        return len(self.buffer) >= batch_size

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.priorities.clear()
        self.position = 0
        self.total_added = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def get_recent(self, n: int = 10) -> list[Transition]:
        """Get the N most recent transitions."""
        if n >= len(self.buffer):
            return list(self.buffer)
        return list(self.buffer[-n:])
