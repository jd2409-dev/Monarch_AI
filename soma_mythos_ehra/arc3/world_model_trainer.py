"""World Model Trainer — trains encoder/transition/reward on collected experience.

Online learning loop:
1. Collect transitions from environment interaction
2. Train world model to predict (next_state, reward) from (state, action)
3. Use reconstruction loss + transition loss + reward loss
4. Update ensemble members with different random seeds for diversity
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from soma_mythos_ehra.arc3.active_world_model import ActiveWorldModel, HypothesisEnsemble
from soma_mythos_ehra.arc3.replay_buffer import ExperienceReplayBuffer, Transition


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    transition_weight: float = 1.0
    reward_weight: float = 1.0
    reconstruction_weight: float = 0.5
    max_grad_norm: float = 1.0
    train_steps_per_episode: int = 100
    min_buffer_size: int = 256


@dataclass
class TrainMetrics:
    """Training metrics from one training round."""
    total_loss: float = 0.0
    transition_loss: float = 0.0
    reward_loss: float = 0.0
    reconstruction_loss: float = 0.0
    train_steps: int = 0
    train_time: float = 0.0


class WorldModelTrainer:
    """Trains the world model ensemble on collected experience.

    The training uses three loss components:
    1. Transition prediction: MSE between predicted and actual next latent
    2. Reward prediction: BCE for reward prediction
    3. Reconstruction: decode predicted latent back to grid
    """

    def __init__(
        self,
        ensemble: HypothesisEnsemble,
        buffer: ExperienceReplayBuffer,
        config: TrainConfig | None = None,
    ) -> None:
        self.ensemble = ensemble
        self.buffer = buffer
        self.config = config or TrainConfig()

        # Separate optimizers for each ensemble member
        self.optimizers = [
            torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            for model in self.ensemble.models
        ]

        # Simple grid decoder for reconstruction loss
        latent_dim = ensemble.models[0].latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128 * 4 * 4),
            nn.ReLU(),
        )

    def train_step(self, model_idx: int = 0) -> TrainMetrics:
        """Run one training step on a sampled batch."""
        if not self.buffer.can_sample(self.config.batch_size):
            return TrainMetrics()

        batch, weights = self.buffer.sample(self.config.batch_size)
        model = self.ensemble.models[model_idx]
        optimizer = self.optimizers[model_idx]

        # Prepare batch tensors
        prev_grids = torch.stack([t.prev_grid for t in batch]).long()
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        next_grids = torch.stack([t.next_grid for t in batch]).long()
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)

        optimizer.zero_grad()

        # Encode current and next states
        latent = model.encode(prev_grids)
        next_latent_target = model.encode(next_grids.detach())

        # Transition prediction
        next_latent_pred = model.predict_next(latent, actions)
        transition_loss = F.mse_loss(next_latent_pred, next_latent_target.detach())

        # Reward prediction
        reward_pred = model.predict_reward(latent, actions).squeeze(-1)
        reward_loss = F.binary_cross_entropy(reward_pred, rewards)

        # Reconstruction loss (decode predicted latent → grid features)
        recon_pred = self.decoder(next_latent_pred)
        # Compare against encoded next state as proxy
        recon_target = self.decoder(next_latent_target.detach())
        reconstruction_loss = F.mse_loss(recon_pred, recon_target.detach())

        # Combined loss
        total_loss = (
            self.config.transition_weight * transition_loss +
            self.config.reward_weight * reward_loss +
            self.config.reconstruction_weight * reconstruction_loss
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(self.decoder.parameters()),
            self.config.max_grad_norm,
        )
        optimizer.step()

        return TrainMetrics(
            total_loss=total_loss.item(),
            transition_loss=transition_loss.item(),
            reward_loss=reward_loss.item(),
            reconstruction_loss=reconstruction_loss.item(),
            train_steps=1,
        )

    def train_episode(self, episode_transitions: list[dict]) -> TrainMetrics:
        """Train on a full episode's transitions.

        Returns aggregated metrics.
        """
        # Add episode to buffer
        count = self.buffer.add_episode(episode_transitions)
        if count == 0:
            return TrainMetrics()

        # Train all ensemble members
        agg = TrainMetrics()
        start_time = time.time()

        for step in range(self.config.train_steps_per_episode):
            if not self.buffer.can_sample(self.config.batch_size):
                break

            # Rotate through ensemble members
            model_idx = step % len(self.ensemble.models)
            metrics = self.train_step(model_idx)

            agg.total_loss += metrics.total_loss
            agg.transition_loss += metrics.transition_loss
            agg.reward_loss += metrics.reward_loss
            agg.reconstruction_loss += metrics.reconstruction_loss
            agg.train_steps += 1

        agg.train_time = time.time() - start_time
        return agg

    def save(self, path: str) -> None:
        """Save all models."""
        state = {
            "ensemble": {
                f"model_{i}": m.state_dict()
                for i, m in enumerate(self.ensemble.models)
            },
            "decoder": self.decoder.state_dict(),
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load all models."""
        state = torch.load(path, weights_only=True)
        for i, m in enumerate(self.ensemble.models):
            if f"model_{i}" in state["ensemble"]:
                m.load_state_dict(state["ensemble"][f"model_{i}"])
        if "decoder" in state:
            self.decoder.load_state_dict(state["decoder"])
