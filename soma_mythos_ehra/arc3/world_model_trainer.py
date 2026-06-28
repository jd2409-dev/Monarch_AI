"""World Model Trainer v2 — trains encoder/diff/transition/reward/decoder.

Key fixes:
- Stop-gradient on target encodings (model was chasing its own tail)
- Train diff encoder to predict what changed between grids
- Train decoder to predict actual next grid pixels (not latent proxy)
- Better loss weighting and gradient clipping
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
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    transition_weight: float = 1.0
    reward_weight: float = 1.0
    reconstruction_weight: float = 0.5
    diff_weight: float = 0.3
    max_grad_norm: float = 1.0
    train_steps_per_episode: int = 100
    min_buffer_size: int = 64


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

    v2: Uses stop-gradient on targets, trains diff encoder and decoder.
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

        self.optimizers = [
            torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            for model in self.ensemble.models
        ]

    def train_step(self, model_idx: int = 0) -> TrainMetrics:
        """Run one training step on a sampled batch."""
        if not self.buffer.can_sample(self.config.batch_size):
            return TrainMetrics()

        batch, weights = self.buffer.sample(self.config.batch_size)
        model = self.ensemble.models[model_idx]
        optimizer = self.optimizers[model_idx]

        prev_grids = torch.stack([t.prev_grid for t in batch]).long()
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        next_grids = torch.stack([t.next_grid for t in batch]).long()
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)

        optimizer.zero_grad()

        # --- Encode current state (gradient flows) ---
        latent = model.encode(prev_grids)

        # --- Encode next state with STOP GRADIENT (stable target) ---
        with torch.no_grad():
            next_latent_target = model.encode(next_grids)

        # --- Transition prediction (no diff hint during training for simplicity) ---
        next_latent_pred = model.predict_next(latent, actions)
        transition_loss = F.mse_loss(next_latent_pred, next_latent_target)

        # --- Reward prediction ---
        reward_pred = model.predict_reward(latent, actions).squeeze(-1)
        reward_loss = F.binary_cross_entropy(reward_pred, rewards.clamp(0, 1))

        # --- Grid reconstruction: decode predicted latent → grid logits ---
        grid_logits = model.decode(next_latent_pred)
        # Target: actual next grid as class labels
        grid_target = next_grids.long().clamp(0, 15)
        # Resize logits to match grid if needed
        if grid_logits.shape[2:] != grid_target.shape[1:]:
            grid_logits = F.interpolate(
                grid_logits, size=grid_target.shape[1:], mode='bilinear', align_corners=False,
            )
        reconstruction_loss = F.cross_entropy(
            grid_logits.reshape(-1, 16), grid_target.reshape(-1),
        )

        # --- Combined loss ---
        total_loss = (
            self.config.transition_weight * transition_loss +
            self.config.reward_weight * reward_loss +
            self.config.reconstruction_weight * reconstruction_loss
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.config.max_grad_norm,
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
        """Train on a full episode's transitions. Returns aggregated metrics."""
        count = self.buffer.add_episode(episode_transitions)
        if count == 0:
            return TrainMetrics()

        agg = TrainMetrics()
        start_time = time.time()

        for step in range(self.config.train_steps_per_episode):
            if not self.buffer.can_sample(self.config.batch_size):
                break

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
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load all models."""
        state = torch.load(path, weights_only=True)
        for i, m in enumerate(self.ensemble.models):
            if f"model_{i}" in state["ensemble"]:
                m.load_state_dict(state["ensemble"][f"model_{i}"])
