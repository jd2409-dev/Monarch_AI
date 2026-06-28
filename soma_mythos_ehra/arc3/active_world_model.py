"""Active World Model — maintains belief space over environment dynamics.

Core components:
1. State Encoder: compresses 64x64 grid into a latent vector
2. Transition Predictor: predicts next state given (state, action)
3. Reward Predictor: predicts whether an action leads to goal
4. Hypothesis Ensemble: multiple models maintain diverse beliefs
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridEncoder(nn.Module):
    """Encodes a 64x64 grid (values 0-15) into a latent vector."""

    def __init__(self, in_channels: int = 16, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # One-hot encode 16 cell values, then conv
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 64)
        self.norm3 = nn.GroupNorm(8, 128)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.projector = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: (batch, H, W) integer grid values 0-15
        Returns:
            (batch, latent_dim) embedding
        """
        # One-hot encode
        onehot = F.one_hot(grid.long(), 16).permute(0, 3, 1, 2).float()
        h = F.relu(self.norm1(self.conv1(onehot)))
        h = self.pool(h)
        h = F.relu(self.norm2(self.conv2(h)))
        h = self.pool(h)
        h = F.relu(self.norm3(self.conv3(h)))
        h = self.adaptive_pool(h)
        h = h.reshape(h.size(0), -1)
        return self.projector(h)


class TransitionPredictor(nn.Module):
    """Predicts next latent state given (current_latent, action)."""

    def __init__(self, latent_dim: int = 256, num_actions: int = 8) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, 32)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        act_emb = self.action_embed(action)
        x = torch.cat([latent, act_emb], dim=-1)
        return self.net(x)


class RewardPredictor(nn.Module):
    """Predicts probability of reaching goal from (state, action)."""

    def __init__(self, latent_dim: int = 256, num_actions: int = 8) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, 32)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        act_emb = self.action_embed(action)
        x = torch.cat([latent, act_emb], dim=-1)
        return self.net(x)


class ActiveWorldModel(nn.Module):
    """Full world model with encoder, transition predictor, and reward predictor.

    Maintains a belief state over environment dynamics and predicts outcomes
    of potential actions.
    """

    def __init__(self, latent_dim: int = 256, num_actions: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = GridEncoder(in_channels=16, latent_dim=latent_dim)
        self.transition = TransitionPredictor(latent_dim, num_actions)
        self.reward = RewardPredictor(latent_dim, num_actions)

    def encode(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode grid to latent state."""
        return self.encoder(grid)

    def predict_next(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent state."""
        return self.transition(latent, action)

    def predict_reward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict reward probability."""
        return self.reward(latent, action)

    def forward(self, grid: torch.Tensor, action: torch.Tensor):
        """Full forward pass: encode → predict transition + reward."""
        latent = self.encoder(grid)
        next_latent = self.transition(latent, action)
        reward = self.reward(latent, action)
        return latent, next_latent, reward


class HypothesisEnsemble(nn.Module):
    """Ensemble of world models for uncertainty estimation.

    Multiple models maintain diverse beliefs about environment dynamics.
    High disagreement = high information gain potential.
    """

    def __init__(self, num_models: int = 3, latent_dim: int = 256, num_actions: int = 8) -> None:
        super().__init__()
        self.models = nn.ModuleList([
            ActiveWorldModel(latent_dim, num_actions)
            for _ in range(num_models)
        ])

    def encode(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode using first model (shared encoder)."""
        return self.models[0].encode(grid)

    def predict_next_ensemble(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get predictions from all models.

        Returns:
            (num_models, batch, latent_dim) predictions
        """
        predictions = []
        for model in self.models:
            next_lat = model.predict_next(latent, action)
            predictions.append(next_lat)
        return torch.stack(predictions)

    def predict_reward_ensemble(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get reward predictions from all models.

        Returns:
            (num_models, batch, 1) predictions
        """
        predictions = []
        for model in self.models:
            rew = model.predict_reward(latent, action)
            predictions.append(rew)
        return torch.stack(predictions)

    def uncertainty(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute predictive uncertainty across ensemble.

        Higher variance = more disagreement = more information to gain.
        """
        predictions = self.predict_next_ensemble(latent, action)
        return predictions.var(dim=0).mean(dim=-1)
