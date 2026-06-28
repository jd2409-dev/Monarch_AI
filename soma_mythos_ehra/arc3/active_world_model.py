"""Active World Model — maintains belief space over environment dynamics.

Improved v2:
- Grid diff encoder: encodes (prev_grid - next_grid) to learn action effects
- Stop-gradient on target encodings for stable training
- Direct grid reconstruction: predict actual next grid pixels
- Transition predictor uses (state, action, diff_hint) for better predictions
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
        onehot = F.one_hot(grid.long().clamp(0, 15), 16).permute(0, 3, 1, 2).float()
        h = F.relu(self.norm1(self.conv1(onehot)))
        h = self.pool(h)
        h = F.relu(self.norm2(self.conv2(h)))
        h = self.pool(h)
        h = F.relu(self.norm3(self.conv3(h)))
        h = self.adaptive_pool(h)
        h = h.reshape(h.size(0), -1)
        return self.projector(h)


class GridDiffEncoder(nn.Module):
    """Encodes the difference between two grids into a diff embedding.

    Captures what changed between states — the core signal for learning
    action effects.
    """

    def __init__(self, in_channels: int = 32, embed_dim: int = 64) -> None:
        super().__init__()
        # 32 channels: 16 for prev_grid one-hot + 16 for next_grid one-hot
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(4)
        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 64)
        self.projector = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, prev_grid: torch.Tensor, next_grid: torch.Tensor) -> torch.Tensor:
        """Encode diff between prev and next grids.

        Args:
            prev_grid: (batch, H, W) previous grid values 0-15
            next_grid: (batch, H, W) next grid values 0-15
        Returns:
            (batch, embed_dim) diff embedding
        """
        prev_onehot = F.one_hot(prev_grid.long().clamp(0, 15), 16).permute(0, 3, 1, 2).float()
        next_onehot = F.one_hot(next_grid.long().clamp(0, 15), 16).permute(0, 3, 1, 2).float()
        x = torch.cat([prev_onehot, next_onehot], dim=1)  # (B, 32, H, W)
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.pool(h)
        h = F.relu(self.norm2(self.conv2(h)))
        h = self.pool(h)
        h = h.reshape(h.size(0), -1)
        return self.projector(h)


class TransitionPredictor(nn.Module):
    """Predicts next latent state given (current_latent, action, diff_hint)."""

    def __init__(self, latent_dim: int = 256, num_actions: int = 8, diff_dim: int = 64) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, 32)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 32 + diff_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(
        self, latent: torch.Tensor, action: torch.Tensor, diff_hint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        act_emb = self.action_embed(action)
        if diff_hint is not None:
            x = torch.cat([latent, act_emb, diff_hint], dim=-1)
        else:
            # Pad with zeros when no diff available
            zeros = torch.zeros(latent.size(0), 64, device=latent.device)
            x = torch.cat([latent, act_emb, zeros], dim=-1)
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


class GridDecoder(nn.Module):
    """Decodes a latent vector back to a 64x64 grid prediction.

    Predicts logits over 16 cell values for each position.
    """

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128 * 4 * 4),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to grid logits.

        Returns:
            (batch, 16, H, W) logits over 16 cell values
        """
        h = self.net(latent)
        h = h.reshape(h.size(0), 128, 4, 4)
        return self.deconv(h)


class ActiveWorldModel(nn.Module):
    """Full world model with encoder, diff encoder, transition predictor, reward predictor, and decoder.

    v2 improvements:
    - Grid diff encoder captures action effects
    - Decoder predicts actual next grid pixels
    - Transition predictor uses diff hint for better predictions
    """

    def __init__(self, latent_dim: int = 256, num_actions: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = GridEncoder(in_channels=16, latent_dim=latent_dim)
        self.diff_encoder = GridDiffEncoder(in_channels=32, embed_dim=64)
        self.transition = TransitionPredictor(latent_dim, num_actions, diff_dim=64)
        self.reward = RewardPredictor(latent_dim, num_actions)
        self.decoder = GridDecoder(latent_dim)

    def encode(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode grid to latent state."""
        return self.encoder(grid)

    def encode_diff(self, prev_grid: torch.Tensor, next_grid: torch.Tensor) -> torch.Tensor:
        """Encode diff between two grids."""
        return self.diff_encoder(prev_grid, next_grid)

    def predict_next(
        self, latent: torch.Tensor, action: torch.Tensor, diff_hint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict next latent state."""
        return self.transition(latent, action, diff_hint)

    def predict_reward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict reward probability."""
        return self.reward(latent, action)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to grid logits."""
        return self.decoder(latent)

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

    def encode_diff(self, prev_grid: torch.Tensor, next_grid: torch.Tensor) -> torch.Tensor:
        """Encode diff using first model."""
        return self.models[0].encode_diff(prev_grid, next_grid)

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
