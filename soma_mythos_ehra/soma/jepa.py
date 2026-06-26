from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridEncoder(nn.Module):
    def __init__(self, num_symbols: int, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_symbols, latent_dim)
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(grids.long()).permute(0, 3, 1, 2).contiguous()
        return F.normalize(self.net(x), dim=-1)


class JEPAWorldModel(nn.Module):
    """Joint embedding predictive world model with scalar energy scoring."""

    def __init__(self, num_symbols: int = 16, num_actions: int = 8, latent_dim: int = 64) -> None:
        super().__init__()
        self.encoder = GridEncoder(num_symbols, latent_dim)
        self.action_embedding = nn.Embedding(num_actions, latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.energy_head = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1),
            nn.Softplus(),
        )

    def encode(self, grids: torch.Tensor) -> torch.Tensor:
        if grids.ndim == 2:
            grids = grids.unsqueeze(0)
        return self.encoder(grids)

    def predict_latent(self, current_latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_latent = self.action_embedding(actions.long().flatten())
        return F.normalize(self.predictor(torch.cat((current_latent, action_latent), dim=-1)), dim=-1)

    def energy(self, current: torch.Tensor, actions: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        z_t = self.encode(current)
        z_next = self.encode(candidates)
        z_hat = self.predict_latent(z_t, actions)
        pred_error = 1.0 - F.cosine_similarity(z_hat, z_next, dim=-1)
        learned = self.energy_head(torch.cat((z_t, z_hat, z_next), dim=-1)).flatten()
        return pred_error.clamp_min(0.0) + learned
