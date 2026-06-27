from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridEncoder(nn.Module):
    def __init__(self, num_symbols: int, latent_dim: int, in_channels: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        if in_channels == 1:
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
        else:
            self.embedding = None
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, latent_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(latent_dim, latent_dim),
            )

    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        if self.in_channels == 1:
            if grids.ndim == 2:
                grids = grids.unsqueeze(0).unsqueeze(0)
            elif grids.ndim == 3:
                grids = grids.unsqueeze(1)
            # grids is now (B, 1, H, W)
            x = self.embedding(grids.long()).squeeze(1).permute(0, 3, 1, 2).contiguous()
        else:
            if grids.ndim == 3:
                grids = grids.unsqueeze(0)
            # grids is (B, C, H, W)
            x = grids.float()
        return F.normalize(self.net(x), dim=-1)


class JEPAWorldModel(nn.Module):
    """Joint embedding predictive world model with scalar energy scoring."""

    def __init__(self, num_symbols: int = 16, num_actions: int = 8, latent_dim: int = 64, in_channels: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.encoder = GridEncoder(num_symbols, latent_dim, in_channels=in_channels)
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

    def _to_multichannel(self, grids: torch.Tensor) -> torch.Tensor:
        """Ensure grids have self.in_channels channels. Always treat 3D input as (B, H, W)."""
        if self.in_channels == 1:
            if grids.ndim == 2:
                return grids.unsqueeze(0)
            if grids.ndim == 3:
                return grids.unsqueeze(1) if grids.shape[0] != 1 else grids.unsqueeze(0)
            return grids
        # in_channels > 1: ensure (B, C, H, W)
        if grids.ndim == 2:
            grids = grids.unsqueeze(0).unsqueeze(1)
        elif grids.ndim == 3:
            grids = grids.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)
        # grids is now (B, C, H, W)
        if grids.shape[1] == self.in_channels:
            return grids
        # Single-channel → multi-channel conversion
        if grids.shape[1] == 1:
            B, _, H, W = grids.shape
            g = grids.squeeze(1)
            out = torch.zeros(B, self.in_channels, H, W, dtype=torch.float32, device=grids.device)
            out[:, 0] = (g == 1).float()
            out[:, 1] = ((g >= 6) & (g <= 11)).float()
            out[:, 2] = (g == 2).float()
            out[:, 3] = ((g == 0) | (g == 3)).float()
            return out
        return grids

    def encode(self, grids: torch.Tensor) -> torch.Tensor:
        grids = self._to_multichannel(grids)
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

    def compute_energy(
        self,
        current: torch.Tensor,
        actions: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Batch-aware energy computation returning per-sample energies.

        Args:
            current: (B, H, W) or (B, 1, H, W) current grid states
            actions: (B,) action indices
            candidates: (B, H, W) or (B, 1, H, W) candidate next states

        Returns:
            (B,) per-sample energy values
        """
        z_t = self.encode(current)
        z_next = self.encode(candidates)
        action_latent = self.action_embedding(actions.long())
        z_hat = F.normalize(self.predictor(torch.cat((z_t, action_latent), dim=-1)), dim=-1)
        pred_error = 1.0 - F.cosine_similarity(z_hat, z_next, dim=-1)
        learned = self.energy_head(torch.cat((z_t, z_hat, z_next), dim=-1)).squeeze(-1)
        return pred_error.clamp_min(0.0) + learned
