"""JEPA Encoder-Decoder — deep CNN encoder + auto-regressive Transformer decoder.

Encoder: compresses (input_grid, output_grid) pair into a 256-dim latent vector.
Decoder: auto-regressively generates DSL token sequence from the latent vector.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CNN Encoder: Grid Pair → 256-dim Latent Vector
# ---------------------------------------------------------------------------

class GridEncoder(nn.Module):
    """Deep CNN that encodes an (input, output) grid pair into a latent vector.

    Input: (batch, 2, max_H, max_W) — channel 0 = input, channel 1 = output
    Output: (batch, latent_dim) — continuous embedding
    """

    def __init__(self, in_channels: int = 2, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Multi-scale feature extraction
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 64)
        self.norm3 = nn.GroupNorm(8, 128)
        self.norm4 = nn.GroupNorm(8, 256)

        # Adaptive pooling to fixed size regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Projection to latent space
        self.projector = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode grid pair to latent vector.

        Args:
            x: (batch, 2, H, W) paired input/output grids
        Returns:
            (batch, latent_dim) embedding
        """
        x = x.float()
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.pool(h)
        h = F.relu(self.norm2(self.conv2(h)))
        h = self.pool(h)
        h = F.relu(self.norm3(self.conv3(h)))
        h = self.pool(h)
        h = F.relu(self.norm4(self.conv4(h)))
        h = self.adaptive_pool(h)
        h = h.view(h.size(0), -1)
        return self.projector(h)


# ---------------------------------------------------------------------------
# Auto-Regressive Token Decoder
# ---------------------------------------------------------------------------

class TokenDecoder(nn.Module):
    """Transformer decoder that auto-regressively generates DSL token sequences.

    Input: latent vector (batch, latent_dim)
    Output: logits over token vocabulary at each step
    """

    def __init__(
        self,
        vocab_size: int,
        latent_dim: int = 256,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        max_seq_len: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_dim = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Latent vector projection to d_model
        self.latent_proj = nn.Linear(latent_dim, d_model)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # SOS token
        self.sos_token = 0

    def forward(
        self,
        latent: torch.Tensor,
        targets: torch.Tensor | None = None,
        max_len: int | None = None,
    ) -> torch.Tensor:
        """Decode tokens from latent vector.

        Args:
            latent: (batch, latent_dim) encoder output
            targets: (batch, seq_len) target token indices (teacher forcing)
            max_len: maximum sequence length for inference
        Returns:
            (batch, seq_len, vocab_size) logits
        """
        if max_len is None:
            max_len = self.max_seq_len

        batch_size = latent.size(0)
        device = latent.device

        if targets is not None:
            # Teacher forcing mode
            seq_len = targets.size(1)
            # Embed target tokens
            tok_emb = self.token_embed(targets)  # (B, L, d_model)
            pos = self.pos_embed[:, :seq_len, :]
            tok_emb = tok_emb + pos

            # Project latent to memory
            memory = self.latent_proj(latent).unsqueeze(1)  # (B, 1, d_model)

            # Create causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

            # Decode
            decoded = self.transformer(tok_emb, memory, tgt_mask=causal_mask)
            logits = self.output_proj(decoded)
            return logits

        else:
            # Autoregressive inference
            return self._autoregressive_decode(latent, max_len)

    def _autoregressive_decode(self, latent: torch.Tensor, max_len: int) -> torch.Tensor:
        """Generate tokens one at a time."""
        batch_size = latent.size(0)
        device = latent.device

        memory = self.latent_proj(latent).unsqueeze(1)

        # Start with SOS token
        generated = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=device)

        all_logits = []
        for step in range(max_len):
            tok_emb = self.token_embed(generated)
            pos = self.pos_embed[:, :generated.size(1), :]
            tok_emb = tok_emb + pos

            causal_mask = torch.triu(
                torch.ones(generated.size(1), generated.size(1), device=device), diagonal=1
            ).bool()

            decoded = self.transformer(tok_emb, memory, tgt_mask=causal_mask)
            logits = self.output_proj(decoded[:, -1:, :])  # (B, 1, vocab)
            all_logits.append(logits)

            # Greedy next token
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)

        return torch.cat(all_logits, dim=1)  # (B, max_len, vocab)


# ---------------------------------------------------------------------------
# Full JEPA Hybrid Model
# ---------------------------------------------------------------------------

class JEPAHybridModel(nn.Module):
    """End-to-end Encoder-Decoder model for grid-to-program synthesis.

    Encoder: CNN compresses (input, output) grid pair → 256-dim latent
    Decoder: Transformer auto-regressively generates DSL token sequence
    """

    def __init__(
        self,
        vocab_size: int,
        latent_dim: int = 256,
        d_model: int = 128,
        max_seq_len: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = GridEncoder(in_channels=2, latent_dim=latent_dim)
        self.decoder = TokenDecoder(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            d_model=d_model,
            max_seq_len=max_seq_len,
        )
        self.vocab_size = vocab_size

    def forward(
        self,
        grid_pair: torch.Tensor,
        targets: torch.Tensor | None = None,
        max_len: int = 32,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            grid_pair: (batch, 2, H, W) input/output grid pair
            targets: (batch, seq_len) target token indices for teacher forcing
            max_len: max decode length
        Returns:
            (batch, seq_len, vocab_size) logits
        """
        latent = self.encoder(grid_pair)
        logits = self.decoder(latent, targets=targets, max_len=max_len)
        return logits

    @torch.no_grad()
    def generate(self, grid_pair: torch.Tensor, max_len: int = 32) -> torch.Tensor:
        """Generate token sequence from grid pair."""
        self.eval()
        latent = self.encoder(grid_pair)
        logits = self.decoder._autoregressive_decode(latent, max_len)
        return logits.argmax(dim=-1)
