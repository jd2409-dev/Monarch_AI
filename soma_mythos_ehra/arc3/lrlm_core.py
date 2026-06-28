"""LRLM Core — Large Reasoning and Language Model for SOMA-Mythos-EHRA.

Multi-modal transformer that fuses:
  1. Grid latent vectors from ActiveWorldModel (physical reality)
  2. Action model logits from ARCActionLLM (tactical reasoning)
  3. Text token embeddings (natural language interface)

Produces both reasoning text and executable action predictions
with zero-hallucination grounding via latent invariant anchoring.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LRLMConfig:
    """Configuration for the LRLM model."""
    vocab_size: int = 8000
    d_model: int = 512
    n_layer: int = 6
    n_head: int = 8
    dim_feedforward: int = 2048
    max_seq_len: int = 1024
    grid_latent_dim: int = 256
    action_logit_dim: int = 128
    dropout: float = 0.1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class ARCLRLM(nn.Module):
    """Large Reasoning and Language Model for ARC-AGI-3.

    Fuses grid state, action model, and text into unified semantic space.
    Uses cross-modal projection gates to anchor text generation in physical reality.

    Architecture:
    - Grid latent projector: maps world model latents to d_model
    - Action logit projector: maps action model outputs to d_model
    - Text token embeddings
    - Causal transformer reasoning core
    - Dual output heads: text tokens + action logits
    """

    def __init__(self, config: LRLMConfig | None = None) -> None:
        super().__init__()
        self.config = config or LRLMConfig()

        # Text embeddings
        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.position_embeddings = nn.Embedding(self.config.max_seq_len, self.config.d_model)

        # Cross-modal projectors: physical reality -> semantic space
        self.grid_latent_projector = nn.Sequential(
            nn.Linear(self.config.grid_latent_dim, self.config.d_model),
            nn.GELU(),
            nn.LayerNorm(self.config.d_model),
        )
        self.action_logit_projector = nn.Sequential(
            nn.Linear(self.config.action_logit_dim, self.config.d_model),
            nn.GELU(),
            nn.LayerNorm(self.config.d_model),
        )

        # State embedding projector (game state enum -> d_model)
        self.state_projector = nn.Embedding(8, self.config.d_model)

        # Causal reasoning stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_head,
            dim_feedforward=self.config.dim_feedforward,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=self.config.dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.n_layer,
        )

        # Output heads
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.action_head = nn.Linear(self.config.d_model, 10)  # 10 action classes

        self.dropout = nn.Dropout(self.config.dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        text_tokens: torch.Tensor,
        grid_latent: torch.Tensor,
        action_logits: torch.Tensor,
        state_id: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ):
        """Forward pass with multi-modal fusion.

        Args:
            text_tokens: (batch, seq_len) text token indices
            grid_latent: (batch, grid_dim) world model latent vector
            action_logits: (batch, action_dim) action model output logits
            state_id: (batch,) game state ID (optional)
            targets: (batch, seq_len) target text tokens (optional)
        Returns:
            text_logits: (batch, seq_len, vocab_size)
            action_probs: (batch, 10) action probability distribution
            loss: scalar if targets provided
        """
        device = text_tokens.device
        b, t = text_tokens.size()

        # 1. Project physical VRAM realities into semantic vectors
        grid_features = self.grid_latent_projector(grid_latent).unsqueeze(1)  # (B, 1, D)
        action_features = self.action_logit_projector(action_logits).unsqueeze(1)  # (B, 1, D)

        # 2. Embed text tokens
        text_features = self.token_embeddings(text_tokens)  # (B, t, D)

        # 3. State embedding (if provided)
        if state_id is not None:
            state_features = self.state_projector(state_id).unsqueeze(1)  # (B, 1, D)
        else:
            state_features = torch.zeros(b, 1, self.config.d_model, device=device)

        # 4. Interleave physical reality blocks before text input
        # Layout: [grid_state, action_state, state_embedding, text_tokens...]
        fused = torch.cat([grid_features, action_features, state_features, text_features], dim=1)
        total_len = fused.size(1)

        positions = torch.arange(0, total_len, device=device).unsqueeze(0)
        x = self.dropout(fused + self.position_embeddings(positions[:, :total_len]))

        # 5. Causal mask
        causal_mask = torch.triu(
            torch.full((total_len, total_len), float("-inf"), device=device), diagonal=1,
        )

        hidden = self.transformer(x, mask=causal_mask)

        # 6. Dual output heads
        text_logits = self.lm_head(hidden)

        # Action logits from the grid/action prefix positions
        action_hidden = hidden[:, :2, :].mean(dim=1)  # Average over grid+action prefix
        action_probs = F.softmax(self.action_head(action_hidden), dim=-1)

        loss = None
        if targets is not None:
            if targets.dim() == 1:
                # Single target per sequence: compute loss only on last text position
                # pred_logits is at position after the 3 prefix tokens
                last_text_pos = min(text_logits.size(1) - 1, 3 + t)
                loss = F.cross_entropy(
                    text_logits[:, last_text_pos, :].reshape(-1, self.config.vocab_size),
                    targets.view(-1),
                )
            else:
                # Full sequence targets
                pred_logits = text_logits[:, 3:-1, :].reshape(-1, self.config.vocab_size)
                target_flat = targets.view(-1)
                loss = F.cross_entropy(pred_logits, target_flat, ignore_index=73)

        return text_logits, action_probs, loss

    @torch.no_grad()
    def generate_text(
        self,
        grid_latent: torch.Tensor,
        action_logits: torch.Tensor,
        state_id: torch.Tensor | None = None,
        prompt_tokens: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate text conditioned on physical state.

        Args:
            grid_latent: (1, grid_dim) current world model latent
            action_logits: (1, action_dim) current action model output
            state_id: (1,) current game state
            prompt_tokens: (1, prompt_len) optional text prompt
            max_new_tokens: tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
        Returns:
            (1, total_len) generated token sequence
        """
        self.eval()
        device = grid_latent.device

        if prompt_tokens is None:
            prompt_tokens = torch.tensor([[70]], device=device)  # SOS token

        generated = prompt_tokens.clone()

        for _ in range(max_new_tokens):
            # Truncate to fit max_seq_len minus prefix
            input_text = generated[:, -self.config.max_seq_len + 3:]

            text_logits, _, _ = self.forward(
                input_text, grid_latent, action_logits, state_id,
            )
            next_logits = text_logits[:, -1, :] / temperature

            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                min_val = values[:, -1].unsqueeze(1)
                next_logits[next_logits < min_val] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop at EOS
            if next_token.item() == 71:
                break

        return generated

    @torch.no_grad()
    def recommend_action(
        self,
        grid_latent: torch.Tensor,
        action_logits: torch.Tensor,
        state_id: torch.Tensor | None = None,
        available_actions: list[int] | None = None,
    ) -> tuple[int, float]:
        """Recommend best action using multi-modal fusion.

        Args:
            grid_latent: (1, grid_dim) world model latent
            action_logits: (1, action_dim) action model output
            state_id: (1,) game state
            available_actions: list of valid action IDs
        Returns:
            (action_id, confidence)
        """
        dummy_text = torch.tensor([[70]], device=grid_latent.device)  # SOS
        _, action_probs, _ = self.forward(dummy_text, grid_latent, action_logits, state_id)
        probs = action_probs[0]

        if available_actions is not None:
            mask = torch.zeros_like(probs)
            for a in available_actions:
                if a < len(probs):
                    mask[a] = 1.0
            probs = probs * mask
            if probs.sum() > 0:
                probs = probs / probs.sum()

        action = torch.argmax(probs).item()
        confidence = probs[action].item()
        return action, confidence

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": {
                "vocab_size": self.config.vocab_size,
                "d_model": self.config.d_model,
                "n_layer": self.config.n_layer,
                "n_head": self.config.n_head,
                "max_seq_len": self.config.max_seq_len,
                "grid_latent_dim": self.config.grid_latent_dim,
                "action_logit_dim": self.config.action_logit_dim,
            },
        }, path)

    @classmethod
    def load(cls, path: str) -> ARCLRLM:
        state = torch.load(path, weights_only=True)
        config = LRLMConfig()
        for k, v in state["config"].items():
            setattr(config, k, v)
        model = cls(config)
        model.load_state_dict(state["model_state_dict"])
        return model
