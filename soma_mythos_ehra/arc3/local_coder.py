"""Local Domain Causal Transformer — compact 15M param model for ARC-AGI-3.

A GPT-style causal transformer that reads trajectory tokens
(state summaries, actions, rewards) and emits valid grammar/AST tokens.

Architecture: 6 layers, 8 heads, 256 dim → ~15M parameters.
Input: tokenized trajectory from interactive exploration
Output: next-token logits over grammar vocabulary
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ARCCoderConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 8
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = False  # No bias in LayerNorm (nanoGPT style)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with flash attention support."""

    def __init__(self, config: ARCCoderConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: ARCCoderConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual."""

    def __init__(self, config: ARCCoderConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ARCDomainLLM(nn.Module):
    """Domain-specific causal transformer for ARC-AGI-3.

    ~15M parameters. Reads trajectory tokens, outputs grammar predictions.
    Designed for sub-millisecond inference inside the active-inference loop.
    """

    def __init__(self, config: ARCCoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or ARCCoderConfig()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.d_model),
            wpe = nn.Embedding(self.config.max_seq_len, self.config.d_model),
            drop = nn.Dropout(self.config.dropout),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            ln_f = nn.LayerNorm(self.config.d_model, bias=self.config.bias),
        ))
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"ARCDomainLLM: {n_params/1e6:.1f}M parameters")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            idx: (batch, seq_len) token indices
            targets: (batch, seq_len) target indices for loss computation
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: cross-entropy loss (None if targets not provided)
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.max_seq_len, f"Sequence length {T} > max {self.config.max_seq_len}"

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,  # ignore PAD
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressively generate token sequences.

        Args:
            idx: (batch, seq_len) initial context tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
        Returns:
            (batch, seq_len + max_new_tokens) generated sequence
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        self.train()
        return idx

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str) -> None:
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str) -> ARCDomainLLM:
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint["config"]
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model
