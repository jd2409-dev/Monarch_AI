"""Local Action Model — causal transformer for action prediction.

Lightweight (~2-5M params) autoregressive model that looks at environment
history (states, actions, rewards) and predicts high-information next actions.
Trained via behavioral cloning on replay buffer transitions.
Runs entirely on local GPU in sub-millisecond time.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ARCActionLLM(nn.Module):
    """Causal transformer for action sequence prediction.

    Architecture:
    - Token embedding + learned positional encoding
    - N-layer transformer encoder with causal mask
    - Linear head projecting to action vocabulary

    Input: trajectory token sequence (batch, seq_len)
    Output: logits over action vocabulary (batch, seq_len, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 256,
        n_layer: int = 4,
        n_head: int = 8,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Forward pass.

        Args:
            idx: (batch, seq_len) token indices
            targets: (batch, seq_len) target token indices (optional, for training)
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar tensor if targets provided, else None
        """
        b, t = idx.size()
        assert t <= self.max_seq_len, f"Sequence length {t} > max {self.max_seq_len}"

        positions = torch.arange(0, t, device=idx.device).unsqueeze(0)
        x = self.dropout(self.token_emb(idx) + self.pos_emb(positions))

        # Causal mask: prevent attending to future tokens
        mask = torch.triu(
            torch.full((t, t), float("-inf"), device=idx.device), diagonal=1
        )

        hidden = self.transformer(x, mask=mask)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            if targets.dim() == 1:
                # Single target per sequence: compute loss only on last position
                loss = F.cross_entropy(
                    logits[:, -1, :].view(-1, self.vocab_size),
                    targets.view(-1),
                )
            else:
                # Full sequence targets: standard next-token prediction
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    targets.view(-1),
                    ignore_index=73,  # PAD token
                )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        max_new_tokens: int = 8,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """Autoregressively generate action tokens from context.

        Args:
            context: (batch, seq_len) initial token sequence
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling cutoff
        Returns:
            (batch, seq_len + max_new_tokens) generated sequence
        """
        self.eval()
        device = context.device
        generated = context.clone()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            input_seq = generated[:, -self.max_seq_len:]

            logits, _ = self.forward(input_seq)
            next_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                min_values = values[:, -1].unsqueeze(1)
                next_logits[next_logits < min_values] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    @torch.no_grad()
    def predict_action_probs(
        self,
        context: torch.Tensor,
        temperature: float = 0.8,
    ) -> torch.Tensor:
        """Predict action probability distribution from context.

        Args:
            context: (batch, seq_len) token sequence
        Returns:
            (batch, vocab_size) probability distribution over next token
        """
        self.eval()
        input_seq = context[:, -self.max_seq_len:]
        logits, _ = self.forward(input_seq)
        next_logits = logits[:, -1, :] / temperature
        return F.softmax(next_logits, dim=-1)

    @torch.no_grad()
    def recommend_action(
        self,
        context: torch.Tensor,
        available_actions: list[int] | None = None,
        temperature: float = 0.8,
    ) -> tuple[int, float]:
        """Recommend the best next action given context.

        Args:
            context: (1, seq_len) token sequence
            available_actions: list of valid action IDs (masks invalid actions)
            temperature: sampling temperature
        Returns:
            (action_id, confidence)
        """
        probs = self.predict_action_probs(context.unsqueeze(0) if context.dim() == 1 else context, temperature)
        probs = probs[0]  # Remove batch dim

        if available_actions is not None:
            # Mask unavailable actions
            mask = torch.zeros_like(probs)
            for a in available_actions:
                if a < len(probs):
                    mask[a] = 1.0
            probs = probs * mask
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                # Fallback: uniform over available
                probs = mask / mask.sum()

        action = torch.argmax(probs).item()
        confidence = probs[action].item()
        return action, confidence

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        torch.save({
            "model_state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "max_seq_len": self.max_seq_len,
        }, path)

    @classmethod
    def load(cls, path: str) -> ARCActionLLM:
        state = torch.load(path, weights_only=True)
        model = cls(
            vocab_size=state["vocab_size"],
            d_model=state["d_model"],
            max_seq_len=state["max_seq_len"],
        )
        model.load_state_dict(state["model_state_dict"])
        return model


class ActionModelTrainer:
    """Trains the action model on replay buffer transitions via behavioral cloning."""

    def __init__(
        self,
        model: ARCActionLLM,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
    ) -> None:
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5,
        )

    def train_on_buffer(
        self,
        buffer_transitions: list[dict],
        tokenizer,
        epochs: int = 5,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> list[float]:
        """Train on replay buffer transitions.

        Args:
            buffer_transitions: list of transition dicts from replay buffer
            tokenizer: GameTrajectoryTokenizer instance
            epochs: number of training epochs
            batch_size: batch size
            verbose: print progress
        Returns:
            list of epoch losses
        """
        if not buffer_transitions:
            return []

        self.model.train()
        losses = []

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            # Create training pairs: (context, next_action)
            for i in range(0, len(buffer_transitions) - 1, batch_size):
                batch = buffer_transitions[i : i + batch_size]
                if len(batch) < 2:
                    continue

                # Build token sequences from consecutive transitions
                input_tokens = []
                target_tokens = []

                for j in range(len(batch)):
                    t = batch[j]
                    # Encode current step
                    step_tokens = tokenizer.encode_step(
                        action=t["action"],
                        reward=t["reward"],
                        game_state="WIN" if t.get("done", False) else "NOT_FINISHED",
                        prev_grid=t.get("prev_grid"),
                        next_grid=t.get("next_grid"),
                    )
                    input_tokens.append(step_tokens)

                    # Target is next action (or same action for last in batch)
                    if j + 1 < len(batch):
                        target_action = batch[j + 1]["action"]
                    else:
                        target_action = t["action"]
                    target_tokens.append(target_action)

                input_tensor = torch.tensor(input_tokens, dtype=torch.long)
                target_tensor = torch.tensor(target_tokens, dtype=torch.long)

                device = next(self.model.parameters()).device
                input_tensor = input_tensor.to(device)
                target_tensor = target_tensor.to(device)

                self.optimizer.zero_grad()
                logits, loss = self.model(input_tensor, targets=target_tensor)

                if loss is not None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1

            self.scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)
            losses.append(avg_loss)

            if verbose:
                print(f"  Action model epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

        return losses
