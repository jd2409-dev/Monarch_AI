"""Train Full LRLM — from-scratch training pipeline for the four-tier dataset.

Trains the ARCDomainLLM on interleaved data from all four tiers:
  - Tier 1: Core physics interactions
  - Tier 2: Synthetic procedural traces
  - Tier 3: Algorithmic logic chains
  - Tier 4: Structural text corpora

Uses cosine LR schedule with warmup, gradient clipping, and mixed-precision
training for fast convergence.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soma_mythos_ehra.arc3.local_coder import ARCDomainLLM, ARCCoderConfig
from soma_mythos_ehra.arc3.four_tier_dataset import FourTierDataset, VOCAB_SIZE
from soma_mythos_ehra.arc3.interleaved_data_loader import InterleavedDataLoader


# ══════════════════════════════════════════════════════════════════════════════
# Training Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    # Model
    vocab_size: int = VOCAB_SIZE  # 8192
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    max_seq_len: int = 512
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 50
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 4

    # Data
    train_data: str = "data/four_tier/train_tokens.pt"
    val_data: str = "data/four_tier/val_tokens.pt"

    # Checkpointing
    save_dir: str = "checkpoints/lrlm_full"
    save_every: int = 5  # epochs
    eval_every: int = 1  # epochs

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ══════════════════════════════════════════════════════════════════════════════
# Learning Rate Schedule
# ══════════════════════════════════════════════════════════════════════════════

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════════════

class LRLMTrainer:
    """Full training pipeline for the LRLM."""

    def __init__(self, config: TrainConfig | None = None) -> None:
        self.config = config or TrainConfig()
        self.device = torch.device(self.config.device)

        # Build model
        model_config = ARCCoderConfig(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout,
        )
        self.model = ARCDomainLLM(model_config).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if self.config.mixed_precision and self.device.type == "cuda" else None

        # Load data
        self.train_tokens = self._load_data(self.config.train_data)
        self.val_tokens = self._load_data(self.config.val_data) if os.path.exists(self.config.val_data) else None

        # Create loaders
        self.train_loader = InterleavedDataLoader.from_tokens(
            self.train_tokens, batch_size=self.config.batch_size
        )
        if self.val_tokens is not None:
            self.val_loader = InterleavedDataLoader.from_tokens(
                self.val_tokens, batch_size=self.config.batch_size
            )
        else:
            self.val_loader = None

        # Scheduler
        total_steps = len(self.train_loader) * self.config.max_epochs // self.config.grad_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, self.config.warmup_steps, total_steps
        )

        # State
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = []

    def _load_data(self, path: str) -> torch.Tensor:
        """Load token tensor from disk."""
        if not os.path.exists(path):
            print(f"Data not found at {path}. Generating synthetic data...")
            return self._generate_synthetic_data()
        return torch.load(path, weights_only=True)

    def _generate_synthetic_data(self) -> torch.Tensor:
        """Generate synthetic training data on the fly."""
        print("Generating synthetic four-tier data (10K samples)...")
        from soma_mythos_ehra.arc3.four_tier_dataset import FourTierDataset as GenDataset

        gen = GenDataset(
            tier1_count=2500,
            tier2_count=2500,
            tier3_count=2500,
            tier4_count=2500,
            max_seq_len=self.config.max_seq_len,
        )
        samples = gen.generate_all()
        tokens = torch.zeros(len(samples), self.config.max_seq_len, dtype=torch.long)
        for i, s in enumerate(samples):
            t = torch.tensor(s.tokens[:self.config.max_seq_len], dtype=torch.long)
            tokens[i, :len(t)] = t
        return tokens

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        batch_count = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)

            # Input: all tokens except last
            # Target: all tokens except first (shifted by 1)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            # Forward pass with mixed precision
            if self.scaler:
                with torch.amp.autocast("cuda"):
                    logits, loss = self.model(inputs, targets)
                    loss = loss / self.config.grad_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                logits, loss = self.model(inputs, targets)
                loss = loss / self.config.grad_accumulation_steps
                loss.backward()

            total_loss += loss.item() * self.config.grad_accumulation_steps
            total_tokens += targets.numel()
            batch_count += 1

            # Gradient accumulation step
            if (batch_idx + 1) % self.config.grad_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            if batch_idx % 100 == 0:
                avg_loss = total_loss / batch_count
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"loss={avg_loss:.4f} lr={lr:.6f} step={self.global_step}")

        avg_loss = total_loss / max(batch_count, 1)
        perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow

        return {
            "train_loss": avg_loss,
            "train_perplexity": perplexity,
            "total_tokens": total_tokens,
            "global_step": self.global_step,
        }

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return {"val_loss": 0.0, "val_perplexity": 0.0}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        batch_count = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            _, loss = self.model(inputs, targets)
            total_loss += loss.item()
            total_tokens += targets.numel()
            batch_count += 1

        avg_loss = total_loss / max(batch_count, 1)
        perplexity = math.exp(min(avg_loss, 20))

        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
            "total_tokens": total_tokens,
        }

    def save_checkpoint(self, epoch: int, metrics: dict) -> None:
        """Save model checkpoint."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, f"lrlm_epoch_{epoch}.pt")
        self.model.save(path)

        # Save training state
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": {
                "vocab_size": self.config.vocab_size,
                "d_model": self.config.d_model,
                "n_layer": self.config.n_layer,
                "n_head": self.config.n_head,
                "max_seq_len": self.config.max_seq_len,
            },
        }
        torch.save(state, os.path.join(self.config.save_dir, "training_state.pt"))

        # Save metrics
        with open(os.path.join(self.config.save_dir, "metrics.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"  Saved checkpoint: {path}")

    def train(self) -> None:
        """Full training loop."""
        print("=" * 60)
        print("LRLM Full Training Pipeline")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model: {self.model.count_parameters() / 1e6:.1f}M params")
        print(f"Train samples: {len(self.train_tokens)}")
        if self.val_tokens is not None:
            print(f"Val samples: {len(self.val_tokens)}")
        print(f"Batches/epoch: {len(self.train_loader)}")
        print(f"Total epochs: {self.config.max_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Grad accum: {self.config.grad_accumulation_steps}")
        print(f"Effective batch: {self.config.batch_size * self.config.grad_accumulation_steps}")
        print(f"LR: {self.config.learning_rate}")
        print(f"Mixed precision: {self.config.mixed_precision}")
        print("=" * 60)

        for epoch in range(1, self.config.max_epochs + 1):
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Evaluate
            val_metrics = {}
            if epoch % self.config.eval_every == 0:
                val_metrics = self.evaluate()

            elapsed = time.time() - t0

            # Log
            metrics = {**train_metrics, **val_metrics, "epoch": epoch, "time": elapsed}
            self.history.append(metrics)

            print(f"Epoch {epoch}/{self.config.max_epochs} ({elapsed:.1f}s) "
                  f"train_loss={train_metrics['train_loss']:.4f} "
                  f"train_ppl={train_metrics['train_perplexity']:.2f}", end="")
            if val_metrics:
                print(f" val_loss={val_metrics['val_loss']:.4f} "
                      f"val_ppl={val_metrics['val_perplexity']:.2f}", end="")
            print()

            # Save best
            if val_metrics and val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(epoch, metrics)
                print(f"  New best val_loss: {self.best_val_loss:.4f}")

            # Periodic save
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(epoch, metrics)

        # Final save
        self.save_checkpoint(self.config.max_epochs, metrics)
        print("=" * 60)
        print(f"Training complete. Best val_loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.config.save_dir}")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Full LRLM from scratch")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n-layer", type=int, default=8, help="Number of layers")
    parser.add_argument("--n-head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--save-dir", type=str, default="checkpoints/lrlm_full", help="Save directory")
    parser.add_argument("--train-data", type=str, default="data/four_tier/train_tokens.pt")
    parser.add_argument("--val-data", type=str, default="data/four_tier/val_tokens.pt")
    parser.add_argument("--no-mp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--generate-data", action="store_true", help="Generate data first")
    args = parser.parse_args()

    # Generate data if requested
    if args.generate_data or not os.path.exists(args.train_data):
        print("Generating four-tier dataset...")
        from soma_mythos_ehra.arc3.four_tier_dataset import FourTierDataset as GenDataset

        gen = GenDataset(
            tier1_count=100_000,
            tier2_count=50_000,
            tier3_count=10_000,
            tier4_count=50_000,
            max_seq_len=args.max_seq_len,
        )
        gen.generate_all(save_dir="data/four_tier")

    config = TrainConfig(
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        max_seq_len=args.max_seq_len,
        save_dir=args.save_dir,
        train_data=args.train_data,
        val_data=args.val_data,
        mixed_precision=not args.no_mp,
    )

    trainer = LRLMTrainer(config)
    trainer.train()
