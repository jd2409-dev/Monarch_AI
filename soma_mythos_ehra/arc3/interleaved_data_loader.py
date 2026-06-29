"""Interleaved Data Loader — 25/25/25/25 mixed batches from Four-Tier Dataset.

Prevents catastrophic forgetting by mixing all four data streams in every batch.
Each batch contains exactly 25% Tier 1 (physics), 25% Tier 2 (procedural),
25% Tier 3 (logic), and 25% Tier 4 (text) samples.
"""
from __future__ import annotations

import random
from typing import Iterator

import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class FourTierDataset(Dataset):
    """Dataset that loads pre-generated four-tier token tensors."""

    def __init__(self, tokens: torch.Tensor) -> None:
        """Initialize from token tensor.

        Args:
            tokens: (N, max_seq_len) tensor of token IDs
        """
        self.tokens = tokens
        self.length = tokens.shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tokens[idx]


class BalancedTierSampler(Sampler):
    """Sampler that ensures 25/25/25/25 balance in every batch.

    Maintains separate index lists for each tier and yields
    balanced batches by drawing equal numbers from each tier.
    """

    def __init__(
        self,
        tier_labels: list[int],
        batch_size: int,
        tier_counts: dict[int, int] | None = None,
    ) -> None:
        """
        Args:
            tier_labels: list of tier IDs (1-4) for each sample
            batch_size: desired batch size (must be divisible by 4)
            tier_counts: optional override for tier sizes
        """
        assert batch_size % 4 == 0, "batch_size must be divisible by 4"

        self.batch_size = batch_size
        self.tier_per_batch = batch_size // 4

        # Group indices by tier
        self.tier_indices: dict[int, list[int]] = {1: [], 2: [], 3: [], 4: []}
        for i, tier in enumerate(tier_labels):
            if tier in self.tier_indices:
                self.tier_indices[tier].append(i)

        # Calculate number of full batches
        min_tier = min(len(v) for v in self.tier_indices.values())
        self.num_batches = min_tier // self.tier_per_batch
        self.total_samples = self.num_batches * batch_size

    def __iter__(self) -> Iterator[int]:
        # Shuffle each tier's indices
        for tier in self.tier_indices:
            random.shuffle(self.tier_indices[tier])

        for batch_idx in range(self.num_batches):
            batch_indices = []
            for tier in [1, 2, 3, 4]:
                start = batch_idx * self.tier_per_batch
                end = start + self.tier_per_batch
                tier_idx = self.tier_indices[tier][start:end]
                batch_indices.extend(tier_idx)
            yield from batch_indices

    def __len__(self) -> int:
        return self.total_samples


class InterleavedDataLoader:
    """Creates balanced data loaders with 25/25/25/25 tier mixing.

    Usage:
        loader = InterleavedDataLoader.from_tokens(
            train_tokens, batch_size=32
        )
        for batch in loader:
            # batch contains exactly 8 samples from each tier
            loss = model(batch)
    """

    def __init__(
        self,
        dataset: FourTierDataset,
        tier_labels: list[int],
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

        sampler = BalancedTierSampler(tier_labels, batch_size)

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.dataloader)

    def __len__(self) -> int:
        return len(self.dataloader)

    @classmethod
    def from_tokens(
        cls,
        tokens: torch.Tensor,
        batch_size: int = 32,
        tier_labels: list[int] | None = None,
    ) -> InterleavedDataLoader:
        """Create loader from token tensor.

        Args:
            tokens: (N, max_seq_len) token tensor
            batch_size: batch size (must be divisible by 4)
            tier_labels: tier ID per sample. If None, auto-assigns based on position.
        """
        dataset = FourTierDataset(tokens)

        if tier_labels is None:
            # Auto-assign tiers based on token patterns
            tier_labels = []
            for i in range(tokens.shape[0]):
                t = tokens[i]
                # Detect tier by first significant token
                first_meaningful = t[t > 0][0].item() if (t > 0).any() else 0

                if 10 <= first_meaningful <= 14:  # GRID_START, LATENT, ACTION, REWARD
                    tier_labels.append(1)
                elif 17 <= first_meaningful <= 18:  # TRACE_START/END
                    tier_labels.append(2)
                elif 19 <= first_meaningful <= 21:  # PROBLEM, STEP, ANSWER
                    tier_labels.append(3)
                elif first_meaningful in (23, 24):  # ESSAY, TITLE
                    tier_labels.append(4)
                else:
                    # Default: assign cyclically
                    tier_labels.append((i % 4) + 1)

        return cls(dataset, tier_labels, batch_size)

    @classmethod
    def from_file(
        cls,
        path: str,
        batch_size: int = 32,
    ) -> InterleavedDataLoader:
        """Load from saved .pt file and auto-assign tiers."""
        tokens = torch.load(path, weights_only=True)
        return cls.from_tokens(tokens, batch_size)


# ══════════════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create dummy data
    N = 1000
    seq_len = 512
    tokens = torch.randint(0, 8192, (N, seq_len))

    # Create tier labels (25% each)
    labels = [1] * 250 + [2] * 250 + [3] * 250 + [4] * 250
    random.shuffle(labels)

    loader = InterleavedDataLoader(
        FourTierDataset(tokens), labels, batch_size=32
    )

    print(f"Batches per epoch: {len(loader)}")
    for i, batch in enumerate(loader):
        print(f"Batch {i}: {batch.shape}")
        if i >= 2:
            break
    print("OK")
