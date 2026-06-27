"""Template-Bounded Dataset Generator — generates data for 104-class template prediction.

Method 2: Pick a template first, run it on random grids, emit 1-hot labels.
This produces clean categorical training data for a 104-class CrossEntropy predictor.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from soma_mythos_ehra.arc3.dataset_generator import extract_structural_features
from soma_mythos_ehra.arc3.dsl_kernel import DSLKernel
from soma_mythos_ehra.arc3.template_library import build_template_library
from soma_mythos_ehra.arc3.synthetic_grids import generate_random_grid


class TemplateDatasetGenerator:
    """Generates (features, template_label) pairs by running templates on random grids."""

    def __init__(self) -> None:
        self.templates = build_template_library()
        self.kernel = DSLKernel(background=0)
        self.num_templates = len(self.templates)

    def generate_labeled_pair(self, max_retries: int = 20) -> tuple[torch.Tensor, int] | None:
        """Generate one (features, template_index) pair.

        Picks a random template, runs it on random grids until one produces
        a valid transformation, then returns the structural features and label.
        """
        for _ in range(max_retries):
            # 1. Pick a random template
            target_idx = torch.randint(0, self.num_templates, (1,)).item()
            name, prog = self.templates[target_idx]

            # 2. Generate random input grid
            grid = generate_random_grid(min_size=3, max_size=10)

            # 3. Execute template on grid
            output = self.kernel.execute(prog, grid)
            if output is None:
                continue

            # 4. Verify transformation actually changed the grid
            if torch.equal(grid, output):
                continue

            # 5. Extract features and return
            features = extract_structural_features(grid, output)
            return features, target_idx

        return None

    def generate_dataset(
        self,
        num_samples: int = 20000,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate full dataset of (features, label) pairs.

        Returns:
            X: (N, 32) feature vectors
            Y: (N,) integer template labels (0 to num_templates-1)
        """
        X_list = []
        Y_list = []

        start_time = time.time()
        generated = 0
        attempts = 0

        while generated < num_samples:
            attempts += 1
            result = self.generate_labeled_pair()
            if result is None:
                continue

            features, label = result
            X_list.append(features)
            Y_list.append(label)
            generated += 1

            if verbose and generated % 1000 == 0:
                elapsed = time.time() - start_time
                rate = generated / elapsed if elapsed > 0 else 0
                print(f"  [{generated}/{num_samples}] {rate:.1f} samples/sec, "
                      f"{attempts} attempts, {elapsed:.1f}s")

        elapsed = time.time() - start_time
        if verbose:
            print(f"\nDataset: {generated} samples from {attempts} attempts in {elapsed:.1f}s")
            print(f"  Rate: {generated/elapsed:.1f} samples/sec")
            print(f"  Success rate: {generated/attempts:.2%}")

        X = torch.stack(X_list)
        Y = torch.tensor(Y_list, dtype=torch.long)
        return X, Y
