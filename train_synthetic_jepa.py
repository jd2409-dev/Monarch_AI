"""Train JEPA Structure Predictor on synthetic recursive grammar data."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from soma_mythos_ehra.arc3.jepa_predictor import JEPAStructureRouter
from soma_mythos_ehra.arc3.recursive_grammar import NUM_TOKENS
from soma_mythos_ehra.arc3.synthetic_dataset import generate_synthetic_dataset, save_synthetic_dataset


def main():
    print("=" * 60)
    print("JEPA Structure Predictor — Synthetic Training")
    print("=" * 60)

    # Step 1: Generate synthetic dataset
    print("\n--- Step 1: Generating Synthetic Dataset ---")
    t0 = time.time()
    X, Y, grids = generate_synthetic_dataset(
        num_samples=10000,
        min_size=3,
        max_size=10,
        max_depth=3,
        retries_per_sample=20,
        verbose=True,
    )
    print(f"\nDataset: X={X.shape}, Y={Y.shape}, grids={grids.shape}")
    print(f"Time: {time.time() - t0:.1f}s")

    # Save dataset
    save_synthetic_dataset(X, Y, grids, "checkpoints/synthetic_dataset.pt")

    # Step 2: Analyze label distribution
    print("\n--- Step 2: Label Analysis ---")
    label_counts = Y.sum(dim=0)
    non_zero = (label_counts > 0).sum().item()
    print(f"Active tokens: {non_zero}/{NUM_TOKENS}")
    top_k = torch.topk(label_counts, 10)
    for i in range(10):
        idx = top_k.indices[i].item()
        from soma_mythos_ehra.arc3.recursive_grammar import TOKEN_VOCAB
        print(f"  {TOKEN_VOCAB[idx]}: {label_counts[idx].item():.0f}")

    # Step 3: Train predictor
    print("\n--- Step 3: Training Predictor ---")
    router = JEPAStructureRouter(feature_dim=32, num_templates=NUM_TOKENS)

    t0 = time.time()
    result = router.train(X, Y, epochs=100, lr=0.001)
    print(f"Training time: {time.time() - t0:.1f}s")

    # Save model
    router.save("checkpoints/jepa_synthetic_predictor.pt")
    print("Model saved to checkpoints/jepa_synthetic_predictor.pt")

    # Step 4: Evaluate
    print("\n--- Step 4: Evaluation ---")
    router.model.eval()
    with torch.no_grad():
        pred = router.predict(X)
        pred_binary = (pred > 0.5).float()

        # Exact match
        exact_match = (pred_binary == Y).all(dim=1).float().mean().item()

        # Any correct (at least one token predicted correctly)
        any_correct = ((pred_binary * Y).sum(dim=1) > 0).float().mean().item()

        # Per-token precision/recall
        tp = (pred_binary * Y).sum(dim=0)
        fp = (pred_binary * (1 - Y)).sum(dim=0)
        fn = ((1 - pred_binary) * Y).sum(dim=0)
        precision = (tp / (tp + fp + 1e-8)).mean().item()
        recall = (tp / (tp + fn + 1e-8)).mean().item()
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"Exact match: {exact_match:.2%}")
        print(f"At least one correct: {any_correct:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1: {f1:.2%}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
