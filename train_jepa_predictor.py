"""Train the JEPA Structure Predictor on ARC training data."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from soma_mythos_ehra.arc3.dataset_generator import generate_dataset, save_dataset
from soma_mythos_ehra.arc3.jepa_predictor import JEPAStructureRouter
from soma_mythos_ehra.arc3.template_library import build_template_library


def main():
    print("=" * 60)
    print("JEPA Structure Predictor Training")
    print("=" * 60)

    # Step 1: Generate dataset
    print("\n--- Step 1: Generating Dataset ---")
    t0 = time.time()
    X, Y = generate_dataset("ARC-AGI/data/training", limit=400)
    print(f"Dataset: X={X.shape}, Y={Y.shape}")
    print(f"Time: {time.time() - t0:.1f}s")

    # Save dataset
    save_dataset(X, Y, "checkpoints/arc_dataset.pt")

    # Step 2: Train predictor
    print("\n--- Step 2: Training Predictor ---")
    templates = build_template_library()
    num_templates = len(templates)

    router = JEPAStructureRouter(feature_dim=32, num_templates=num_templates)

    t0 = time.time()
    result = router.train(X, Y, epochs=100, lr=0.001)
    print(f"Training time: {time.time() - t0:.1f}s")

    # Save model
    router.save("checkpoints/jepa_structure_predictor.pt")
    print("Model saved to checkpoints/jepa_structure_predictor.pt")

    # Step 3: Evaluate
    print("\n--- Step 3: Evaluation ---")
    router.model.eval()
    with torch.no_grad():
        pred = router.predict(X)
        # Compute accuracy (did we predict the right template?)
        pred_binary = (pred > 0.5).float()
        exact_match = (pred_binary == Y).all(dim=1).float().mean().item()
        any_correct = ((pred_binary * Y).sum(dim=1) > 0).float().mean().item()
        print(f"Exact match: {exact_match:.2%}")
        print(f"At least one correct: {any_correct:.2%}")

    print("\nTraining complete!")


if __name__ == "__main__":
    import torch
    main()
