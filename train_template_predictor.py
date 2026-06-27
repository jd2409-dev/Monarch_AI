"""Train 104-class Template Predictor on template-bounded synthetic data."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from soma_mythos_ehra.arc3.jepa_predictor import JEPA_template_router
from soma_mythos_ehra.arc3.template_dataset import TemplateDatasetGenerator


def main():
    print("=" * 60)
    print("104-Class Template Predictor Training (Method 2)")
    print("=" * 60)

    # Step 1: Generate template-bounded dataset
    print("\n--- Step 1: Generating Template-Bounded Dataset ---")
    gen = TemplateDatasetGenerator()
    t0 = time.time()
    X, Y = gen.generate_dataset(num_samples=20000, verbose=True)
    print(f"Dataset: X={X.shape}, Y={Y.shape}")
    print(f"Time: {time.time() - t0:.1f}s")

    # Save dataset
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"X": X, "Y": Y}, "checkpoints/template_dataset.pt")
    print("Saved to checkpoints/template_dataset.pt")

    # Step 2: Analyze label distribution
    print("\n--- Step 2: Label Distribution ---")
    counts = torch.bincount(Y, minlength=gen.num_templates)
    non_zero = (counts > 0).sum().item()
    print(f"Active templates: {non_zero}/{gen.num_templates}")
    top_k = torch.topk(counts, 10)
    for i in range(10):
        idx = top_k.indices[i].item()
        print(f"  Template {idx} ({gen.templates[idx][0]}): {counts[idx].item()}")

    # Step 3: Train predictor
    print("\n--- Step 3: Training 104-Class Predictor ---")
    router = JEPA_template_router(feature_dim=32, num_classes=gen.num_templates)

    t0 = time.time()
    result = router.train(X, Y, epochs=100, lr=0.001)
    print(f"Training time: {time.time() - t0:.1f}s")

    # Save model
    router.save("checkpoints/jepa_template_predictor_104.pt")
    print("Model saved to checkpoints/jepa_template_predictor_104.pt")

    # Step 4: Evaluate
    print("\n--- Step 4: Evaluation ---")
    router.model.eval()
    with torch.no_grad():
        logits = router.predict(X)
        preds = logits.argmax(dim=1)

        # Top-1 accuracy
        top1 = (preds == Y).float().mean().item()

        # Top-5 accuracy
        top5_indices = torch.topk(logits, 5, dim=1).indices
        top5 = (top5_indices == Y.unsqueeze(1)).any(dim=1).float().mean().item()

        # Top-10 accuracy
        top10_indices = torch.topk(logits, 10, dim=1).indices
        top10 = (top10_indices == Y.unsqueeze(1)).any(dim=1).float().mean().item()

        print(f"Top-1 accuracy: {top1:.2%}")
        print(f"Top-5 accuracy: {top5:.2%}")
        print(f"Top-10 accuracy: {top10:.2%}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
