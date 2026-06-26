"""JEPA Training Script.

Trains the JEPA world model using a VICReg-hybrid objective:
- Alignment loss: predicted latent should match true next state latent
- Contrastive loss: true next state should have lower energy than negatives
- Variance regularization: prevent representation collapse

Usage:
    python -m soma_mythos_ehra.training.train_jepa \\
        --recordings-dir recordings/ \\
        --output-dir checkpoints/ \\
        --epochs 50 \\
        --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from soma_mythos_ehra.soma.jepa import JEPAWorldModel
from soma_mythos_ehra.training.jepa_dataset import ARCRecordingDataset, create_contrastive_pairs


def alignment_loss(z_pred: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and true latent vectors."""
    return F.mse_loss(z_pred, z_true)


def contrastive_loss(
    energy_pos: torch.Tensor,
    energy_neg: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Hinge loss: positive pairs should have lower energy than negatives."""
    return F.relu(energy_pos.unsqueeze(1) - energy_neg + margin).mean()


def variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg variance loss: prevent collapse by enforcing std > eps."""
    std = z.std(dim=0) + eps
    return torch.relu(eps - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance loss: decorrelate features."""
    B, D = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (B - 1)
    diag = torch.diag(cov)
    off_diag = cov - torch.diag(diag)
    return (off_diag ** 2).sum() / D


def train_epoch(
    model: JEPAWorldModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_negatives: int = 4,
    align_weight: float = 1.0,
    contrast_weight: float = 0.5,
    var_weight: float = 0.1,
    cov_weight: float = 0.05,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_align = 0.0
    total_contrast = 0.0
    total_var = 0.0
    total_cov = 0.0
    total_loss = 0.0
    num_batches = 0

    for states, actions, next_states in dataloader:
        states = states.to(device)
        actions = actions.to(device)
        next_states = next_states.to(device)

        B = states.shape[0]

        # Encode states
        z_t = model.encode(states)
        z_next = model.encode(next_states)
        z_hat = model.predict_latent(z_t, actions)

        # Alignment: predicted latent should match true next state latent
        loss_align = alignment_loss(z_hat, z_next)

        # Contrastive: true next state should have lower energy than negatives
        energy_pos = model.compute_energy(states, actions, next_states)

        # Sample random negatives
        neg_indices = torch.randint(0, B, (B, num_negatives), device=device)
        neg_states = next_states[neg_indices]
        neg_actions = actions.unsqueeze(1).expand(-1, num_negatives).reshape(-1)
        neg_states_flat = neg_states.reshape(-1, *neg_states.shape[2:])
        states_expanded = states.unsqueeze(1).expand(-1, num_negatives, *states.shape[1:]).reshape(-1, *states.shape[1:])
        energy_neg = model.compute_energy(states_expanded, neg_actions, neg_states_flat)

        loss_contrast = contrastive_loss(energy_pos, energy_neg)

        # Variance and covariance regularization
        loss_var = variance_loss(z_next)
        loss_cov = covariance_loss(z_next)

        # Total loss
        loss = (
            align_weight * loss_align
            + contrast_weight * loss_contrast
            + var_weight * loss_var
            + cov_weight * loss_cov
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_align += loss_align.item()
        total_contrast += loss_contrast.item()
        total_var += loss_var.item()
        total_cov += loss_cov.item()
        total_loss += loss.item()
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "alignment": total_align / max(num_batches, 1),
        "contrastive": total_contrast / max(num_batches, 1),
        "variance": total_var / max(num_batches, 1),
        "covariance": total_cov / max(num_batches, 1),
    }


def evaluate(
    model: JEPAWorldModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_align = 0.0
    total_energy_pos = 0.0
    total_energy_neg = 0.0
    num_batches = 0

    with torch.no_grad():
        for states, actions, next_states in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)

            B = states.shape[0]

            z_t = model.encode(states)
            z_next = model.encode(next_states)
            z_hat = model.predict_latent(z_t, actions)

            loss_align = alignment_loss(z_hat, z_next)
            energy_pos = model.compute_energy(states, actions, next_states).mean()

            # Sample negatives
            neg_indices = torch.randint(0, B, (B, 4), device=device)
            neg_states = next_states[neg_indices]
            neg_actions = actions.unsqueeze(1).expand(-1, 4).reshape(-1)
            neg_states_flat = neg_states.reshape(-1, *neg_states.shape[2:])
            states_expanded = states.unsqueeze(1).expand(-1, 4, *states.shape[1:]).reshape(-1, *states.shape[1:])
            energy_neg = model.compute_energy(states_expanded, neg_actions, neg_states_flat).mean()

            total_align += loss_align.item()
            total_energy_pos += energy_pos.item()
            total_energy_neg += energy_neg.item()
            num_batches += 1

    return {
        "alignment": total_align / max(num_batches, 1),
        "energy_positive": total_energy_pos / max(num_batches, 1),
        "energy_negative": total_energy_neg / max(num_batches, 1),
        "energy_gap": (total_energy_neg - total_energy_pos) / max(num_batches, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JEPA world model")
    parser.add_argument("--recordings-dirs", type=str, required=True, help="Comma-separated list of directories with recording JSONL files")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"), help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-negatives", type=int, default=4, help="Number of negative samples")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--grid-h", type=int, default=64, help="Grid height")
    parser.add_argument("--grid-w", type=int, default=64, help="Grid width")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse multiple directories
    dirs = [Path(d.strip()) for d in args.recordings_dirs.split(",")]
    print(f"Loading recordings from {len(dirs)} directories...")

    # Load dataset
    dataset = ARCRecordingDataset(dirs, target_h=args.grid_h, target_w=args.grid_w)
    print(f"Loaded {len(dataset)} transitions")

    if len(dataset) == 0:
        print("ERROR: No transitions found. Check recordings directory.")
        sys.exit(1)

    # Split into train/val
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    model = JEPAWorldModel(num_symbols=17, num_actions=8, latent_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            num_negatives=args.num_negatives,
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        # Log metrics
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"align={train_metrics['alignment']:.4f} | "
            f"contrast={train_metrics['contrastive']:.4f} | "
            f"val_align={val_metrics['alignment']:.4f} | "
            f"energy_gap={val_metrics['energy_gap']:.4f} | "
            f"lr={lr:.2e}"
        )

        # Record history
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_alignment": train_metrics["alignment"],
            "train_contrastive": train_metrics["contrastive"],
            "val_alignment": val_metrics["alignment"],
            "val_energy_positive": val_metrics["energy_positive"],
            "val_energy_negative": val_metrics["energy_negative"],
            "val_energy_gap": val_metrics["energy_gap"],
            "lr": lr,
        })

        # Save best model
        if val_metrics["alignment"] < best_val_loss:
            best_val_loss = val_metrics["alignment"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": {
                    "num_symbols": 17,
                    "num_actions": 8,
                    "latent_dim": 64,
                    "grid_h": args.grid_h,
                    "grid_w": args.grid_w,
                },
            }
            torch.save(checkpoint, args.output_dir / "best_jepa.pt")
            print(f"  -> Saved best model (val_align={best_val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save(checkpoint, args.output_dir / f"jepa_epoch_{epoch:03d}.pt")

    # Save final model
    torch.save(checkpoint, args.output_dir / "final_jepa.pt")

    # Save training history
    with (args.output_dir / "training_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val alignment: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
