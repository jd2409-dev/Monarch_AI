"""Train hybrid encoder-decoder with longer training and larger model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from soma_mythos_ehra.arc3.expanded_grammar import EXTENDED_NUM_TOKENS
from soma_mythos_ehra.arc3.jepa_encoder import JEPAHybridModel


def train():
    d = torch.load("checkpoints/hybrid_dataset.pt")
    grid_pairs = d["grid_pairs"].float()
    token_seqs = d["token_seqs"]

    print(f"Dataset: {grid_pairs.shape}, {token_seqs.shape}")

    model = JEPAHybridModel(vocab_size=EXTENDED_NUM_TOKENS, latent_dim=512, d_model=256, max_seq_len=32)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    dataset = TensorDataset(grid_pairs, token_seqs)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_loss = float("inf")
    for epoch in range(100):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for grids, seqs in loader:
            optimizer.zero_grad()
            inp = seqs[:, :-1]
            tgt = seqs[:, 1:]
            logits = model(grids, targets=inp)
            loss = criterion(logits.reshape(-1, EXTENDED_NUM_TOKENS), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = tgt != 0
            correct += ((preds == tgt) & mask).sum().item()
            total += mask.sum().item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        acc = correct / total if total > 0 else 0

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} acc={acc:.2%} lr={scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/hybrid_jepa_model.pt")

    print(f"\nBest loss: {best_loss:.4f}")

    # Evaluate
    model.load_state_dict(torch.load("checkpoints/hybrid_jepa_model.pt"))
    model.eval()
    grids = grid_pairs[:20]
    targets = token_seqs[:20]

    with torch.no_grad():
        logits = model(grids, targets=targets[:, :-1])
        preds = logits.argmax(dim=-1)
        mask = targets[:, 1:] != 0
        tf_acc = ((preds == targets[:, 1:]) & mask).float().sum() / mask.float().sum()
        print(f"Teacher-forced accuracy: {tf_acc:.2%}")

        gen = model.generate(grids, max_len=32)
        gen_acc = (gen == targets).float().mean()
        print(f"Autoregressive accuracy: {gen_acc:.2%}")


if __name__ == "__main__":
    train()
