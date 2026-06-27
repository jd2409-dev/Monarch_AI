"""Train the Hybrid JEPA Encoder-Decoder on expanded synthetic data.

Pipeline:
1. Generate (input, output, token_sequence) triples from expanded grammar
2. Train CNN encoder + Transformer decoder with teacher forcing
3. Evaluate generation accuracy
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from soma_mythos_ehra.arc3.expanded_grammar import (
    EXTENDED_NUM_TOKENS,
    EXTENDED_TOKEN_VOCAB,
    EXTENDED_TOKEN_TO_IDX,
    sample_extended_program,
)
from soma_mythos_ehra.arc3.ast_executor import ASTExecutor
from soma_mythos_ehra.arc3.jepa_encoder import JEPAHybridModel
from soma_mythos_ehra.arc3.synthetic_grids import generate_random_grid


def ast_to_token_sequence(node, max_len: int = 32) -> list[int]:
    """Flatten an AST into a token index sequence for the decoder."""
    tokens = []

    def _walk(n):
        if len(tokens) >= max_len:
            return
        # Map node name to token index
        name = n.name
        if name in EXTENDED_TOKEN_TO_IDX:
            tokens.append(EXTENDED_TOKEN_TO_IDX[name])
        elif name in ["recolor", "flood_fill", "shift", "wrap"]:
            # Map DSL names to grammar tokens
            for tok in EXTENDED_TOKEN_TO_IDX:
                if tok.startswith(name):
                    tokens.append(EXTENDED_TOKEN_TO_IDX[tok])
                    break
        for c in n.children:
            _walk(c)

    _walk(node)
    # Pad to max_len
    while len(tokens) < max_len:
        tokens.append(0)  # SOS/pad token
    return tokens[:max_len]


def generate_hybrid_dataset(
    num_samples: int = 20000,
    max_size: int = 10,
    max_seq_len: int = 32,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate (grid_pairs, token_sequences, grammar_labels) dataset.

    Returns:
        grid_pairs: (N, 2, max_size, max_size) padded input/output pairs
        token_seqs: (N, max_seq_len) token index sequences
        grammar_labels: (N,) multi-hot grammar token labels
    """
    executor = ASTExecutor(background=0)

    grid_list = []
    seq_list = []
    label_list = []

    start_time = time.time()
    generated = 0
    attempts = 0

    while generated < num_samples:
        attempts += 1

        # Sample random program from expanded grammar
        program = sample_extended_program(max_depth=3)

        # Generate random grid
        grid = generate_random_grid(min_size=3, max_size=max_size)

        # Execute
        output = executor.execute(program, grid)
        if output is None:
            continue
        if torch.equal(grid, output):
            continue

        # Tokenize AST
        token_seq = ast_to_token_sequence(program, max_seq_len)

        # Multi-hot grammar label
        label = torch.zeros(EXTENDED_NUM_TOKENS)
        for tok_idx in token_seq:
            if tok_idx > 0 and tok_idx < EXTENDED_NUM_TOKENS:
                label[tok_idx] = 1.0

        # Pad grids
        pad_h = max_size - grid.shape[0]
        pad_w = max_size - grid.shape[1]
        padded_in = torch.nn.functional.pad(grid, (0, pad_w, 0, pad_h), value=0)
        padded_out = torch.nn.functional.pad(output, (0, pad_w, 0, pad_h), value=0)
        grid_pair = torch.stack([padded_in, padded_out])

        grid_list.append(grid_pair)
        seq_list.append(torch.tensor(token_seq, dtype=torch.long))
        label_list.append(label)

        generated += 1
        if verbose and generated % 2000 == 0:
            elapsed = time.time() - start_time
            rate = generated / elapsed if elapsed > 0 else 0
            print(f"  [{generated}/{num_samples}] {rate:.1f} samples/sec, {attempts} attempts")

    elapsed = time.time() - start_time
    if verbose:
        print(f"Dataset: {generated} samples from {attempts} attempts in {elapsed:.1f}s")
        print(f"  Rate: {generated/elapsed:.1f} samples/sec")

    grids = torch.stack(grid_list)
    seqs = torch.stack(seq_list)
    labels = torch.stack(label_list)
    return grids, seqs, labels


def train_hybrid(
    num_samples: int = 20000,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    max_seq_len: int = 32,
    max_grid_size: int = 10,
) -> None:
    """Full training pipeline."""
    print("=" * 60)
    print("Hybrid JEPA Encoder-Decoder Training")
    print("=" * 60)

    # Step 1: Generate dataset
    print("\n--- Step 1: Generating Expanded Dataset ---")
    grid_pairs, token_seqs, grammar_labels = generate_hybrid_dataset(
        num_samples=num_samples,
        max_size=max_grid_size,
        max_seq_len=max_seq_len,
    )
    print(f"grid_pairs: {grid_pairs.shape}")
    print(f"token_seqs: {token_seqs.shape}")
    print(f"grammar_labels: {grammar_labels.shape}")

    # Save
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        "grid_pairs": grid_pairs,
        "token_seqs": token_seqs,
        "grammar_labels": grammar_labels,
    }, "checkpoints/hybrid_dataset.pt")

    # Step 2: Create model
    print("\n--- Step 2: Building Model ---")
    model = JEPAHybridModel(
        vocab_size=EXTENDED_NUM_TOKENS,
        latent_dim=256,
        d_model=128,
        max_seq_len=max_seq_len,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Step 3: Train
    print("\n--- Step 3: Training ---")
    dataset = TensorDataset(grid_pairs, token_seqs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore pad token

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_tokens = 0
        total_tokens = 0

        for batch_grids, batch_seqs in loader:
            optimizer.zero_grad()

            # Teacher forcing: feed targets as input, predict next token
            input_seqs = batch_seqs[:, :-1]
            target_seqs = batch_seqs[:, 1:]

            logits = model(batch_grids, targets=input_seqs)
            # logits: (B, L, V), target: (B, L)
            loss = criterion(logits.reshape(-1, EXTENDED_NUM_TOKENS), target_seqs.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = target_seqs != 0
            correct_tokens += ((preds == target_seqs) & mask).sum().item()
            total_tokens += mask.sum().item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, token_acc={token_acc:.2%}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/hybrid_jepa_model.pt")

    print(f"Best loss: {best_loss:.4f}")

    # Step 4: Evaluate generation
    print("\n--- Step 4: Evaluation ---")
    model.eval()
    model.load_state_dict(torch.load("checkpoints/hybrid_jepa_model.pt"))

    # Generate on a batch
    sample_grids = grid_pairs[:10]
    sample_targets = token_seqs[:10]

    with torch.no_grad():
        generated = model.generate(sample_grids, max_len=max_seq_len)
        # Compare with targets
        matches = (generated == sample_targets).float().mean().item()
        exact = (generated == sample_targets).all(dim=1).float().mean().item()
        print(f"  Token accuracy: {matches:.2%}")
        print(f"  Exact sequence match: {exact:.2%}")

    print("\nTraining complete!")


if __name__ == "__main__":
    train_hybrid()
