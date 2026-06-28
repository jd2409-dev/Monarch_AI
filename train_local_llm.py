"""Train Local Domain LLM — generates synthetic trajectories, trains causal transformer.

Pipeline:
1. Generate synthetic (grid, action, reward) trajectories from grammar programs
2. Tokenize trajectories into integer sequences
3. Train ARCDomainLLM with next-token prediction
4. Save trained model for inference in the active-inference loop
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from soma_mythos_ehra.arc3.local_coder import ARCDomainLLM, ARCCoderConfig
from soma_mythos_ehra.arc3.trajectory_tokenizer import (
    batch_tokenize, tokenize_trajectory, TOKEN_MAP, PAD, SOS, EOS,
)
from soma_mythos_ehra.arc3.ast_executor import ASTExecutor
from soma_mythos_ehra.arc3.recursive_grammar import sample_program, ALL_PRIMITIVES, NUM_TOKENS, TOKEN_VOCAB
from soma_mythos_ehra.arc3.synthetic_grids import generate_random_grid


def generate_synthetic_trajectories(
    num_trajectories: int = 5000,
    max_steps: int = 10,
    max_grid_size: int = 8,
    verbose: bool = True,
) -> list[dict]:
    """Generate synthetic trajectories by executing grammar programs on random grids.

    Each trajectory is a sequence of (grid, action, reward) transitions
    that resulted from applying a grammar program.
    """
    executor = ASTExecutor(background=0)
    trajectories = []

    start_time = time.time()
    generated = 0
    attempts = 0

    while generated < num_trajectories:
        attempts += 1

        # Sample random program
        program = sample_program(max_depth=2)

        # Generate random grid
        grid = generate_random_grid(min_size=3, max_size=max_grid_size)

        # Execute program step by step (simulate as sequential actions)
        prev_grids = [grid.numpy().copy()]
        actions = []
        rewards = []
        states = []
        grammar_tokens = []

        # Map program to action sequence
        action_seq = _program_to_actions(program)

        current_grid = grid.clone()
        for step in range(min(len(action_seq), max_steps)):
            action = action_seq[step]

            # Simulate action effect
            next_grid = _simulate_action(current_grid, action)
            reward = 0.0
            state = "NOT_FINISHED"

            # Check if grid changed significantly
            if not torch.equal(current_grid, next_grid):
                # Check if this looks like a "win" (grid matches some pattern)
                if step == len(action_seq) - 1:
                    reward = 1.0
                    state = "WIN"

            prev_grids.append(next_grid.numpy().copy())
            actions.append(action)
            rewards.append(reward)
            states.append(state)

            current_grid = next_grid

        # Encode program as grammar tokens
        grammar_tokens = _encode_program_tokens(program)

        if actions:
            trajectories.append({
                "prev_grids": prev_grids,
                "actions": actions,
                "rewards": rewards,
                "states": states,
                "grammar_tokens": grammar_tokens,
            })
            generated += 1

        if verbose and generated % 1000 == 0:
            elapsed = time.time() - start_time
            rate = generated / elapsed if elapsed > 0 else 0
            print(f"  [{generated}/{num_trajectories}] {rate:.0f} traj/sec, {attempts} attempts")

    elapsed = time.time() - start_time
    if verbose:
        print(f"Generated {generated} trajectories in {elapsed:.1f}s ({generated/elapsed:.0f}/sec)")

    return trajectories


def _program_to_actions(program) -> list[int]:
    """Convert a grammar AST to a sequence of action numbers."""
    actions = []

    def _walk(node):
        name = node.name if hasattr(node, "name") else str(node)
        # Map primitive names to actions
        action_map = {
            "rotate_90": 1, "rotate_180": 1, "rotate_270": 1,
            "flip_h": 2, "flip_v": 2, "transpose": 2,
            "scale_2": 3, "scale_3": 3,
            "recolor_map": 4, "fill_holes": 4, "flood_fill": 4,
            "shift_down": 5, "shift_up": 5, "shift_left": 5, "shift_right": 5,
            "wrap_h": 6, "wrap_v": 6,
        }
        if name in action_map:
            actions.append(action_map[name])
        for c in (node.children if hasattr(node, "children") else []):
            _walk(c)

    _walk(program)
    return actions if actions else [1]


def _simulate_action(grid: torch.Tensor, action: int) -> torch.Tensor:
    """Simulate an action on a grid."""
    from soma_mythos_ehra.arc3.transforms import (
        apply_rotate_90, apply_flip_h, apply_scale_up,
        apply_shift_objects, apply_fill_holes,
    )
    result = grid.clone()
    try:
        if action == 1:
            result = apply_rotate_90(grid)
        elif action == 2:
            result = apply_flip_h(grid)
        elif action == 3:
            result = apply_scale_up(grid, 2)
        elif action == 4:
            result = apply_fill_holes(grid)
        elif action == 5:
            result = apply_shift_objects(grid, 1, 0)
        elif action == 6:
            result = apply_shift_objects(grid, 0, 1)
    except Exception:
        pass
    return result


def _encode_program_tokens(program) -> list[int]:
    """Encode a program AST into grammar token IDs."""
    tokens = []

    def _walk(node):
        name = node.name if hasattr(node, "name") else str(node)
        if name in TOKEN_MAP.grammar_to_id:
            tokens.append(TOKEN_MAP.grammar_to_id[name])
        for c in (node.children if hasattr(node, "children") else []):
            _walk(c)

    _walk(program)
    return tokens[:20]  # Cap at 20 grammar tokens


def prepare_training_data(
    trajectories: list[dict],
    max_len: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert trajectories to training tensors."""
    input_ids, target_ids = batch_tokenize(trajectories, max_len=max_len)
    return input_ids, target_ids


def train_local_llm(
    num_trajectories: int = 10000,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    max_seq_len: int = 256,
    verbose: bool = True,
) -> ARCDomainLLM:
    """Full training pipeline."""
    print("=" * 60)
    print("Training Local Domain LLM for ARC-AGI-3")
    print("=" * 60)

    # Step 1: Generate trajectories
    print("\n--- Step 1: Generating Synthetic Trajectories ---")
    trajectories = generate_synthetic_trajectories(
        num_trajectories=num_trajectories,
        max_steps=8,
        verbose=verbose,
    )

    # Step 2: Prepare data
    print("\n--- Step 2: Tokenizing ---")
    input_ids, target_ids = prepare_training_data(trajectories, max_len=max_seq_len)
    print(f"input_ids: {input_ids.shape}, target_ids: {target_ids.shape}")

    # Save dataset
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"input_ids": input_ids, "target_ids": target_ids}, "checkpoints/local_llm_dataset.pt")

    # Step 3: Build model
    print("\n--- Step 3: Building Model ---")
    config = ARCCoderConfig(
        vocab_size=TOKEN_MAP.vocab_size,
        d_model=256,
        n_layer=6,
        n_head=8,
        max_seq_len=max_seq_len,
    )
    model = ARCDomainLLM(config)

    # Step 4: Train
    print("\n--- Step 4: Training ---")
    dataset = TensorDataset(input_ids, target_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inp, tgt in loader:
            optimizer.zero_grad()
            logits, loss = model(inp, targets=tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            # Token accuracy
            preds = logits.argmax(dim=-1)
            mask = tgt != 0
            correct += ((preds == tgt) & mask).sum().item()
            total += mask.sum().item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        acc = correct / total if total > 0 else 0

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={acc:.2%}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save("checkpoints/local_arc_llm.pt")

    print(f"\nBest loss: {best_loss:.4f}")

    # Step 5: Evaluate generation
    print("\n--- Step 5: Evaluation ---")
    model = ARCDomainLLM.load("checkpoints/local_arc_llm.pt")
    model.eval()

    # Generate from a sample trajectory context
    sample_input = input_ids[:4]
    with torch.no_grad():
        generated = model.generate(sample_input, max_new_tokens=32, temperature=0.8)
        print(f"  Input shape: {sample_input.shape}")
        print(f"  Generated shape: {generated.shape}")
        # Check if generated tokens are valid grammar
        for i in range(min(4, generated.shape[0])):
            tokens = generated[i].tolist()
            grammar = [TOKEN_MAP.id_to_grammar.get(t) for t in tokens if t in TOKEN_MAP.id_to_grammar]
            print(f"  Sample {i}: {len(grammar)} grammar tokens from {len(tokens)} total")

    print("\nTraining complete!")
    return model


if __name__ == "__main__":
    train_local_llm()
