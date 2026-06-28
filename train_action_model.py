"""Train Action Model — behavioral cloning on replay buffer transitions.

Collects transitions from interactive agent runs, then trains the local
action transformer to predict next actions from trajectory history.
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def collect_transitions(num_games: int = 3, episodes: int = 2, verbose: bool = True):
    """Run agent on games to collect transitions in the buffer."""
    from soma_mythos_ehra.arc3.interactive_agent import InteractiveAgent, AgentConfig

    config = AgentConfig(
        max_steps=200,
        max_episodes=episodes,
        ensemble_size=3,
        train_steps_per_episode=50,
        verbose=verbose,
    )
    agent = InteractiveAgent(config)

    games = ["ls20-9607627b", "re86-8af5384d", "ka59-38d34dbb"][:num_games]

    print(f"Collecting transitions from {len(games)} games x {episodes} episodes...")
    for gid in games:
        print(f"\n  Playing {gid}...")
        agent.play_game(gid, episodes)

    print(f"\n  Buffer: {len(agent.buffer)} transitions")
    return agent


def train_action_model(agent, epochs: int = 10, verbose: bool = True):
    """Train the action model on collected buffer transitions."""
    import torch
    from soma_mythos_ehra.arc3.local_action_model import ARCActionLLM, ActionModelTrainer
    from soma_mythos_ehra.arc3.game_tokenizer import GameTrajectoryTokenizer

    model = ARCActionLLM(vocab_size=128, d_model=256, n_layer=4, n_head=8, max_seq_len=64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"\nAction Model: {model.count_parameters():,} params on {device}")

    tokenizer = GameTrajectoryTokenizer(max_seq_len=64)
    trainer = ActionModelTrainer(model, learning_rate=3e-4)

    # Convert buffer to list of dicts
    transitions = []
    for t in agent.buffer.buffer:
        transitions.append({
            "action": t.action,
            "reward": t.reward,
            "prev_grid": t.prev_grid,
            "next_grid": t.next_grid,
            "done": t.done,
            "level": t.level,
        })

    print(f"Training on {len(transitions)} transitions...")
    losses = trainer.train_on_buffer(
        transitions, tokenizer,
        epochs=epochs, batch_size=32, verbose=verbose,
    )

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/action_model.pt")
    print(f"\n  Saved to checkpoints/action_model.pt")

    return model, losses


def train_lrlm(agent, epochs: int = 5, verbose: bool = True):
    """Train the LRLM on buffer data with self-supervised objectives."""
    import torch
    import torch.nn as nn
    from soma_mythos_ehra.arc3.lrlm_core import ARCLRLM, LRLMConfig
    from soma_mythos_ehra.arc3.game_tokenizer import GameTrajectoryTokenizer

    config = LRLMConfig(
        vocab_size=128,
        d_model=256,
        n_layer=4,
        n_head=8,
        max_seq_len=64,
        grid_latent_dim=256,
        action_logit_dim=128,
    )
    model = ARCLRLM(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"\nLRLM: {model.count_parameters():,} params on {device}")

    tokenizer = GameTrajectoryTokenizer(max_seq_len=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Convert buffer
    transitions = []
    for t in agent.buffer.buffer:
        transitions.append({
            "action": t.action,
            "reward": t.reward,
            "prev_grid": t.prev_grid,
            "next_grid": t.next_grid,
            "done": t.done,
            "level": t.level,
        })

    if len(transitions) < 32:
        print("  Not enough data for LRLM training")
        return model

    print(f"Training LRLM on {len(transitions)} transitions...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(transitions) - 1, 16):
            batch = transitions[i:i + 16]
            if len(batch) < 2:
                continue

            # Build token sequences
            input_tokens = []
            targets = []
            for t in batch:
                tokens = tokenizer.encode_step(
                    action=t["action"],
                    reward=t["reward"],
                    game_state="WIN" if t["done"] else "NOT_FINISHED",
                    prev_grid=t["prev_grid"],
                    next_grid=t["next_grid"],
                )
                input_tokens.append(tokens)
                targets.append(t["action"])

            input_tensor = torch.tensor(input_tokens, dtype=torch.long).to(device)
            target_tensor = torch.tensor(targets, dtype=torch.long).to(device)

            # Dummy latents (will be real during integration)
            grid_latent = torch.randn(1, 256).to(device)
            action_logits = torch.randn(1, 128).to(device)

            optimizer.zero_grad()
            text_logits, action_probs, loss = model(
                input_tensor[:, :32],  # Use first 32 tokens as text
                grid_latent.expand(len(batch), -1),
                action_logits.expand(len(batch), -1),
                targets=target_tensor,
            )

            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if verbose:
            print(f"  LRLM epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/lrlm.pt")
    print(f"\n  Saved LRLM to checkpoints/lrlm.pt")

    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Action Model + LRLM Training Pipeline")
    print("=" * 60)

    # Step 1: Collect transitions
    agent = collect_transitions(
        num_games=args.games,
        episodes=args.episodes,
        verbose=not args.quiet,
    )

    # Step 2: Train action model
    action_model, losses = train_action_model(
        agent, epochs=args.epochs, verbose=not args.quiet,
    )

    # Step 3: Train LRLM
    lrlm = train_lrlm(
        agent, epochs=max(3, args.epochs // 2), verbose=not args.quiet,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Action Model: {action_model.count_parameters():,} params")
    print(f"  LRLM: {lrlm.count_parameters():,} params")
    print(f"  Buffer: {len(agent.buffer)} transitions")
    print(f"  Action model final loss: {losses[-1]:.4f}" if losses else "")
    print("=" * 60)


if __name__ == "__main__":
    main()
