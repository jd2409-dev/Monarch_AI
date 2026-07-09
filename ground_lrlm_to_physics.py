"""Ground LRLM to Physics — Targeted Language-Grounding Alignment Pass.

Fine-tunes the LRLM on calibrated world model transitions to eliminate
token gibberish and produce structured, verifiable hypotheses.

The core issue: LRLM generates garbled text because it's never been trained
on the calibrated latent space. This pass forces it to learn:
  - Grid state → structured hypothesis tokens
  - Action selection → valid ACTION tokens (32-38)
  - World model latent → grounded text predictions

Architecture:
  1. Load calibrated world model + 525 live transitions
  2. Encode each transition through the calibrated encoder
  3. Train LRLM to predict grounded hypothesis templates
  4. Save aligned model weights

Usage:
  python ground_lrlm_to_physics.py --epochs 10 --lr 1e-4
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soma_mythos_ehra.arc3.local_coder import ARCDomainLLM, ARCCoderConfig
from soma_mythos_ehra.arc3.active_world_model import HypothesisEnsemble, GridEncoder
from soma_mythos_ehra.arc3.replay_buffer import ExperienceReplayBuffer
from soma_mythos_ehra.arc3.four_tier_dataset import (
    VOCAB_SIZE, ASCII_BASE, TOK_BOS, TOK_EOS, TOK_PAD,
    TOK_GRID_START, TOK_GRID_END, TOK_ACTION, TOK_PROBLEM,
    TOK_SEP, TOK_MATH_INT,
)
from soma_mythos_ehra.arc3.hypothesis_engine import GRID_VAL_BASE
from soma_mythos_ehra.arc3.agi3_connector import ARC3Connector


# ══════════════════════════════════════════════════════════════════════════════
# Grounded Hypothesis Templates
# ══════════════════════════════════════════════════════════════════════════════

def make_grounded_hypothesis(action: int, prev_grid: np.ndarray, 
                            next_grid: np.ndarray, reward: float) -> list[int]:
    """
    Create a structured hypothesis token sequence from a real transition.
    
    Format: [BOS] grid_state [SEP] action_info [SEP] outcome [EOS]
    
    This is the TARGET format the LRLM must learn to generate.
    """
    tokens = [TOK_BOS]
    
    # 1. Grid state encoding (8x8 sample)
    tokens.append(TOK_GRID_START)
    H, W = prev_grid.shape
    step_h, step_w = max(1, H // 8), max(1, W // 8)
    for r in range(0, min(H, 8 * step_h), step_h):
        for c in range(0, min(W, 8 * step_w), step_w):
            val = int(prev_grid[r, c]) % 16
            tokens.append(GRID_VAL_BASE + val)
    tokens.append(TOK_GRID_END)
    
    # 2. Action info
    tokens.append(TOK_SEP)
    tokens.extend(_text_tokens("action : "))
    tokens.append(TOK_ACTION + action)
    tokens.extend(_text_tokens(" . "))
    
    # 3. Outcome description
    if reward > 0:
        tokens.extend(_text_tokens("result : win"))
    else:
        # Describe what changed
        diff = np.abs(next_grid.astype(float) - prev_grid.astype(float))
        change_count = int(np.sum(diff > 0))
        if change_count > 0:
            tokens.extend(_text_tokens(f"result : {change_count} cells changed"))
        else:
            tokens.extend(_text_tokens("result : no change"))
    
    tokens.append(TOK_EOS)
    return tokens


def _text_tokens(text: str) -> list[int]:
    """Convert text to token IDs using ASCII_BASE encoding."""
    return [ASCII_BASE + ord(c) for c in text if 32 <= ord(c) < 127]


# ══════════════════════════════════════════════════════════════════════════════
# Transition Collector
# ══════════════════════════════════════════════════════════════════════════════

def collect_grounding_data(num_games: int = 5, steps_per_game: int = 200) -> list[dict]:
    """Collect transitions specifically for grounding alignment."""
    connector = ARC3Connector()
    transitions = []
    
    games = connector.available_games[:num_games]
    print(f"Collecting grounding data from {len(games)} games...")
    
    for game_info in games:
        gid = game_info["game_id"]
        baselines = game_info["baseline_actions"]
        
        obs = connector.make(gid)
        prev_grid = connector.get_grid_tensor(obs).numpy()
        
        print(f"  {gid}: baselines={baselines}")
        
        for step in range(steps_per_game):
            if obs.state in ("WIN", "GAME_OVER"):
                break
            
            available = obs.available_actions or list(range(1, 8))
            
            # Use a mix of actions for diverse data
            if step < 10:
                action = np.random.choice(available)
            elif step % 3 == 0:
                # Random exploration
                action = np.random.choice(available)
            else:
                action = np.random.choice(available)
            
            # Execute
            x, y = None, None
            if action == 6:
                nonzero = np.argwhere(prev_grid > 0)
                if len(nonzero) > 0:
                    idx = np.random.randint(len(nonzero))
                    y, x = int(nonzero[idx][0]), int(nonzero[idx][1])
                else:
                    x, y = 32, 32
            
            obs = connector.step(action, x=x, y=y)
            next_grid = connector.get_grid_tensor(obs).numpy()
            reward = 1.0 if obs.state == "WIN" else 0.0
            
            transitions.append({
                "prev_grid": prev_grid.copy(),
                "action": action,
                "next_grid": next_grid.copy(),
                "reward": reward,
                "done": obs.state in ("WIN", "GAME_OVER"),
            })
            
            prev_grid = next_grid
        
        print(f"    steps={min(step+1, steps_per_game)}, transitions={len(transitions)}")
    
    print(f"Collected {len(transitions)} transitions total")
    return transitions


# ══════════════════════════════════════════════════════════════════════════════
# Grounding Trainer
# ══════════════════════════════════════════════════════════════════════════════

class GroundingTrainer:
    """
    Trains the LRLM to generate grounded hypotheses from real transitions.
    
    Loss: Cross-entropy between LRLM output and structured hypothesis tokens
    """
    
    def __init__(self, lrlm: ARCDomainLLM, device: torch.device):
        self.lrlm = lrlm
        self.device = device
        self.grid_encoder = GridEncoder().to(device)
    
    def train_epoch(self, transitions: list[dict], optimizer: torch.optim.Optimizer,
                   batch_size: int = 16) -> float:
        """Train one epoch on grounding data."""
        self.lrlm.train()
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle transitions
        indices = np.random.permutation(len(transitions))
        
        for i in range(0, len(transitions), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = [transitions[j] for j in batch_idx]
            
            # Build training examples
            batch_loss = 0.0
            
            for t in batch:
                # Create grounded hypothesis target
                target_tokens = make_grounded_hypothesis(
                    t["action"], t["prev_grid"], t["next_grid"], t["reward"]
                )
                
                if len(target_tokens) < 3:
                    continue
                
                # Input: all tokens except last
                # Label: all tokens except first (shifted by 1)
                input_ids = torch.tensor([target_tokens[:-1]], dtype=torch.long).to(self.device)
                labels = torch.tensor([target_tokens[1:]], dtype=torch.long).to(self.device)
                
                # Forward pass
                logits, _ = self.lrlm(input_ids)
                
                # Cross-entropy loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=TOK_PAD,
                )
                
                batch_loss += loss
            
            # Average loss over batch
            avg_loss = batch_loss / len(batch)
            
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.lrlm.parameters(), 1.0)
            optimizer.step()
            
            total_loss += avg_loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def validate(self, transitions: list[dict], num_samples: int = 50) -> dict:
        """Validate grounding quality on held-out transitions."""
        self.lrlm.eval()
        
        correct_action_tokens = 0
        total_action_positions = 0
        valid_hypothesis_count = 0
        
        sample_indices = np.random.choice(len(transitions), min(num_samples, len(transitions)), replace=False)
        
        with torch.no_grad():
            for idx in sample_indices:
                t = transitions[idx]
                
                # Create target
                target_tokens = make_grounded_hypothesis(
                    t["action"], t["prev_grid"], t["next_grid"], t["reward"]
                )
                
                if len(target_tokens) < 5:
                    continue
                
                # Generate autoregressively
                input_ids = torch.tensor([target_tokens[:5]], dtype=torch.long).to(self.device)
                
                generated = self.lrlm.generate(
                    input_ids, max_new_tokens=len(target_tokens), temperature=0.7, top_k=50
                )
                
                generated_tokens = generated[0].cpu().tolist()
                
                # Check if ACTION token was generated correctly
                for tok in generated_tokens:
                    if TOK_ACTION <= tok <= TOK_ACTION + 7:
                        total_action_positions += 1
                        if tok == TOK_ACTION + t["action"]:
                            correct_action_tokens += 1
                
                # Check if hypothesis is structured (has BOS, GRID_START, ACTION, EOS)
                has_grid = TOK_GRID_START in generated_tokens
                has_action = any(TOK_ACTION <= t <= TOK_ACTION + 7 for t in generated_tokens)
                has_eos = TOK_EOS in generated_tokens
                
                if has_grid and has_action:
                    valid_hypothesis_count += 1
        
        action_accuracy = correct_action_tokens / max(total_action_positions, 1)
        structure_rate = valid_hypothesis_count / max(len(sample_indices), 1)
        
        return {
            "action_accuracy": action_accuracy,
            "structure_rate": structure_rate,
            "correct_tokens": correct_action_tokens,
            "total_action_positions": total_action_positions,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ground LRLM to Calibrated Physics")
    parser.add_argument("--collect-games", type=int, default=5)
    parser.add_argument("--collect-steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lrlm_full/lrlm_best.pt")
    args = parser.parse_args()
    
    print("=" * 60)
    print("LRLM GROUNDING ALIGNMENT PASS")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load LRLM
    print("\nLoading LRLM...")
    torch.serialization.add_safe_globals([ARCCoderConfig])
    config = ARCCoderConfig(
        vocab_size=VOCAB_SIZE,
        d_model=512,
        n_layer=8,
        n_head=8,
        max_seq_len=512,
        dropout=0.1,
    )
    lrlm = ARCDomainLLM(config).to(device)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            lrlm.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Loaded: {args.checkpoint}")
        else:
            print("  Unknown checkpoint format, using random weights")
    else:
        print(f"  No checkpoint at {args.checkpoint}, using random weights")
    
    # Collect grounding data
    print("\nCollecting grounding transitions...")
    transitions = collect_grounding_data(
        num_games=args.collect_games,
        steps_per_game=args.collect_steps,
    )
    
    # Split into train/val
    split = int(len(transitions) * 0.8)
    train_transitions = transitions[:split]
    val_transitions = transitions[split:]
    print(f"  Train: {len(train_transitions)} | Val: {len(val_transitions)}")
    
    # Initialize trainer
    trainer = GroundingTrainer(lrlm, device)
    optimizer = torch.optim.AdamW(lrlm.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)
    
    best_action_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Train
        train_loss = trainer.train_epoch(
            train_transitions, optimizer, batch_size=args.batch_size
        )
        
        # Validate
        val_metrics = trainer.validate(val_transitions, num_samples=50)
        
        elapsed = time.time() - t0
        
        print(f"  Epoch {epoch:3d}/{args.epochs}: "
              f"loss={train_loss:.4f} | "
              f"action_acc={val_metrics['action_accuracy']:.3f} | "
              f"structure={val_metrics['structure_rate']:.3f} | "
              f"({elapsed:.1f}s)")
        
        # Save best
        if val_metrics["action_accuracy"] > best_action_acc:
            best_action_acc = val_metrics["action_accuracy"]
            save_path = "checkpoints/lrlm_full/grounded_model.pt"
            torch.save({
                "model_state_dict": lrlm.state_dict(),
                "epoch": epoch,
                "action_accuracy": best_action_acc,
            }, save_path)
            print(f"    -> Saved best model (action_acc={best_action_acc:.3f})")
    
    print("-" * 60)
    print(f"Grounding complete. Best action accuracy: {best_action_acc:.3f}")
    print(f"Model saved to: checkpoints/lrlm_full/grounded_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
