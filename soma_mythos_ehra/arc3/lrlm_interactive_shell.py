"""LRLM Interactive Shell — live chat with the scratch-built Large Reasoning and Language Model.

Loads the converged 5.4M-parameter ARCDomainLLM from checkpoints/lrlm_full/
and establishes a live autoregressive generation loop. Supports:
  - Essay generation (Tier 4)
  - Algorithmic reasoning (Tier 3)
  - Procedural grammar traces (Tier 2)
  - Grid physics descriptions (Tier 1)
  - Freeform conversation

All outputs are grounded in the 8192-token shared vocabulary.
No pre-trained weights. No external APIs. Pure local CUDA inference.
"""
from __future__ import annotations

import sys
import os
import math
import time

import torch
import torch.nn.functional as F

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soma_mythos_ehra.arc3.local_coder import ARCDomainLLM, ARCCoderConfig
from soma_mythos_ehra.arc3.four_tier_dataset import (
    VOCAB_SIZE, ASCII_BASE, TOK_BOS, TOK_EOS, TOK_PAD,
    TOK_GRID_START, TOK_GRID_END, TOK_ACTION, TOK_REWARD,
    TOK_REASON_START, TOK_REASON_END, TOK_TRACE_START, TOK_TRACE_END,
    TOK_PROBLEM, TOK_STEP, TOK_ANSWER, TOK_ESSAY, TOK_CODE,
    TOK_MATH_SORT, TOK_MATH_FIND, TOK_MATH_IF, TOK_MATH_ELSE,
    text_to_tokens, tokens_to_text,
)


# ══════════════════════════════════════════════════════════════════════════════
# Token vocabulary for display
# ══════════════════════════════════════════════════════════════════════════════

SPECIAL_NAMES = {
    TOK_PAD: "[PAD]", TOK_BOS: "[BOS]", TOK_EOS: "[EOS]",
    TOK_GRID_START: "[GRID]", TOK_GRID_END: "[/GRID]",
    TOK_ACTION: "[ACT]", TOK_REWARD: "[REW]",
    TOK_REASON_START: "[REASON]", TOK_REASON_END: "[/REASON]",
    TOK_TRACE_START: "[TRACE]", TOK_TRACE_END: "[/TRACE]",
    TOK_PROBLEM: "[PROBLEM]", TOK_STEP: "[STEP]",
    TOK_ANSWER: "[ANSWER]", TOK_ESSAY: "[ESSAY]",
    TOK_CODE: "[CODE]",
    TOK_MATH_SORT: "[SORT]", TOK_MATH_FIND: "[FIND]",
    TOK_MATH_IF: "[IF]", TOK_MATH_ELSE: "[ELSE]",
}


def decode_token(token_id: int) -> str:
    """Decode a single token ID to readable string."""
    if token_id in SPECIAL_NAMES:
        return SPECIAL_NAMES[token_id]
    if ASCII_BASE + 32 <= token_id <= ASCII_BASE + 127:
        return chr(token_id - ASCII_BASE)
    return f"<{token_id}>"


def decode_tokens(token_ids: list[int]) -> str:
    """Decode a sequence of token IDs to readable text."""
    parts = []
    for t in token_ids:
        parts.append(decode_token(t))
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Interactive Shell
# ══════════════════════════════════════════════════════════════════════════════

class LRLMShell:
    """Interactive terminal for the scratch-built LRLM."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/lrlm_full/lrlm_best.pt",
        device: str | None = None,
        temperature: float = 0.8,
        top_k: int = 50,
        max_gen_tokens: int = 256,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.temperature = temperature
        self.top_k = top_k
        self.max_gen_tokens = max_gen_tokens

        # Build model
        config = ARCCoderConfig(
            vocab_size=VOCAB_SIZE,
            d_model=256,
            n_layer=4,
            n_head=8,
            max_seq_len=512,
            dropout=0.0,
        )
        self.model = ARCDomainLLM(config).to(self.device)

        # Load converged weights
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                self.model.load_state_dict(ckpt["state_dict"])
            elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt)
            print(f"  Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"  WARNING: Checkpoint not found at {checkpoint_path}")
            print(f"  Running with random weights.")

        self.model.eval()
        self.param_count = sum(p.numel() for p in self.model.parameters())
        self.query_count = 0

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """Autoregressively generate a response to the prompt."""
        # Encode prompt
        prompt_tokens = text_to_tokens(prompt)
        if not prompt_tokens:
            return ""

        # Prepend BOS
        input_ids = torch.tensor([[TOK_BOS] + prompt_tokens], dtype=torch.long, device=self.device)

        generated = []
        for _ in range(self.max_gen_tokens):
            # Crop to max_seq_len
            crop = input_ids[:, -508:]  # leave room for growth

            logits, _ = self.model(crop)
            next_logits = logits[:, -1, :] / self.temperature

            # Top-k filtering
            if self.top_k > 0:
                values, _ = torch.topk(next_logits, min(self.top_k, next_logits.size(-1)))
                min_val = values[:, -1].unsqueeze(1)
                next_logits[next_logits < min_val] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            if token_id == TOK_EOS or token_id == TOK_PAD:
                break

            generated.append(token_id)

            # Append to context
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return decode_tokens(generated)

    def run(self) -> None:
        """Run the interactive shell loop."""
        print()
        print("=" * 70)
        print("  LRLM COGNITIVE CONSOLE")
        print("  Status: 210K Four-Tier Converged | Vocab: 8192 | PPL: 1.49")
        print(f"  Device: {self.device} | Params: {self.param_count / 1e6:.1f}M")
        print(f"  Temperature: {self.temperature} | Top-k: {self.top_k}")
        print("=" * 70)
        print()
        print("  Commands:")
        print("    'status'   - model info")
        print("    'stats'    - generation statistics")
        print("    'temp N'   - set temperature (0.1-2.0)")
        print("    'topk N'   - set top-k (1-100)")
        print("    'help'     - show example prompts")
        print("    'exit'     - quit")
        print()
        print("  Domains:")
        print("    Tier 1: Grid physics  - 'describe what action 3 does to a grid'")
        print("    Tier 2: Procedural    - 'apply mirror_h then grow_objects'")
        print("    Tier 3: Algorithmic   - 'sort [9,2,5] step by step'")
        print("    Tier 4: Essay/Language - 'write about pattern recognition'")
        print()
        print("-" * 70)

        while True:
            try:
                raw = input("\n[Operator] ──> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nSession terminated.")
                break

            if not raw:
                continue

            # Commands
            cmd = raw.lower()
            if cmd == "exit" or cmd == "quit":
                print("Shutting down LRLM console.")
                break
            elif cmd == "status":
                print(f"  Model: ARCDomainLLM ({self.param_count / 1e6:.1f}M params)")
                print(f"  Vocab: {VOCAB_SIZE} tokens")
                print(f"  Device: {self.device}")
                print(f"  Queries served: {self.query_count}")
                continue
            elif cmd == "stats":
                print(f"  Queries: {self.query_count}")
                print(f"  Temperature: {self.temperature}")
                print(f"  Top-k: {self.top_k}")
                continue
            elif cmd.startswith("temp "):
                try:
                    self.temperature = float(cmd.split()[1])
                    print(f"  Temperature set to {self.temperature}")
                except (IndexError, ValueError):
                    print("  Usage: temp 0.8")
                continue
            elif cmd.startswith("topk "):
                try:
                    self.top_k = int(cmd.split()[1])
                    print(f"  Top-k set to {self.top_k}")
                except (IndexError, ValueError):
                    print("  Usage: topk 50")
                continue
            elif cmd == "help":
                print("  Example prompts:")
                print()
                print("  [Tier 4 - Essay]")
                print("    write an essay on the nature of intelligence")
                print("    explain how pattern recognition works")
                print()
                print("  [Tier 3 - Algorithmic]")
                print("    sort [9, 2, 5, 7, 1] step by step")
                print("    find 5 in [1, 3, 5, 7, 9]")
                print()
                print("  [Tier 2 - Procedural]")
                print("    apply mirror_h then grow_objects to a 4x4 grid")
                print("    if grid_is_square then tessellate_2x2")
                print()
                print("  [Tier 1 - Physics]")
                print("    describe what action 3 does to a grid")
                print("    what happens when you shift objects left")
                continue

            # Generate response
            t0 = time.time()
            self.query_count += 1
            response = self.generate(raw)
            elapsed = time.time() - t0

            print(f"\n[LRLM Brain] ──> {response}")
            print(f"  ({elapsed:.2f}s, {len(response)} chars)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LRLM Interactive Shell")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lrlm_full/lrlm_best.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    shell = LRLMShell(
        checkpoint_path=args.checkpoint,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        max_gen_tokens=args.max_tokens,
    )
    shell.run()
