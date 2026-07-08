"""Benchmark LRLM Agent — inductive reasoning on ARC-AGI-3 interactive track.

Runs the full Scientific Method Loop:
  1. LRLM generates text hypotheses about game rules
  2. World model ensemble verifies each hypothesis
  3. Invalidated hypotheses are pruned from the search tree
  4. Validated rules accumulate in the scratchpad
  5. Scratchpad context improves future reasoning

Usage:
  python benchmark_lrlm.py --max-games 5 --episodes 3
  python benchmark_lrlm.py --stress-test
  python benchmark_lrlm.py --verification 0.5 --max-steps 300
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from soma_mythos_ehra.arc3.lrlm_agent import LRLMAgent, LRLMAgentConfig, stress_test


def run_benchmark(
    max_games: int | None = None,
    episodes: int = 3,
    max_steps: int = 200,
    verification_threshold: float = 0.3,
    confidence_threshold: float = 0.15,
    verbose: bool = True,
) -> None:
    config = LRLMAgentConfig(
        max_steps=max_steps,
        max_episodes=episodes,
        verification_threshold=verification_threshold,
        confidence_threshold=confidence_threshold,
        ensemble_size=5,
        buffer_capacity=100000,
        verbose=verbose,
    )

    agent = LRLMAgent(config)
    agent.play_all(max_games=max_games, episodes=episodes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LRLM Agent Benchmark")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--verification", type=float, default=0.3)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--stress-test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.stress_test:
        config = LRLMAgentConfig(verbose=not args.quiet)
        agent = LRLMAgent(config)
        stress_test(agent)
    else:
        run_benchmark(
            max_games=args.max_games,
            episodes=args.episodes,
            max_steps=args.max_steps,
            verification_threshold=args.verification,
            confidence_threshold=args.confidence,
            verbose=not args.quiet,
        )
