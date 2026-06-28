"""Benchmark for ARC-AGI-3 Interactive Agent v3 — full active-inference loop."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from soma_mythos_ehra.arc3.interactive_agent import InteractiveAgent, AgentConfig


def run_benchmark(
    max_games: int | None = None,
    episodes_per_game: int = 8,
    verbose: bool = True,
) -> None:
    print("=" * 60)
    print("ARC-AGI-3 Agent v3 Benchmark")
    print("LLM code evolution + efficiency optimization + scaled training")
    print("=" * 60)

    config = AgentConfig(
        max_steps=400,
        max_episodes=episodes_per_game,
        exploration_rate=0.3,
        ensemble_size=5,
        latent_dim=256,
        buffer_capacity=100000,
        train_steps_per_episode=100,
        evolve_every_n_episodes=2,
        mastery_threshold=3,
        use_llm=True,
        verbose=verbose,
    )

    agent = InteractiveAgent(config)
    games = agent.connector.available_games

    if max_games:
        games = games[:max_games]

    print(f"\nGames: {len(games)} | Episodes: {episodes_per_game}")
    print(f"Ensemble: {config.ensemble_size} models | LLM: {config.use_llm}")

    total_won = 0

    for i, game in enumerate(games):
        gid = game["game_id"]
        baselines = game["baseline_actions"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(games)}] {gid}")
            print(f"  Tags: {game['tags']}, Baselines: {baselines}")

        stats_list = agent.play_game(gid, episodes_per_game)
        final = stats_list[-1] if stats_list else None

        if final and final.won:
            total_won += 1

        if final and verbose:
            status = "WON" if final.won else "LOST"
            rhae_str = f"RHAE={final.efficiency:.2f}" if final.efficiency > 0 else ""
            print(f"\n  {status} | steps={final.total_steps} | "
                  f"levels={final.levels_completed} | {rhae_str}")

    print(f"\n{'='*60}")
    print(f"RESULT: {total_won}/{len(games)} won ({100*total_won/max(len(games),1):.1f}%)")
    print(f"Buffer: {len(agent.buffer)} | {agent.efficiency.get_efficiency_report()}")
    print(agent.curriculum.report())
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_benchmark(max_games=args.max_games, episodes_per_game=args.episodes, verbose=not args.quiet)
