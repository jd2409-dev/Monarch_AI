"""Benchmark for ARC-AGI-3 Interactive Agent v2.

Runs the full active-inference loop with online learning:
- Experience replay + world model training
- Code evolution for rule inference
- Multi-level curriculum progression
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from soma_mythos_ehra.arc3.interactive_agent import InteractiveAgent, AgentConfig


def compute_rhae(agent_actions: int, human_baseline: int) -> float:
    """Compute Relative Human Action Efficiency."""
    if agent_actions <= 0:
        return 0.0
    rhae = (human_baseline / agent_actions) ** 2
    return min(rhae, 1.15)


def run_benchmark(
    max_games: int | None = None,
    episodes_per_game: int = 5,
    verbose: bool = True,
) -> None:
    """Run the interactive agent benchmark with full learning loop."""
    print("=" * 60)
    print("ARC-AGI-3 Interactive Agent v2 Benchmark")
    print("With online learning + code evolution + curriculum")
    print("=" * 60)

    config = AgentConfig(
        max_steps=300,
        max_episodes=episodes_per_game,
        exploration_rate=0.3,
        temperature=1.0,
        ensemble_size=3,
        latent_dim=256,
        buffer_capacity=50000,
        train_steps_per_episode=50,
        evolve_every_n_episodes=3,
        mastery_threshold=3,
        verbose=verbose,
    )

    agent = InteractiveAgent(config)
    games = agent.connector.available_games

    if max_games:
        games = games[:max_games]

    print(f"\nGames: {len(games)}")
    print(f"Episodes per game: {episodes_per_game}")

    all_stats = []
    total_won = 0

    for i, game in enumerate(games):
        gid = game["game_id"]
        title = game["title"]
        baselines = game["baseline_actions"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(games)}] {title} ({gid})")
            print(f"  Tags: {game['tags']}")
            print(f"  Human baselines: {baselines}")

        stats_list = agent.play_game(gid, episodes_per_game)
        all_stats.extend(stats_list)

        final = stats_list[-1] if stats_list else None
        if final and final.won:
            total_won += 1

        # RHAE calculation
        if final:
            avg_actions = sum(final.actions_taken) / max(len(final.actions_taken), 1)
            if baselines:
                avg_baseline = sum(baselines[:max(final.levels_completed, 1)]) / max(final.levels_completed, 1)
                rhae = compute_rhae(int(avg_actions), int(avg_baseline))
            else:
                rhae = 0.0

            status = "WON" if final.won else "LOST"
            if verbose:
                print(f"\n  Final: {status} | steps={final.total_steps} | "
                      f"levels={final.levels_completed} | time={final.episode_time:.1f}s")
                if final.train_metrics:
                    last_train = final.train_metrics[-1]
                    print(f"  Training: loss={last_train.total_loss:.4f}")
                if final.code_scores:
                    print(f"  Code evolution: best_score={max(final.code_scores):.2f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Games played: {len(games)}")
    print(f"Games won: {total_won}/{len(games)} ({100*total_won/len(games):.1f}%)")
    print(f"Buffer size: {len(agent.buffer)}")
    print(f"Code hypotheses: {len(agent.code_evolver.population)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_benchmark(
        max_games=args.max_games,
        episodes_per_game=args.episodes,
        verbose=not args.quiet,
    )
