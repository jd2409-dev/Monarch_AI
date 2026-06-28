"""Benchmark for ARC-AGI-3 Interactive Agent.

Evaluates the agent on available ARC-AGI-3 environments and computes
RHAE (Relative Human Action Efficiency) scores.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from soma_mythos_ehra.arc3.interactive_agent import InteractiveAgent, AgentConfig


def compute_rhae(agent_actions: int, human_baseline: int) -> float:
    """Compute Relative Human Action Efficiency for one level.

    RHAE = (human_baseline / agent_actions) ^ 2
    Capped at 1.15 (agent can be at most 15% more efficient than human).
    """
    if agent_actions <= 0:
        return 0.0
    rhae = (human_baseline / agent_actions) ** 2
    return min(rhae, 1.15)


def run_benchmark(max_games: int | None = None, verbose: bool = True) -> None:
    """Run the interactive agent benchmark."""
    print("=" * 60)
    print("ARC-AGI-3 Interactive Agent Benchmark")
    print("=" * 60)

    config = AgentConfig(
        max_steps=300,
        exploration_rate=0.3,
        temperature=1.0,
        ensemble_size=3,
        latent_dim=256,
        verbose=verbose,
    )

    agent = InteractiveAgent(config)
    games = agent.connector.available_games

    if max_games:
        games = games[:max_games]

    print(f"\nAvailable games: {len(games)}")

    results = []
    total_won = 0
    total_rhae = 0.0
    total_baseline = 0

    for i, game in enumerate(games):
        gid = game["game_id"]
        title = game["title"]
        baselines = game["baseline_actions"]

        if verbose:
            print(f"\n--- [{i+1}/{len(games)}] {title} ({gid}) ---")
            print(f"  Tags: {game['tags']}")
            print(f"  Human baselines: {baselines}")

        agent.play(gid)
        stats = agent.stats

        # Compute RHAE per level
        level_rhae = []
        for lvl in range(min(stats.levels_completed, len(baselines))):
            h_base = baselines[lvl]
            a_actions = stats.total_actions // max(stats.levels_completed, 1)
            rhae = compute_rhae(a_actions, h_base)
            level_rhae.append(rhae)

        avg_rhae = sum(level_rhae) / len(level_rhae) if level_rhae else 0.0
        total_rhae += avg_rhae
        total_baseline += sum(baselines)

        result = {
            "game_id": gid,
            "title": title,
            "won": stats.won,
            "levels_completed": stats.levels_completed,
            "total_actions": stats.total_actions,
            "rhae": avg_rhae,
            "time": stats.episode_time,
        }
        results.append(result)

        if stats.won:
            total_won += 1

        status = "WON" if stats.won else "LOST"
        print(f"  {status}: levels={stats.levels_completed}, "
              f"actions={stats.total_actions}, "
              f"RHAE={avg_rhae:.2f}, time={stats.episode_time:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Games played: {len(results)}")
    print(f"Games won: {total_won}/{len(results)} ({100*total_won/len(results):.1f}%)")
    print(f"Average RHAE: {total_rhae/len(results):.2f}")
    print(f"{'='*60}")

    # Detailed results
    print("\nDetailed Results:")
    for r in results:
        status = "WIN" if r["won"] else "LOSE"
        print(f"  {r['title']:8s} {status:5s} levels={r['levels_completed']} "
              f"actions={r['total_actions']:3d} RHAE={r['rhae']:.2f} "
              f"time={r['time']:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_benchmark(max_games=args.max_games, verbose=not args.quiet)
