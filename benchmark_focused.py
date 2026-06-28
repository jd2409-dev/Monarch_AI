"""Focused Benchmark — deep exploration of a single game to find winning strategy.

Strategy:
1. Run many episodes on the same game
2. Track which action sequences lead to progress (grid changes)
3. Build winning code hypotheses from observed patterns
4. Use trajectory replay when a winning sequence is found
"""
from __future__ import annotations

import sys
import time
import argparse

sys.path.insert(0, ".")


def run_focused(game_id: str, max_episodes: int = 10, verbose: bool = True):
    from soma_mythos_ehra.arc3.interactive_agent import InteractiveAgent, AgentConfig

    config = AgentConfig(
        max_steps=200,
        max_episodes=1,
        exploration_rate=0.5,
        ensemble_size=3,
        train_steps_per_episode=100,
        evolve_every_n_episodes=1,
        verbose=verbose,
    )

    agent = InteractiveAgent(config)

    print(f"=== Focused Exploration: {game_id} ===")
    print(f"Episodes: {max_episodes}")
    print(f"LLM: {'Yes' if agent.code_evolver.has_llm else 'No'}")
    print()

    # Track best results
    best_reward = 0.0
    best_actions = []
    best_code = None
    all_wins = []

    for ep in range(max_episodes):
        print(f"\n--- Episode {ep+1}/{max_episodes} ---")

        agent.play(game_id)
        stats = agent.stats

        if stats.won:
            all_wins.append(stats.actions_taken[:])
            print(f"  *** WIN! Actions: {stats.actions_taken}")
            if stats.efficiency > best_reward:
                best_reward = stats.efficiency
                best_actions = stats.actions_taken[:]

        # Check code evolution
        if agent.code_evolver.best_hypothesis:
            score = agent.code_evolver.best_hypothesis.score
            if score > 0:
                print(f"  Code score: {score:.3f} ({agent.code_evolver.best_hypothesis.source})")
                if score > (best_reward or 0):
                    best_code = agent.code_evolver.best_hypothesis.code
                    best_reward = score

        # Check hypothesis
        top = agent.hypothesis_mgr.get_top_hypotheses(1)[0]
        print(f"  Hypothesis: {top.name} ({top.probability:.3f})")
        print(f"  Buffer: {len(agent.buffer)} | Steps: {stats.total_steps}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY for {game_id}")
    print(f"{'='*60}")
    print(f"Wins: {len(all_wins)}/{max_episodes}")
    print(f"Best reward: {best_reward:.3f}")
    if best_actions:
        print(f"Best action sequence: {best_actions}")
    if best_code:
        print(f"Best code:\n{best_code}")

    # Try trajectory replay if we won
    if all_wins:
        print(f"\nWon with action sequence: {all_wins[0]}")
        print("Attempting trajectory replay...")

        # Play again with replay
        config2 = AgentConfig(
            max_steps=200,
            max_episodes=3,
            exploration_rate=0.1,
            ensemble_size=3,
            train_steps_per_episode=50,
            verbose=verbose,
        )
        agent2 = InteractiveAgent(config2)
        if __import__('os').path.exists("checkpoints/world_model.pt"):
            agent2.load("checkpoints/world_model.pt")

        for ep in range(3):
            agent2.play(game_id)
            stats2 = agent2.stats
            if stats2.won:
                print(f"  Replay WIN at episode {ep+1}! Efficiency: {stats2.efficiency:.3f}")
            else:
                print(f"  Replay episode {ep+1}: {stats2.final_state}")

    return len(all_wins), best_actions, best_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20-9607627b")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_focused(args.game, args.episodes, not args.quiet)
