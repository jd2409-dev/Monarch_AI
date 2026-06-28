"""Interactive Agent — the main agent loop for ARC-AGI-3 environments.

Ties together the connector, world model, explorer, and hypothesis manager
into a coherent agent that explores environments, builds mental models,
and solves puzzles through active experimentation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import numpy as np

from soma_mythos_ehra.arc3.agi3_connector import ARC3Connector, FrameObservation, EpisodeRecord
from soma_mythos_ehra.arc3.active_world_model import HypothesisEnsemble
from soma_mythos_ehra.arc3.info_max_explorer import InfoMaxExplorer, ActionPlan
from soma_mythos_ehra.arc3.hypothesis_manager import HypothesisManager


@dataclass
class AgentConfig:
    """Configuration for the interactive agent."""
    max_steps: int = 500
    exploration_rate: float = 0.3
    temperature: float = 1.0
    ensemble_size: int = 3
    latent_dim: int = 256
    num_actions: int = 8
    early_stop_on_win: bool = True
    verbose: bool = True


@dataclass
class AgentStats:
    """Statistics from an agent run."""
    game_id: str = ""
    total_steps: int = 0
    levels_completed: int = 0
    won: bool = False
    final_state: str = "NOT_FINISHED"
    episode_time: float = 0.0
    actions_taken: list[int] = field(default_factory=list)
    hypothesis_history: list[list[str]] = field(default_factory=list)


class InteractiveAgent:
    """Main agent that explores ARC-AGI-3 environments.

    The agent:
    1. Observes the grid state
    2. Maintains hypotheses about environment rules
    3. Selects actions that maximize information gain
    4. Updates beliefs based on observed transitions
    5. Exploits learned rules to solve puzzles
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.connector = ARC3Connector()
        self.ensemble = HypothesisEnsemble(
            num_models=self.config.ensemble_size,
            latent_dim=self.config.latent_dim,
            num_actions=self.config.num_actions,
        )
        self.explorer = InfoMaxExplorer(
            self.ensemble,
            exploration_rate=self.config.exploration_rate,
            temperature=self.config.temperature,
        )
        self.hypothesis_mgr = HypothesisManager()
        self.stats = AgentStats()

    def play(self, game_id: str) -> EpisodeRecord:
        """Play one environment from start to finish.

        Args:
            game_id: The game environment to play (e.g., "ls20")
        Returns:
            EpisodeRecord with full trajectory
        """
        self.stats = AgentStats(game_id=game_id)
        start_time = time.time()

        # Reset environment
        obs = self.connector.make(game_id)
        self.explorer.reset()
        self.hypothesis_mgr.reset()

        prev_grid = self.connector.get_grid_tensor(obs)

        if self.config.verbose:
            print(f"\n--- Playing {game_id} ---")
            print(f"  Initial state: {obs.state}, levels: {obs.levels_completed}")
            print(f"  Available actions: {obs.available_actions}")

        # Main exploration loop
        for step in range(self.config.max_steps):
            if obs.state in ("WIN", "GAME_OVER"):
                break

            # Check if ACTION6 (click) is available
            use_click = 6 in obs.available_actions

            # Get hypothesis-driven action biases
            h_bonuses = self.hypothesis_mgr.suggest_action_bias(obs.available_actions)

            # Select action via info-max exploration
            plan = self.explorer.select_action(
                prev_grid,
                obs.available_actions,
                use_click=use_click,
                grid_shape=obs.grid.shape,
            )

            # Execute action
            obs = self.connector.step(plan.action, x=plan.x, y=plan.y)
            next_grid = self.connector.get_grid_tensor(obs)

            # Update world model beliefs
            self.hypothesis_mgr.update(prev_grid, plan.action, next_grid)

            # Record stats
            self.stats.total_steps += 1
            self.stats.actions_taken.append(plan.action)
            top_hyps = self.hypothesis_mgr.get_top_hypotheses(2)
            self.stats.hypothesis_history.append([h.name for h in top_hyps])

            if self.config.verbose and step % 20 == 0:
                top = self.hypothesis_mgr.get_top_hypotheses(1)[0]
                print(f"  Step {step}: state={obs.state}, "
                      f"action={plan.action}, info={plan.info_gain:.2f}, "
                      f"top_hyp={top.name}({top.probability:.2f})")

            # Win condition
            if obs.state == "WIN" and self.config.early_stop_on_win:
                if self.config.verbose:
                    print(f"  WIN at step {step}! Levels: {obs.levels_completed}")
                break

            prev_grid = next_grid

        # Finalize
        self.stats.final_state = obs.state
        self.stats.won = obs.state == "WIN"
        self.stats.levels_completed = obs.levels_completed
        self.stats.episode_time = time.time() - start_time

        if self.config.verbose:
            print(f"  Final: {self.stats.final_state}, "
                  f"steps={self.stats.total_steps}, "
                  f"time={self.stats.episode_time:.1f}s")

        return self.connector.get_episode()

    def play_all(self, max_games: int | None = None) -> list[AgentStats]:
        """Play all available environments.

        Returns:
            List of AgentStats for each game played.
        """
        games = self.connector.available_games
        if max_games:
            games = games[:max_games]

        all_stats = []
        total_won = 0

        for i, game in enumerate(games):
            gid = game["game_id"]
            if self.config.verbose:
                print(f"\n=== Game {i+1}/{len(games)}: {gid} ===")
                print(f"  Tags: {game['tags']}")
                print(f"  Baselines: {game['baseline_actions']}")

            self.play(gid)
            all_stats.append(self.stats)

            if self.stats.won:
                total_won += 1

            if self.config.verbose:
                print(f"  Result: {'WON' if self.stats.won else 'LOST'} "
                      f"({total_won}/{i+1} won)")

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Total: {total_won}/{len(games)} won "
                  f"({100*total_won/len(games):.1f}%)")

        return all_stats
