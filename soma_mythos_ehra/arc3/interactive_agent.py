"""Interactive Agent v2 — full active-inference loop with online learning.

Integrates:
1. Experience Replay Buffer — collects transitions during exploration
2. World Model Trainer — trains encoder/transition/reward after each episode
3. LLM Code Evolver — generates/mutates Python hypotheses about rules
4. Curriculum Manager — multi-level progression with knowledge transfer
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
from soma_mythos_ehra.arc3.replay_buffer import ExperienceReplayBuffer
from soma_mythos_ehra.arc3.world_model_trainer import WorldModelTrainer, TrainConfig, TrainMetrics
from soma_mythos_ehra.arc3.code_evolver import LLMCodeEvolver, EvolutionConfig
from soma_mythos_ehra.arc3.curriculum_manager import CurriculumManager


@dataclass
class AgentConfig:
    """Configuration for the interactive agent."""
    max_steps: int = 500
    max_episodes: int = 20
    exploration_rate: float = 0.3
    temperature: float = 1.0
    ensemble_size: int = 3
    latent_dim: int = 256
    num_actions: int = 8
    buffer_capacity: int = 50000
    train_steps_per_episode: int = 100
    evolve_every_n_episodes: int = 3
    mastery_threshold: int = 3
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
    train_metrics: list[TrainMetrics] = field(default_factory=list)
    code_scores: list[float] = field(default_factory=list)


class InteractiveAgent:
    """Full active-inference agent with online learning.

    The agent:
    1. Explores environments via info-max action selection
    2. Stores all transitions in replay buffer
    3. Trains world model ensemble after each episode
    4. Evolves code hypotheses to infer environment rules
    5. Manages curriculum across game levels
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.connector = ARC3Connector()

        # World model ensemble
        self.ensemble = HypothesisEnsemble(
            num_models=self.config.ensemble_size,
            latent_dim=self.config.latent_dim,
            num_actions=self.config.num_actions,
        )

        # Explorer
        self.explorer = InfoMaxExplorer(
            self.ensemble,
            exploration_rate=self.config.exploration_rate,
            temperature=self.config.temperature,
        )

        # Bayesian belief tracking
        self.hypothesis_mgr = HypothesisManager()

        # Experience replay
        self.buffer = ExperienceReplayBuffer(capacity=self.config.buffer_capacity)

        # Online trainer
        self.trainer = WorldModelTrainer(
            self.ensemble, self.buffer,
            config=TrainConfig(train_steps_per_episode=self.config.train_steps_per_episode),
        )

        # Code evolver
        self.code_evolver = LLMCodeEvolver(EvolutionConfig(
            population_size=10,
            num_generations=5,
        ))

        # Curriculum
        self.curriculum = CurriculumManager(mastery_threshold=self.config.mastery_threshold)

        self.stats = AgentStats()
        self.episode_count = 0

    def play(self, game_id: str) -> EpisodeRecord:
        """Play one environment from start to finish with full learning loop."""
        self.stats = AgentStats(game_id=game_id)
        start_time = time.time()

        # Get game info
        games = self.connector.available_games
        game_info = next((g for g in games if g["game_id"] == game_id), None)
        baselines = game_info["baseline_actions"] if game_info else []

        # Register with curriculum
        num_levels = max(len(baselines), 5)
        self.curriculum.register_game(game_id, num_levels, baselines)

        # Get next level to practice
        level_idx = self.curriculum.get_next_level(game_id)

        # Reset environment
        obs = self.connector.make(game_id)
        self.explorer.reset()
        self.hypothesis_mgr.reset()

        prev_grid = self.connector.get_grid_tensor(obs)

        if self.config.verbose:
            print(f"\n--- Playing {game_id} (level {level_idx}) ---")
            print(f"  State: {obs.state}, Actions: {obs.available_actions}")
            print(f"  Buffer: {len(self.buffer)} transitions")

        # Collect episode transitions
        episode_transitions = []

        # Main exploration loop
        for step in range(self.config.max_steps):
            if obs.state in ("WIN", "GAME_OVER"):
                break

            use_click = 6 in obs.available_actions

            # Try code-evolved policy first
            if self.code_evolver.best_hypothesis and self.code_evolver.best_hypothesis.score > 0.4:
                code_action = self.code_evolver.get_best_action(
                    obs.grid, obs.available_actions
                )
                if code_action in obs.available_actions:
                    plan = ActionPlan(
                        action=code_action, info_gain=0.0,
                        predicted_reward=0.5, uncertainty=0.0,
                    )
                else:
                    plan = self.explorer.select_action(
                        prev_grid, obs.available_actions,
                        use_click=use_click, grid_shape=obs.grid.shape,
                    )
            else:
                plan = self.explorer.select_action(
                    prev_grid, obs.available_actions,
                    use_click=use_click, grid_shape=obs.grid.shape,
                )

            # Execute action
            obs = self.connector.step(plan.action, x=plan.x, y=plan.y)
            next_grid = self.connector.get_grid_tensor(obs)

            # Record transition
            reward = 1.0 if obs.state == "WIN" else 0.0
            done = obs.state in ("WIN", "GAME_OVER")
            transition = {
                "prev_grid": prev_grid,
                "action": plan.action,
                "next_grid": next_grid,
                "reward": reward,
                "done": done,
                "available_actions": obs.available_actions,
                "level": level_idx,
            }
            episode_transitions.append(transition)

            # Add to replay buffer
            self.buffer.add(
                prev_grid=prev_grid,
                action=plan.action,
                next_grid=next_grid,
                reward=reward,
                done=done,
                available_actions=obs.available_actions,
                level=level_idx,
            )

            # Record observation for code evolver
            self.code_evolver.record_observation(
                prev_grid.numpy(), plan.action, next_grid.numpy(), reward,
            )

            # Update beliefs
            self.hypothesis_mgr.update(prev_grid, plan.action, next_grid)

            # Record stats
            self.stats.total_steps += 1
            self.stats.actions_taken.append(plan.action)
            top_hyps = self.hypothesis_mgr.get_top_hypotheses(2)
            self.stats.hypothesis_history.append([h.name for h in top_hyps])

            if self.config.verbose and step % 20 == 0:
                top = self.hypothesis_mgr.get_top_hypotheses(1)[0]
                print(f"  Step {step}: {obs.state}, action={plan.action}, "
                      f"hyp={top.name}({top.probability:.2f}), "
                      f"buffer={len(self.buffer)}")

            if obs.state == "WIN" and self.config.early_stop_on_win:
                if self.config.verbose:
                    print(f"  WIN at step {step}! Levels: {obs.levels_completed}")
                break

            prev_grid = next_grid

        # --- Post-episode learning ---

        # 1. Train world model on collected experience
        if len(self.buffer) >= 128:
            if self.config.verbose:
                print(f"  Training world model ({self.config.train_steps_per_episode} steps)...")
            train_metrics = self.trainer.train_episode(episode_transitions)
            self.stats.train_metrics.append(train_metrics)
            if self.config.verbose:
                print(f"    Loss: {train_metrics.total_loss:.4f} "
                      f"(trans={train_metrics.transition_loss:.4f}, "
                      f"rew={train_metrics.reward_loss:.4f})")

        # 2. Evolve code hypotheses periodically
        self.episode_count += 1
        if self.episode_count % self.config.evolve_every_n_episodes == 0:
            if self.config.verbose:
                print("  Evolving code hypotheses...")
            if not self.code_evolver.population:
                self.code_evolver.initialize_population()
            self.code_evolver.evolve()
            best = self.code_evolver.best_hypothesis
            if best:
                self.stats.code_scores.append(best.score)
                if self.config.verbose:
                    print(f"    Best hypothesis: score={best.score:.2f}, "
                          f"correct={best.correct_predictions}/{best.total_predictions}")

        # 3. Record curriculum progress
        won = obs.state == "WIN"
        self.curriculum.record_attempt(
            game_id, level_idx, won,
            self.stats.total_steps, time.time() - start_time,
        )

        # Finalize
        self.stats.final_state = obs.state
        self.stats.won = won
        self.stats.levels_completed = obs.levels_completed
        self.stats.episode_time = time.time() - start_time

        if self.config.verbose:
            print(f"  Final: {self.stats.final_state}, steps={self.stats.total_steps}, "
                  f"time={self.stats.episode_time:.1f}s")
            print(f"  Buffer: {len(self.buffer)} | {self.curriculum.report()}")

        return self.connector.get_episode()

    def play_game(self, game_id: str, max_episodes: int | None = None) -> list[AgentStats]:
        """Play multiple episodes of a game, learning between each."""
        max_eps = max_episodes or self.config.max_episodes
        all_stats = []

        for ep in range(max_eps):
            if self.config.verbose:
                print(f"\n=== Episode {ep+1}/{max_eps} for {game_id} ===")

            self.play(game_id)
            all_stats.append(self.stats)

            # Check if game is fully mastered
            if game_id in self.curriculum.games:
                game = self.curriculum.games[game_id]
                if game.completion_pct >= 1.0:
                    if self.config.verbose:
                        print(f"\n  GAME MASTERED! All {len(game.levels)} levels complete.")
                    break

        return all_stats

    def play_all(self, max_games: int | None = None, episodes_per_game: int = 5) -> list[AgentStats]:
        """Play all available environments with learning."""
        games = self.connector.available_games
        if max_games:
            games = games[:max_games]

        all_stats = []
        total_won = 0

        for i, game in enumerate(games):
            gid = game["game_id"]
            if self.config.verbose:
                print(f"\n{'='*60}")
                print(f"Game {i+1}/{len(games)}: {gid}")
                print(f"Tags: {game['tags']}, Baselines: {game['baseline_actions']}")

            stats = self.play_game(gid, episodes_per_game)
            all_stats.extend(stats)

            final = stats[-1] if stats else None
            if final and final.won:
                total_won += 1

            if self.config.verbose:
                print(f"  Result: {'WON' if final and final.won else 'LOST'} "
                      f"({total_won}/{i+1} games won)")

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"FINAL: {total_won}/{len(games)} games won "
                  f"({100*total_won/len(games):.1f}%)")
            print(f"Buffer: {len(self.buffer)} transitions")
            print(self.curriculum.report())

        return all_stats

    def save(self, path: str) -> None:
        """Save agent state."""
        self.trainer.save(path)

    def load(self, path: str) -> None:
        """Load agent state."""
        self.trainer.load(path)
