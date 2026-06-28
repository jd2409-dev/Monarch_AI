"""Interactive Agent v4 — improved world model + code evolution + exploration.

v4 improvements:
- Stop-gradient on world model targets (stable training)
- Grid diff encoder for learning action effects
- Heuristic-based code evolution (27 executable hypotheses)
- Lower thresholds for faster learning
- Systematic exploration strategy
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
from soma_mythos_ehra.arc3.efficiency_optimizer import ActionEfficiencyOptimizer


@dataclass
class AgentConfig:
    max_steps: int = 500
    max_episodes: int = 20
    exploration_rate: float = 0.3
    temperature: float = 1.0
    ensemble_size: int = 5
    latent_dim: int = 256
    num_actions: int = 8
    buffer_capacity: int = 100000
    train_steps_per_episode: int = 200
    evolve_every_n_episodes: int = 2
    mastery_threshold: int = 3
    early_stop_on_win: bool = True
    use_llm: bool = True
    llm_model: str = "gpt-4o-mini"
    verbose: bool = True


@dataclass
class AgentStats:
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
    replay_count: int = 0
    efficiency: float = 0.0


class InteractiveAgent:
    """Full active-inference agent with LLM code evolution and efficiency optimization."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.connector = ARC3Connector()

        # World model ensemble (scaled: 5 models)
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
        self.buffer = ExperienceReplayBuffer(capacity=self.config.buffer_capacity)

        self.trainer = WorldModelTrainer(
            self.ensemble, self.buffer,
            config=TrainConfig(
                train_steps_per_episode=self.config.train_steps_per_episode,
                batch_size=64,
            ),
        )

        self.code_evolver = LLMCodeEvolver(EvolutionConfig(
            population_size=12,
            num_generations=8,
            use_llm=self.config.use_llm,
            llm_model=self.config.llm_model,
        ))

        self.curriculum = CurriculumManager(mastery_threshold=self.config.mastery_threshold)
        self.efficiency = ActionEfficiencyOptimizer()

        self.stats = AgentStats()
        self.episode_count = 0

    def play(self, game_id: str) -> EpisodeRecord:
        """Play one environment with full learning + efficiency loop."""
        self.stats = AgentStats(game_id=game_id)
        start_time = time.time()

        # Get game info
        games = self.connector.available_games
        game_info = next((g for g in games if g["game_id"] == game_id), None)
        baselines = game_info["baseline_actions"] if game_info else []

        num_levels = max(len(baselines), 5)
        self.curriculum.register_game(game_id, num_levels, baselines)
        level_idx = self.curriculum.get_next_level(game_id)
        human_baseline = baselines[level_idx] if level_idx < len(baselines) else 50

        # Reset
        obs = self.connector.make(game_id)
        self.explorer.reset()
        self.hypothesis_mgr.reset()

        prev_grid = self.connector.get_grid_tensor(obs)

        if self.config.verbose:
            print(f"\n--- Playing {game_id} (level {level_idx}) ---")
            print(f"  Baseline: {human_baseline} actions | Buffer: {len(self.buffer)}")

        episode_transitions = []
        grid_hashes = []
        replay_used = False

        for step in range(self.config.max_steps):
            if obs.state in ("WIN", "GAME_OVER"):
                break

            use_click = 6 in obs.available_actions

            # --- Action selection priority ---
            action_plan = None

            # 1. Trajectory replay (if we've won before)
            if self.efficiency.should_replay(game_id, level_idx):
                replay_action = self.efficiency.get_replay_action(
                    game_id, step, level_idx, prev_grid,
                )
                if replay_action is not None and replay_action in obs.available_actions:
                    action_plan = ActionPlan(
                        action=replay_action, info_gain=0.0,
                        predicted_reward=1.0, uncertainty=0.0,
                    )
                    replay_used = True
                    self.stats.replay_count += 1

            # 2. Code-evolved policy
            if action_plan is None and self.code_evolver.best_hypothesis:
                if self.code_evolver.best_hypothesis.score > 0.1:
                    code_action = self.code_evolver.get_best_action(
                        obs.grid, obs.available_actions,
                    )
                    if code_action in obs.available_actions:
                        action_plan = ActionPlan(
                            action=code_action, info_gain=0.0,
                            predicted_reward=0.5, uncertainty=0.0,
                        )

            # 3. Info-max exploration (default)
            if action_plan is None:
                action_plan = self.explorer.select_action(
                    prev_grid, obs.available_actions,
                    use_click=use_click, grid_shape=obs.grid.shape,
                )

            # Execute
            obs = self.connector.step(action_plan.action, x=action_plan.x, y=action_plan.y)
            next_grid = self.connector.get_grid_tensor(obs)

            # Record
            reward = 1.0 if obs.state == "WIN" else 0.0
            done = obs.state in ("WIN", "GAME_OVER")
            grid_hashes.append(hash(prev_grid.numpy().tobytes()))

            transition = {
                "prev_grid": prev_grid, "action": action_plan.action,
                "next_grid": next_grid, "reward": reward, "done": done,
                "available_actions": obs.available_actions, "level": level_idx,
            }
            episode_transitions.append(transition)

            self.buffer.add(
                prev_grid=prev_grid, action=action_plan.action,
                next_grid=next_grid, reward=reward, done=done,
                available_actions=obs.available_actions, level=level_idx,
            )

            self.code_evolver.record_observation(
                prev_grid.numpy(), action_plan.action, next_grid.numpy(), reward,
            )
            self.efficiency.record_transition(
                game_id, prev_grid, action_plan.action, level_idx,
            )
            self.hypothesis_mgr.update(prev_grid, action_plan.action, next_grid)

            self.stats.total_steps += 1
            self.stats.actions_taken.append(action_plan.action)

            if self.config.verbose and step % 25 == 0:
                top = self.hypothesis_mgr.get_top_hypotheses(1)[0]
                print(f"  Step {step}: {obs.state}, act={action_plan.action}, "
                      f"hyp={top.name}({top.probability:.2f}), buf={len(self.buffer)}")

            if obs.state == "WIN" and self.config.early_stop_on_win:
                if self.config.verbose:
                    print(f"  WIN at step {step}!")
                break

            prev_grid = next_grid

        # --- Post-episode learning ---

        # 1. Record win trajectory
        won = obs.state == "WIN"
        if won:
            self.efficiency.record_win(
                game_id, self.stats.actions_taken, grid_hashes,
                level_idx, human_baseline,
            )
            # Compute efficiency
            if human_baseline > 0:
                self.stats.efficiency = (human_baseline / max(self.stats.total_steps, 1)) ** 2

        # 2. Train world model
        if len(self.buffer) >= 64:
            if self.config.verbose:
                print(f"  Training ({self.config.train_steps_per_episode} steps)...")
            train_metrics = self.trainer.train_episode(episode_transitions)
            self.stats.train_metrics.append(train_metrics)
            if self.config.verbose:
                print(f"    loss={train_metrics.total_loss:.4f} "
                      f"(trans={train_metrics.transition_loss:.4f}, "
                      f"rew={train_metrics.reward_loss:.4f})")

        # 3. Evolve code
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
                    print(f"    Best: score={best.score:.2f}, "
                          f"correct={best.correct_predictions}/{best.total_predictions}, "
                          f"source={best.source}")

        # 4. Record curriculum
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
                  f"time={self.stats.episode_time:.1f}s, "
                  f"replays={self.stats.replay_count}")

        return self.connector.get_episode()

    def play_game(self, game_id: str, max_episodes: int | None = None) -> list[AgentStats]:
        """Play multiple episodes, learning between each."""
        max_eps = max_episodes or self.config.max_episodes
        all_stats = []

        for ep in range(max_eps):
            if self.config.verbose:
                print(f"\n=== Episode {ep+1}/{max_eps} for {game_id} ===")

            self.play(game_id)
            all_stats.append(self.stats)

            if game_id in self.curriculum.games:
                game = self.curriculum.games[game_id]
                if game.completion_pct >= 1.0:
                    if self.config.verbose:
                        print(f"\n  GAME MASTERED!")
                    break

        return all_stats

    def play_all(self, max_games: int | None = None, episodes_per_game: int = 5) -> list[AgentStats]:
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
            print(f"\n{'='*60}")
            print(f"FINAL: {total_won}/{len(games)} games won")
            print(f"Buffer: {len(self.buffer)} | {self.efficiency.get_efficiency_report()}")
            print(self.curriculum.report())

        return all_stats

    def save(self, path: str) -> None:
        self.trainer.save(path)

    def load(self, path: str) -> None:
        self.trainer.load(path)
