"""Action Efficiency Optimizer — reduces action count to match human baselines.

Strategies:
1. Trajectory replay: remember winning action sequences, replay them
2. Heuristic shortcuts: detect patterns and skip exploration
3. Baseline tracking: measure efficiency vs human per level
4. Action pruning: eliminate actions that never lead to state changes
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TrajectoryRecord:
    """A recorded winning trajectory."""
    actions: list[int]
    grid_hashes: list[int]
    game_id: str
    level: int
    total_actions: int
    human_baseline: int
    efficiency: float = 0.0

    def __post_init__(self):
        if self.human_baseline > 0:
            self.efficiency = (self.human_baseline / max(self.total_actions, 1)) ** 2


@dataclass
class EfficiencyStats:
    """Per-game efficiency tracking."""
    game_id: str
    level_scores: list[float] = field(default_factory=list)
    human_baselines: list[int] = field(default_factory=list)
    agent_actions: list[int] = field(default_factory=list)
    trajectories_replayed: int = 0

    @property
    def avg_efficiency(self) -> float:
        if not self.level_scores:
            return 0.0
        return sum(self.level_scores) / len(self.level_scores)

    @property
    def overall_rhae(self) -> float:
        """Overall RHAE across all levels."""
        if not self.human_baselines or not self.agent_actions:
            return 0.0
        total_score = 0.0
        for h, a in zip(self.human_baselines, self.agent_actions):
            if a > 0:
                total_score += (h / a) ** 2
        return total_score / len(self.human_baselines)


class ActionEfficiencyOptimizer:
    """Optimizes action efficiency through trajectory replay and pruning."""

    def __init__(self) -> None:
        self.trajectories: dict[str, list[TrajectoryRecord]] = {}
        self.stats: dict[str, EfficiencyStats] = {}
        self.pruned_actions: dict[str, set[int]] = {}
        self.grid_history: list[int] = []

    def record_transition(
        self,
        game_id: str,
        grid: torch.Tensor,
        action: int,
        level: int,
    ) -> None:
        """Record a transition for pattern detection."""
        grid_hash = hash(grid.numpy().tobytes())
        self.grid_history.append(grid_hash)
        if len(self.grid_history) > 1000:
            self.grid_history = self.grid_history[-1000:]

        # Detect stuck loops (same grid hash repeated)
        if len(self.grid_history) >= 3:
            if self.grid_history[-1] == self.grid_history[-2] == self.grid_history[-3]:
                # Mark this action as ineffective at this state
                key = f"{game_id}_{level}"
                if key not in self.pruned_actions:
                    self.pruned_actions[key] = set()
                self.pruned_actions[key].add(action)

    def record_win(
        self,
        game_id: str,
        actions: list[int],
        grid_hashes: list[int],
        level: int,
        human_baseline: int,
    ) -> None:
        """Record a winning trajectory."""
        record = TrajectoryRecord(
            actions=list(actions),
            grid_hashes=list(grid_hashes),
            game_id=game_id,
            level=level,
            total_actions=len(actions),
            human_baseline=human_baseline,
        )

        if game_id not in self.trajectories:
            self.trajectories[game_id] = []
        self.trajectories[game_id].append(record)

        # Update stats
        if game_id not in self.stats:
            self.stats[game_id] = EfficiencyStats(game_id=game_id)
        stats = self.stats[game_id]
        while len(stats.level_scores) <= level:
            stats.level_scores.append(0.0)
            stats.human_baselines.append(0)
            stats.agent_actions.append(0)

        stats.level_scores[level] = record.efficiency
        stats.human_baselines[level] = human_baseline
        stats.agent_actions[level] = len(actions)

    def get_replay_action(
        self,
        game_id: str,
        step: int,
        level: int,
        current_grid: torch.Tensor,
    ) -> int | None:
        """Try to replay a known winning trajectory.

        Returns the next action if a matching trajectory exists,
        or None if no replay is possible.
        """
        if game_id not in self.trajectories:
            return None

        # Find trajectories for this level
        matching = [t for t in self.trajectories[game_id] if t.level == level]
        if not matching:
            return None

        # Find trajectory where current grid matches at this step
        current_hash = hash(current_grid.numpy().tobytes())
        for traj in matching:
            if step < len(traj.grid_hashes):
                if traj.grid_hashes[step] == current_hash:
                    # Replay: return next action
                    if step + 1 < len(traj.actions):
                        return traj.actions[step + 1]

        return None

    def get_pruned_actions(self, game_id: str, level: int) -> set[int]:
        """Get actions that are known to be ineffective."""
        key = f"{game_id}_{level}"
        return self.pruned_actions.get(key, set())

    def should_replay(self, game_id: str, level: int) -> bool:
        """Check if we should try trajectory replay."""
        if game_id not in self.trajectories:
            return False
        matching = [t for t in self.trajectories[game_id] if t.level == level]
        return len(matching) >= 2  # Need at least 2 wins to trust replay

    def get_efficiency_report(self) -> str:
        """Generate efficiency report."""
        lines = ["=== Action Efficiency ==="]
        for gid, stats in self.stats.items():
            lines.append(
                f"  {gid}: RHAE={stats.overall_rhae:.2f}, "
                f"levels={len(stats.level_scores)}, "
                f"replays={stats.trajectories_replayed}"
            )
        if not self.stats:
            lines.append("  (no data)")
        return "\n".join(lines)

    def suggest_action(
        self,
        game_id: str,
        level: int,
        step: int,
        current_grid: torch.Tensor,
        available_actions: list[int],
    ) -> int | None:
        """Suggest the most efficient action.

        Priority:
        1. Trajectory replay (if we've won before)
        2. Pruned action avoidance
        3. None (let explorer decide)
        """
        # Try trajectory replay
        if self.should_replay(game_id, level):
            replay_action = self.get_replay_action(game_id, step, level, current_grid)
            if replay_action is not None and replay_action in available_actions:
                return replay_action

        return None  # Let explorer decide
