"""ARC-AGI-3 Environment Connector — wraps the official SDK for interactive play.

Provides a clean interface for the agent to interact with ARC-AGI-3 environments:
- Reset, step, observe grid state
- Track available actions per turn
- Manage scorecards for evaluation
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

try:
    import arc_agi
    from arcengine import GameAction, GameState
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


@dataclass
class FrameObservation:
    """Processed observation from one environment step."""
    grid: np.ndarray
    state: str  # "NOT_FINISHED", "WIN", "GAME_OVER"
    available_actions: list[int]
    levels_completed: int
    game_id: str
    guid: str
    raw: Any = None


@dataclass
class EpisodeRecord:
    """Records a full episode for analysis."""
    game_id: str
    frames: list[FrameObservation] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    final_state: str = "NOT_FINISHED"
    total_actions: int = 0
    levels_completed: int = 0


class ARC3Connector:
    """Connector to ARC-AGI-3 interactive environments."""

    def __init__(self, api_key: str | None = None) -> None:
        if not HAS_SDK:
            raise ImportError("arc-agi SDK not installed. Run: pip install arc-agi")
        self.arc = arc_agi.Arcade()
        self.env = None
        self.current_game_id: str | None = None
        self.episode = EpisodeRecord(game_id="")

    @property
    def available_games(self) -> list[dict]:
        """List available game environments."""
        envs = self.arc.available_environments
        return [
            {
                "game_id": e.game_id,
                "title": e.title,
                "tags": e.tags,
                "baseline_actions": e.baseline_actions,
            }
            for e in envs
        ]

    def make(self, game_id: str) -> FrameObservation:
        """Create and reset an environment."""
        self.env = self.arc.make(game_id)
        self.current_game_id = game_id
        self.episode = EpisodeRecord(game_id=game_id)
        raw = self.env.reset()
        return self._process_obs(raw)

    def step(self, action: int, x: int | None = None, y: int | None = None) -> FrameObservation:
        """Take an action in the environment.

        Args:
            action: Action number (1-7). Action 6 requires x, y coordinates.
            x: X coordinate for ACTION6 (0-63 range).
            y: Y coordinate for ACTION6 (0-63 range).
        Returns:
            Updated FrameObservation.
        """
        if self.env is None:
            raise RuntimeError("No environment loaded. Call make() first.")

        if action == 6 and x is not None and y is not None:
            raw = self.env.step(GameAction.ACTION6, data={"x": x, "y": y})
        else:
            action_name = f"ACTION{action}"
            action_enum = GameAction[action_name]
            raw = self.env.step(action_enum)

        obs = self._process_obs(raw)

        # Record episode
        self.episode.actions.append(action)
        self.episode.frames.append(obs)
        reward = 1.0 if obs.state == "WIN" else 0.0
        self.episode.rewards.append(reward)
        self.episode.total_actions += 1

        if obs.state == "WIN":
            self.episode.levels_completed = obs.levels_completed

        if obs.state in ("WIN", "GAME_OVER"):
            self.episode.final_state = obs.state

        return obs

    def get_grid_tensor(self, obs: FrameObservation) -> torch.Tensor:
        """Convert grid observation to a PyTorch tensor."""
        return torch.tensor(obs.grid, dtype=torch.long)

    def get_human_baseline(self, game_id: str) -> list[int] | None:
        """Get human baseline actions per level for a game."""
        for e in self.arc.available_environments:
            if e.game_id == game_id:
                return e.baseline_actions
        return None

    def get_episode(self) -> EpisodeRecord:
        """Get the current episode record."""
        return self.episode

    def close(self) -> None:
        """Close the environment."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

    def _process_obs(self, raw) -> FrameObservation:
        """Process raw SDK observation into clean format."""
        if raw is None:
            return FrameObservation(
                grid=np.zeros((64, 64), dtype=np.int8),
                state="GAME_OVER",
                available_actions=[],
                levels_completed=0,
                game_id=self.current_game_id or "",
                guid="",
            )

        grid = np.array(raw.frame[0]) if raw.frame else np.zeros((64, 64), dtype=np.int8)
        state = raw.state.name if hasattr(raw.state, "name") else str(raw.state)
        return FrameObservation(
            grid=grid,
            state=state,
            available_actions=raw.available_actions or [],
            levels_completed=raw.levels_completed,
            game_id=raw.game_id,
            guid=raw.guid,
            raw=raw,
        )
