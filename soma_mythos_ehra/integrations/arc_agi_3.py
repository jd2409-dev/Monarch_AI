from __future__ import annotations

import random
from typing import Any

import torch

from soma_mythos_ehra import MonarchAI, MonarchConfig
from soma_mythos_ehra.mythos.search import extract_agent_pos

try:
    from agents.agent import Agent
    from arcengine import FrameData, GameAction, GameState
except ImportError as exc:  # pragma: no cover - optional integration guard
    raise ImportError("ARC-AGI-3 integration requires the official agents harness and arcengine") from exc

# ARC uses different cell values than our simulator.
# ARC: 0=empty, 1=agent, 3=wall, 4+=objects
# Simulator: 0=empty, 1=wall, 2=agent, 3=goal
ARC_TO_SIM_MAP = {
    0: 0,  # empty
    1: 2,  # agent
    2: 0,  # (not used in ARC)
    3: 1,  # wall
}
ARC_AGENT_VALUE = 1
ARC_WALL_VALUES = {3, 4, 5, 8, 9, 11, 12}


def remap_arc_grid(grid: torch.Tensor) -> torch.Tensor:
    """Remap ARC cell values to simulator values."""
    out = torch.zeros_like(grid)
    # Agent
    out[grid == ARC_AGENT_VALUE] = 2
    # Walls
    for v in ARC_WALL_VALUES:
        out[grid == v] = 1
    return out


class Monarch_AI(Agent):
    """ARC-AGI-3 adapter class registered as Monarch_AI.

    Drop this class into the official ARC-AGI-3 Agents registry or import it
    from a thin template module. It intentionally avoids language-model calls:
    the action is selected by the SOMA tensor simulator, Mythos lookahead, and
    EHRA action filter.
    """

    MAX_ACTIONS = 100

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._monarch = MonarchAI(
            MonarchConfig(
                agent_name="Monarch_AI",
                max_actions=1,
                horizon=10,
                simulations=48,
                num_symbols=17,
                latent_dim=64,
                model_path="checkpoints/best_jepa.pt",
            )
        )
        self._global_positions: list[tuple[int, int]] = []
        self._action_history: list[int] = []
        self._rng = random.Random(2409)

    @property
    def name(self) -> str:
        game_id = getattr(self, "game_id", "arc")
        return f"{game_id}.Monarch_AI.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _detect_oscillation(self) -> bool:
        if len(self._action_history) < 6:
            return False
        recent = self._action_history[-6:]
        unique = set(recent)
        return len(unique) == 2 and len(recent) >= 6

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self._global_positions.clear()
            self._action_history.clear()
            action = GameAction.RESET
            action.reasoning = {"agent": "Monarch_AI", "reason": "start_or_restart"}
            return action

        grid = torch.tensor(latest_frame.frame, dtype=torch.long)
        # Remap ARC values to simulator values
        grid = remap_arc_grid(grid)
        available = []
        for raw_action in latest_frame.available_actions:
            action_id = int(raw_action.value) if hasattr(raw_action, "value") else int(raw_action)
            if action_id != int(GameAction.RESET.value):
                available.append(action_id)
        if not available:
            available = [int(a.value) for a in GameAction if a is not GameAction.RESET]

        if self._detect_oscillation() and len(available) > 2:
            last_two = set(self._action_history[-2:])
            unexplored = [a for a in available if a not in last_two]
            if unexplored:
                selected_id = self._rng.choice(unexplored)
                agent_pos = extract_agent_pos(grid)
                self._global_positions.append(agent_pos)
                self._action_history.append(selected_id)
                action = GameAction.from_id(int(selected_id))
                if action.is_complex():
                    action.set_data({"x": 32, "y": 32, "game_id": getattr(self, "game_id", "")})
                action.reasoning = {
                    "agent": "Monarch_AI",
                    "architecture": "SOMA-Mythos-EHRA",
                    "strategy": "oscillation_breakout",
                }
                return action

        runtime = self._monarch.build_runtime(grid)

        if self._global_positions:
            meta = runtime.search.meta
            for pos in self._global_positions[-meta.tabu_window:]:
                meta.recent_positions.append(pos)
                meta.position_visit_counts[pos] = meta.position_visit_counts.get(pos, 0) + 1
            meta.step_count = len(self._global_positions)

        from soma_mythos_ehra.types import GridState
        result = runtime.run(GridState(grid.detach().cpu()), available_actions=tuple(available))

        selected_id = result.actions[0] if result.actions else available[0]

        agent_pos = extract_agent_pos(result.final_state.grid)
        self._global_positions.append(agent_pos)
        self._action_history.append(selected_id)

        action = GameAction.from_id(int(selected_id))
        if action.is_complex():
            action.set_data({"x": 32, "y": 32, "game_id": getattr(self, "game_id", "")})
        action.reasoning = {
            "agent": "Monarch_AI",
            "architecture": "SOMA-Mythos-EHRA",
            "telemetry": str(result.telemetry_path),
        }
        return action
