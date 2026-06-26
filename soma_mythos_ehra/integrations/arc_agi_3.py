from __future__ import annotations

from typing import Any

import torch

from soma_mythos_ehra import MonarchAI, MonarchConfig

try:
    from agents.agent import Agent
    from arcengine import FrameData, GameAction, GameState
except ImportError as exc:  # pragma: no cover - optional integration guard
    raise ImportError("ARC-AGI-3 integration requires the official agents harness and arcengine") from exc


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
                latent_dim=32,
            )
        )

    @property
    def name(self) -> str:
        game_id = getattr(self, "game_id", "arc")
        return f"{game_id}.Monarch_AI.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
            action.reasoning = {"agent": "Monarch_AI", "reason": "start_or_restart"}
            return action

        grid = torch.tensor(latest_frame.frame, dtype=torch.long)
        available = []
        for raw_action in latest_frame.available_actions:
            action_id = int(raw_action.value) if hasattr(raw_action, "value") else int(raw_action)
            if action_id != int(GameAction.RESET.value):
                available.append(action_id)
        if not available:
            available = [int(a.value) for a in GameAction if a is not GameAction.RESET]

        result = self._monarch.solve(grid, available_actions=tuple(available))
        selected_id = result.actions[0] if result.actions else available[0]
        action = GameAction.from_id(int(selected_id))
        if action.is_complex():
            action.set_data({"x": 32, "y": 32, "game_id": getattr(self, "game_id", "")})
        action.reasoning = {
            "agent": "Monarch_AI",
            "architecture": "SOMA-Mythos-EHRA",
            "telemetry": str(result.telemetry_path),
        }
        return action
