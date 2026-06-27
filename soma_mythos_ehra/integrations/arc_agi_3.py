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
# ARC level ls20: 0=empty, 1=agent, 3=floor, 4=wall, 5=border,
#   8=box(object), 9=target(goal), 11=corridor, 12=platform
# Simulator: 0=empty, 1=wall, 2=agent, 3=goal
ARC_AGENT_VALUE = 1
ARC_FLOOR_VALUES = {0, 3, 11}  # walkable cells
ARC_WALL_VALUES = {4, 5}  # impassable cells
ARC_GOAL_VALUES = {9}  # target positions (goal)
ARC_BOX_VALUES = {8}  # pushable objects (block movement)
ARC_PLATFORM_VALUES = {12}  # moving platforms (walkable)

# Multi-channel layout: 0=wall, 1=interactive, 2=dynamic(agent), 3=floor
MC_CHANNELS = 4
MC_WALL = 0
MC_INTERACTIVE = 1
MC_DYNAMIC = 2
MC_FLOOR = 3


def remap_arc_grid(grid: torch.Tensor) -> torch.Tensor:
    """Remap ARC cell values to simulator single-channel values."""
    out = torch.zeros_like(grid)
    out[grid == ARC_AGENT_VALUE] = 2
    for v in ARC_WALL_VALUES:
        out[grid == v] = 1
    for v in ARC_BOX_VALUES:
        out[grid == v] = 12  # BOX
    for v in ARC_GOAL_VALUES:
        out[grid == v] = 13  # TARGET
    for v in ARC_PLATFORM_VALUES:
        out[grid == v] = 0
    return out


def remap_arc_grid_multichannel(grid: torch.Tensor) -> torch.Tensor:
    """Remap ARC grid to a 4-channel multi-channel representation.

    Handles both (H, W) and (C, H, W) input shapes from ARC.
    """
    if grid.ndim == 3:
        grid = grid[0]  # Take first channel if 3D
    H, W = grid.shape[-2], grid.shape[-1]
    out = torch.zeros(MC_CHANNELS, H, W, dtype=torch.long)

    # Channel 0: Walls (border, wall — block movement)
    for v in ARC_WALL_VALUES:
        out[MC_WALL] = out[MC_WALL] | (grid == v).long()

    # Channel 1: Interactive (goals/targets, doors, switches, teleporters)
    for v in ARC_GOAL_VALUES:
        out[MC_INTERACTIVE] = out[MC_INTERACTIVE] | (grid == v).long()

    # Channel 2: Dynamic (agent)
    out[MC_DYNAMIC] = (grid == ARC_AGENT_VALUE).long()

    # Channel 3: Floor / walkable (empty, floor, corridor, platform, boxes)
    for v in ARC_FLOOR_VALUES:
        out[MC_FLOOR] = out[MC_FLOOR] | (grid == v).long()
    for v in ARC_PLATFORM_VALUES:
        out[MC_FLOOR] = out[MC_FLOOR] | (grid == v).long()
    for v in ARC_BOX_VALUES:
        out[MC_FLOOR] = out[MC_FLOOR] | (grid == v).long()

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

    # Direction mapping: ACTION1=Up(-1,0), ACTION2=Down(+1,0), ACTION3=Left(0,-1), ACTION4=Right(0,1)
    _DIRECTION_MAP = {
        1: (-1, 0),  # ACTION1 Up
        2: (+1, 0),  # ACTION2 Down
        3: (0, -1),  # ACTION3 Left
        4: (0, +1),  # ACTION4 Right
    }
    # Priority order for heuristic exploration: Right, Down, Left, Up
    _EXPLORE_PRIORITY = [4, 2, 3, 1]

    def _init_heuristic_state(self) -> None:
        if not hasattr(self, "_est_pos"):
            self._est_pos = [32, 20]
        if not hasattr(self, "_visited"):
            self._visited: set[tuple[int, int]] = set()
        if not hasattr(self, "_step_in_level"):
            self._step_in_level = 0

    def _choose_heuristic(
        self, gray_grid: torch.Tensor, available: list[int]
    ) -> int:
        self._init_heuristic_state()
        pos_y, pos_x = self._est_pos
        self._visited.add((pos_y, pos_x))

        H, W = gray_grid.shape[-2], gray_grid.shape[-1]
        for dir_action in self._EXPLORE_PRIORITY:
            if dir_action not in available:
                continue
            dy, dx = self._DIRECTION_MAP[dir_action]
            new_y = pos_y + dy
            new_x = pos_x + dx
            if 0 <= new_y < H and 0 <= new_x < W:
                cell = int(gray_grid[0, new_y, new_x].item())
                # Walkable after remap: 0=empty, 2=agent, 3=goal, 12=box, 13=target
                if cell in (0, 2, 3, 12, 13):
                    if (new_y, new_x) not in self._visited:
                        self._est_pos = [new_y, new_x]
                        self._step_in_level += 1
                        return dir_action

        for dir_action in self._EXPLORE_PRIORITY:
            if dir_action in available:
                dy, dx = self._DIRECTION_MAP[dir_action]
                new_y = pos_y + dy
                new_x = pos_x + dx
                if 0 <= new_y < H and 0 <= new_x < W:
                    cell = int(gray_grid[0, new_y, new_x].item())
                    if cell in (0, 2, 3, 12, 13):
                        self._est_pos = [new_y, new_x]
                        self._step_in_level += 1
                        return dir_action

        self._step_in_level += 1
        return available[0]

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self._global_positions.clear()
            self._action_history.clear()
            if hasattr(self, "_est_pos"):
                del self._est_pos
            if hasattr(self, "_visited"):
                del self._visited
            if hasattr(self, "_step_in_level"):
                del self._step_in_level
            action = GameAction.RESET
            action.reasoning = {"agent": "Monarch_AI", "reason": "start_or_restart"}
            return action

        grid = torch.tensor(latest_frame.frame, dtype=torch.long)
        mc_grid = remap_arc_grid_multichannel(grid)
        gray_grid = remap_arc_grid(grid)

        available = []
        for raw_action in latest_frame.available_actions:
            action_id = int(raw_action.value) if hasattr(raw_action, "value") else int(raw_action)
            if action_id != int(GameAction.RESET.value):
                available.append(action_id)
        if not available:
            available = [int(a.value) for a in GameAction if a is not GameAction.RESET]

        # MODE 1: Heuristic exploration (first 60 steps or after reset)
        self._init_heuristic_state()
        if self._step_in_level < 60:
            # Every 20 steps try USE (ACTION5) to interact
            if self._step_in_level % 20 == 19 and int(GameAction.USE.value) in available:
                selected_id = int(GameAction.USE.value)
            else:
                selected_id = self._choose_heuristic(gray_grid, available)

            agent_pos = torch.tensor([self._est_pos[0], self._est_pos[1]])
            self._global_positions.append(agent_pos)
            self._action_history.append(selected_id)

            action = GameAction.from_id(int(selected_id))
            if action.is_complex():
                action.set_data({"x": self._est_pos[1], "y": self._est_pos[0], "game_id": getattr(self, "game_id", "")})
            action.reasoning = {
                "agent": "Monarch_AI",
                "architecture": "SOMA-Mythos-EHRA",
                "strategy": "heuristic_exploration",
                "est_pos": list(self._est_pos),
                "step": self._step_in_level,
            }
            return action

        # MODE 2: Oscillation breakout
        if self._detect_oscillation() and len(available) > 2:
            last_two = set(self._action_history[-2:])
            unexplored = [a for a in available if a not in last_two]
            if unexplored:
                selected_id = self._rng.choice(unexplored)
                agent_pos = extract_agent_pos(gray_grid)
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

        # MODE 3: MCTS with runtime
        runtime = self._monarch.build_runtime(gray_grid)

        if self._global_positions:
            meta = runtime.search.meta
            for pos in self._global_positions[-meta.tabu_window:]:
                meta.recent_positions.append(pos)
                meta.position_visit_counts[pos] = meta.position_visit_counts.get(pos, 0) + 1
            meta.step_count = len(self._global_positions)

        from soma_mythos_ehra.types import GridState
        result = runtime.run(GridState(gray_grid.detach().cpu()), available_actions=tuple(available))

        selected_id = result.actions[0] if result.actions else available[0]

        agent_pos = extract_agent_pos(result.final_state.grid)
        self._global_positions.append(agent_pos)
        self._action_history.append(selected_id)
        self._step_in_level += 1

        action = GameAction.from_id(int(selected_id))
        if action.is_complex():
            action.set_data({"x": 32, "y": 32, "game_id": getattr(self, "game_id", "")})
        action.reasoning = {
            "agent": "Monarch_AI",
            "architecture": "SOMA-Mythos-EHRA",
            "strategy": "mcts",
            "telemetry": str(result.telemetry_path),
        }
        return action
