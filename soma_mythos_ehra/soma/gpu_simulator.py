from __future__ import annotations

from dataclasses import dataclass

import torch

from soma_mythos_ehra.types import DIRECTION_DELTAS, Action


@dataclass(frozen=True)
class SimulatorConfig:
    wall_value: int = 1
    agent_value: int = 2
    goal_value: int = 3
    empty_value: int = 0


class TensorGridSimulator:
    """Vectorized grid transition engine.

    The simulator treats a batch of 2D integer grids as discrete physics states.
    Directional moves are evaluated as tensor operations on CPU or CUDA. Walls
    are immutable, collisions are rejected, and goal contact marks low-energy
    terminal states.
    """

    def __init__(
        self,
        initial_grid: torch.Tensor,
        config: SimulatorConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or SimulatorConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        grid = self._coerce_grid(initial_grid).to(self.device)
        self.shape = tuple(grid.shape[-2:])
        self.wall_mask = grid == self.config.wall_value
        self.goal_mask = grid == self.config.goal_value

    def _coerce_grid(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.ndim == 2:
            return grid.to(dtype=torch.long).unsqueeze(0)
        if grid.ndim == 3:
            if grid.shape[0] in (3, 4, 16) and grid.shape[1] == grid.shape[2]:
                grid = grid.float().argmax(dim=0)
                return grid.to(dtype=torch.long).unsqueeze(0)
            if grid.shape[-1] in (3, 4, 16) and grid.shape[0] == grid.shape[1]:
                grid = grid.float().argmax(dim=-1)
                return grid.to(dtype=torch.long).unsqueeze(0)
            return grid.to(dtype=torch.long)
        raise ValueError("grid must have shape (H, W) or (B, H, W)")

    def to_device(self, grid: torch.Tensor) -> torch.Tensor:
        return self._coerce_grid(grid).to(self.device)

    def step_batch(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        states = self.to_device(states)
        actions = actions.to(self.device, dtype=torch.long).flatten()
        if states.shape[0] == 1 and actions.numel() > 1:
            states = states.repeat(actions.numel(), 1, 1)
        if states.shape[0] != actions.numel():
            raise ValueError("states batch and actions batch must align")

        next_states = states.clone()
        energy = torch.ones(actions.shape[0], device=self.device, dtype=torch.float32)

        agent_pos = self._agent_positions(states)
        h, w = states.shape[-2:]
        for action, (dy, dx) in DIRECTION_DELTAS.items():
            mask = actions == int(action)
            if not mask.any():
                continue
            pos = agent_pos[mask]
            y = torch.clamp(pos[:, 0] + dy, 0, h - 1)
            x = torch.clamp(pos[:, 1] + dx, 0, w - 1)
            blocked = self.wall_mask[0, y, x] | ((y == pos[:, 0]) & (x == pos[:, 1]))
            batch_idx = torch.nonzero(mask, as_tuple=False).flatten()
            valid_idx = batch_idx[~blocked]
            if valid_idx.numel():
                old = agent_pos[valid_idx]
                new_y = y[~blocked]
                new_x = x[~blocked]
                next_states[valid_idx, old[:, 0], old[:, 1]] = self.config.empty_value
                reached_goal = self.goal_mask[0, new_y, new_x]
                next_states[valid_idx, new_y, new_x] = self.config.agent_value
                energy[valid_idx] = torch.where(reached_goal, torch.zeros_like(energy[valid_idx]), 0.25)
            energy[batch_idx[blocked]] = 5.0

        passive = ~torch.isin(actions, torch.tensor([int(a) for a in DIRECTION_DELTAS], device=self.device))
        if passive.any():
            energy[passive] = 1.5
        return next_states, energy

    def rollout(self, state: torch.Tensor, sequence: tuple[int, ...]) -> tuple[torch.Tensor, float]:
        current = self.to_device(state)
        total = 0.0
        for action in sequence:
            current, energy = self.step_batch(current, torch.tensor([action], device=self.device))
            total += float(energy[0].item())
        return current, total

    def _agent_positions(self, states: torch.Tensor) -> torch.Tensor:
        positions = (states == self.config.agent_value).flatten(1).float().argmax(dim=1)
        width = states.shape[-1]
        return torch.stack((positions // width, positions % width), dim=1).long()

    def distance_to_goal_energy(self, states: torch.Tensor) -> torch.Tensor:
        states = self.to_device(states)
        agent = self._agent_positions(states).float()
        goal = torch.nonzero(self.goal_mask[0], as_tuple=False)
        if goal.numel() == 0:
            return torch.ones(states.shape[0], device=self.device)
        dist = torch.cdist(agent, goal.float(), p=1).min(dim=1).values
        return dist / max(states.shape[-2] + states.shape[-1], 1)
