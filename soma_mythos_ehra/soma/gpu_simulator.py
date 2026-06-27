from __future__ import annotations

from dataclasses import dataclass

import torch

from soma_mythos_ehra.types import DIRECTION_DELTAS, Action, CellType


@dataclass(frozen=True)
class SimulatorConfig:
    wall_value: int = int(CellType.WALL)
    agent_value: int = int(CellType.AGENT)
    goal_value: int = int(CellType.GOAL)
    empty_value: int = int(CellType.EMPTY)
    switch_a_value: int = int(CellType.SWITCH_A)
    switch_b_value: int = int(CellType.SWITCH_B)
    door_a_closed: int = int(CellType.DOOR_A_CLOSED)
    door_a_open: int = int(CellType.DOOR_A_OPEN)
    door_b_closed: int = int(CellType.DOOR_B_CLOSED)
    door_b_open: int = int(CellType.DOOR_B_OPEN)
    teleporter_blue: int = int(CellType.TELEPORTER_BLUE)
    teleporter_red: int = int(CellType.TELEPORTER_RED)
    box_value: int = int(CellType.BOX)
    target_value: int = int(CellType.TARGET)
    box_on_target: int = int(CellType.BOX_ON_TARGET)


class TensorGridSimulator:
    """Vectorized grid transition engine with ARC-style interactive objects.

    Extends basic movement physics with:
    - **Switches**: Stepping on a switch toggles all doors of the same pair.
    - **Doors**: Block passage when closed, become passable when opened by
      the matching switch.
    - **Teleporters**: Pairs of tiles that teleport the agent to the other
      end when stepped on.

    All mechanics are fully vectorized across batch dimensions for GPU
    parallelism.
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
        self.static_wall_mask = grid == self.config.wall_value
        self.goal_mask = (grid == self.config.goal_value) | (grid == self.config.target_value)

    def _coerce_grid(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.ndim == 2:
            return grid.to(dtype=torch.long).unsqueeze(0)
        if grid.ndim == 3:
            if grid.shape[0] in (3, 4, 12, 16) and grid.shape[1] == grid.shape[2]:
                grid = grid.float().argmax(dim=0)
                return grid.to(dtype=torch.long).unsqueeze(0)
            if grid.shape[-1] in (3, 4, 12, 16) and grid.shape[0] == grid.shape[1]:
                grid = grid.float().argmax(dim=-1)
                return grid.to(dtype=torch.long).unsqueeze(0)
            return grid.to(dtype=torch.long)
        if grid.ndim == 4:
            # Multi-channel input (B, C, H, W) — convert to grayscale via argmax
            return grid.float().argmax(dim=1).to(dtype=torch.long)
        raise ValueError("grid must have shape (H, W), (B, H, W), or (B, C, H, W)")

    def to_device(self, grid: torch.Tensor) -> torch.Tensor:
        return self._coerce_grid(grid).to(self.device)

    # ------------------------------------------------------------------
    # Door state management
    # ------------------------------------------------------------------

    def _is_door_closed(self, cell_val: int) -> bool:
        """Check if a cell value represents a closed door."""
        return cell_val in (self.config.door_a_closed, self.config.door_b_closed)

    def _is_door_open(self, cell_val: int) -> bool:
        """Check if a cell value represents an open door."""
        return cell_val in (self.config.door_a_open, self.config.door_b_open)

    def _toggle_doors_in_batch(self, grid: torch.Tensor, batch_idx: int, pair: str) -> torch.Tensor:
        """Toggle all doors of a given pair for a single batch element."""
        if pair == "a":
            closed_val, open_val = self.config.door_a_closed, self.config.door_a_open
        else:
            closed_val, open_val = self.config.door_b_closed, self.config.door_b_open
        grid = grid.clone()
        closed_mask = grid[batch_idx] == closed_val
        open_mask = grid[batch_idx] == open_val
        grid[batch_idx, closed_mask] = open_val
        grid[batch_idx, open_mask] = closed_val
        return grid

    def _cell_blocks_movement(self, cell_val: int) -> bool:
        """Check if a cell value blocks agent movement."""
        return (
            cell_val == self.config.wall_value
            or self._is_door_closed(cell_val)
            or cell_val == self.config.box_value
        )

    def _is_box(self, cell_val: int) -> bool:
        return cell_val == self.config.box_value

    def _is_target(self, cell_val: int) -> bool:
        return cell_val == self.config.target_value

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def step_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Handle multi-channel input: extract grayscale for physics,
        # then write results back into multi-channel output.
        input_ndim = states.ndim
        if input_ndim == 4 and states.shape[1] > 1:
            multi_channel = states.to(self.device)
            gray = self._coerce_grid(states).to(self.device)
        else:
            multi_channel = None
            gray = self.to_device(states)
        states = gray
        actions = actions.to(self.device, dtype=torch.long).flatten()
        if states.shape[0] == 1 and actions.numel() > 1:
            states = states.repeat(actions.numel(), 1, 1)
        if states.shape[0] != actions.numel():
            raise ValueError("states batch and actions batch must align")

        next_states = states.clone()
        energy = torch.ones(actions.shape[0], device=self.device, dtype=torch.float32)

        # Record original cell values at destinations BEFORE placing agent
        # This is needed for switch/teleporter detection
        agent_pos = self._agent_positions(states)
        h, w = states.shape[-2:]

        # Build per-sample collision masks (walls + closed doors)
        collision_masks = self._build_per_sample_collision_masks(states)

        # Pre-compute destination positions and original cell values for each action
        dest_y = torch.zeros(states.shape[0], dtype=torch.long, device=self.device)
        dest_x = torch.zeros(states.shape[0], dtype=torch.long, device=self.device)
        original_dest_val = torch.zeros(states.shape[0], dtype=torch.long, device=self.device)

        for action, (dy, dx) in DIRECTION_DELTAS.items():
            mask = actions == int(action)
            if not mask.any():
                continue
            pos = agent_pos[mask]
            y = torch.clamp(pos[:, 0] + dy, 0, h - 1)
            x = torch.clamp(pos[:, 1] + dx, 0, w - 1)
            dest_y[mask] = y
            dest_x[mask] = x
            # Record what was at the destination BEFORE the move
            original_dest_val[mask] = states[mask, y, x]

        # Determine which movements are blocked
        is_direction = torch.isin(actions, torch.tensor([int(a) for a in DIRECTION_DELTAS], device=self.device))
        blocked = torch.zeros(states.shape[0], dtype=torch.bool, device=self.device)

        # Track box-pushing: which samples push a box, and where the box goes
        pushing_box = torch.zeros(states.shape[0], dtype=torch.bool, device=self.device)
        box_dest_y = torch.zeros(states.shape[0], dtype=torch.long, device=self.device)
        box_dest_x = torch.zeros(states.shape[0], dtype=torch.long, device=self.device)

        if is_direction.any():
            b_idx = torch.nonzero(is_direction, as_tuple=False).flatten()
            blocked[b_idx] = collision_masks[b_idx, dest_y[b_idx], dest_x[b_idx]]

            # Check for box-pushing: if destination is a box, check cell behind
            dest_is_box = (original_dest_val[b_idx] == self.config.box_value)
            if dest_is_box.any():
                push_idx = b_idx[dest_is_box]
                # Get the action direction for each pushing sample
                push_actions = actions[push_idx]
                for push_i, pi in enumerate(push_idx.item() if push_idx.dim() == 0 else []):
                    pass  # Handle scalar case
                # Vectorized: get direction deltas for each push
                push_dy = torch.zeros(push_idx.shape[0], dtype=torch.long, device=self.device)
                push_dx = torch.zeros(push_idx.shape[0], dtype=torch.long, device=self.device)
                for act, (ddy, ddx) in DIRECTION_DELTAS.items():
                    act_mask = actions[push_idx] == int(act)
                    push_dy[act_mask] = ddy
                    push_dx[act_mask] = ddx
                # Cell behind the box = dest + direction
                behind_y = torch.clamp(dest_y[push_idx] + push_dy, 0, h - 1)
                behind_x = torch.clamp(dest_x[push_idx] + push_dx, 0, w - 1)
                # Check if behind cell is pushable (empty, target, or goal)
                behind_val = states[push_idx, behind_y, behind_x]
                pushable = torch.isin(behind_val, torch.tensor([
                    self.config.empty_value, self.config.target_value,
                    self.config.goal_value, self.config.box_on_target,
                ], device=self.device))
                # Also check behind isn't another box
                pushable = pushable & (behind_val != self.config.box_value)
                # Don't push if behind would be same as agent's old position
                pushable = pushable & ~(
                    (behind_y == agent_pos[push_idx, 0]) & (behind_x == agent_pos[push_idx, 1])
                )
                # Mark pushable boxes as not blocking
                can_push = push_idx[pushable] if pushable.dim() > 0 else push_idx
                if can_push.numel() > 0:
                    pushing_box[can_push] = True
                    box_dest_y[can_push] = behind_y[pushable] if pushable.dim() > 0 else behind_y
                    box_dest_x[can_push] = behind_x[pushable] if pushable.dim() > 0 else behind_x
                    blocked[can_push] = False
                # Non-pushable boxes still block
                cant_push = push_idx[~pushable] if pushable.dim() > 0 else b_idx.new_empty(0)
                if cant_push.numel() > 0:
                    blocked[cant_push] = True

            # Also block if agent would stay in place (clamped to boundary)
            blocked[b_idx] = blocked[b_idx] | (
                (dest_y[b_idx] == agent_pos[b_idx, 0]) & (dest_x[b_idx] == agent_pos[b_idx, 1])
            )

        # Execute movements for unblocked samples
        valid = is_direction & ~blocked
        if valid.any():
            v_idx = torch.nonzero(valid, as_tuple=False).flatten()
            old = agent_pos[v_idx]
            new_y = dest_y[v_idx]
            new_x = dest_x[v_idx]

            # Sokoban box-pushing: move box before moving agent
            is_pushing = pushing_box[v_idx]
            if is_pushing.any():
                push_v_idx = v_idx[is_pushing]
                bd_y = box_dest_y[push_v_idx]
                bd_x = box_dest_x[push_v_idx]
                # Determine what the box lands on
                behind_val = next_states[push_v_idx, bd_y, bd_x]
                # If behind is a target, box becomes BOX_ON_TARGET; otherwise stays BOX
                on_target = (behind_val == self.config.target_value)
                box_new_val = torch.where(on_target,
                    torch.tensor(self.config.box_on_target, device=self.device),
                    torch.tensor(self.config.box_value, device=self.device),
                )
                next_states[push_v_idx, bd_y, bd_x] = box_new_val
                # Clear box from its original position (agent will take that spot)
                next_states[push_v_idx, new_y[is_pushing], new_x[is_pushing]] = self.config.empty_value

            # Restore original cell at old position if it was a persistent object
            original_old_vals = states[v_idx, old[:, 0], old[:, 1]]
            persistent_mask = torch.isin(
                original_old_vals,
                torch.tensor([
                    self.config.switch_a_value, self.config.switch_b_value,
                    self.config.teleporter_blue, self.config.teleporter_red,
                ], device=self.device),
            )
            next_states[v_idx, old[:, 0], old[:, 1]] = torch.where(
                persistent_mask, original_old_vals,
                torch.full_like(original_old_vals, self.config.empty_value),
            )
            reached_goal = self.goal_mask[0, new_y, new_x]
            next_states[v_idx, new_y, new_x] = self.config.agent_value
            energy[v_idx] = torch.where(
                reached_goal, torch.zeros_like(energy[v_idx]), 0.25,
            )

        energy[blocked] = 5.0

        # --- Post-move interactions using ORIGINAL destination values ---
        # We need to apply effects based on what the agent stepped ON,
        # not what the cell contains after the agent moved there.

        # 1. Switch activation: agent stepped on a switch -> toggle matching doors
        moved = is_direction if is_direction.any() else torch.zeros(states.shape[0], dtype=torch.bool, device=self.device)
        next_states = self._apply_switches_from_dest(next_states, original_dest_val, moved)

        # 2. Teleportation: agent stepped on a teleporter -> teleport to pair
        next_states = self._apply_teleporters_from_dest(next_states, original_dest_val, moved)

        # 3. USE action: interact with the switch/teleporter the agent is standing on
        use_mask = actions == int(Action.USE)
        if use_mask.any():
            next_states, use_energy = self._apply_use(next_states, agent_pos, use_mask)
            energy[use_mask] = use_energy[use_mask]

        # Passive / unrecognized actions
        passive = ~torch.isin(
            actions,
            torch.tensor([int(a) for a in DIRECTION_DELTAS] + [int(Action.USE)], device=self.device),
        )
        if passive.any():
            energy[passive] = 1.5

        # Reconstruct multi-channel output if input was multi-channel
        if multi_channel is not None:
            next_states = self._reconstruct_multichannel(multi_channel, next_states, actions, agent_pos, dest_y, dest_x, original_dest_val, is_direction)

        return next_states, energy

    def _reconstruct_multichannel(
        self,
        original_mc: torch.Tensor,
        next_gray: torch.Tensor,
        actions: torch.Tensor,
        old_agent_pos: torch.Tensor,
        dest_y: torch.Tensor,
        dest_x: torch.Tensor,
        original_dest_val: torch.Tensor,
        is_direction: torch.Tensor,
    ) -> torch.Tensor:
        """Rebuild multi-channel tensor from single-channel physics result."""
        B, C, H, W = original_mc.shape
        out = original_mc.clone()

        # Clear all channels at old agent positions
        for b in range(B):
            ay, ax = old_agent_pos[b]
            out[b, :, ay, ax] = 0
            # Set floor channel at old position (unless it's a persistent object)
            old_val = int(original_mc[b, 0, ay, ax].item()) if C > 0 else 0
            out[b, 0, ay, ax] = 1  # floor at vacated spot

        # For moved agents, set agent channel at new position
        moved = is_direction & (next_gray[torch.arange(B, device=self.device), old_agent_pos[:, 0], old_agent_pos[:, 1]] != next_gray[torch.arange(B, device=self.device), dest_y, dest_x])
        if moved.any():
            m_idx = torch.nonzero(moved, as_tuple=False).flatten()
            for b in m_idx:
                ny, nx = int(dest_y[b].item()), int(dest_x[b].item())
                # Set all channels at new position to 0, then set agent
                out[b, :, ny, nx] = 0
                out[b, 3, ny, nx] = 1  # agent channel (index 3)

        # Channel layout: 0=wall, 1=interactive, 2=dynamic(agent), 3=floor
        # Recompute channels from the final grayscale state
        for b in range(B):
            g = next_gray[b]
            out[b, 0] = (g == self.config.wall_value).long()
            interactive = torch.zeros(H, W, dtype=torch.long, device=self.device)
            for v in (self.config.switch_a_value, self.config.switch_b_value,
                      self.config.door_a_closed, self.config.door_a_open,
                      self.config.door_b_closed, self.config.door_b_open,
                      self.config.teleporter_blue, self.config.teleporter_red):
                interactive = interactive | (g == v)
            out[b, 1] = interactive.long()
            out[b, 2] = (g == self.config.agent_value).long()
            out[b, 3] = ((g == self.config.empty_value) | (g == self.config.goal_value) | (g == self.config.box_value) | (g == self.config.target_value) | (g == self.config.box_on_target)).long()

        return out

    def _build_per_sample_collision_masks(self, states: torch.Tensor) -> torch.Tensor:
        """Build (B, H, W) collision masks: walls + closed doors per batch element."""
        B, H, W = states.shape
        masks = self.static_wall_mask.expand(B, -1, -1).clone()
        # For each sample, check for closed doors
        for b in range(B):
            door_a_closed = (states[b] == self.config.door_a_closed)
            door_b_closed = (states[b] == self.config.door_b_closed)
            masks[b] = masks[b] | door_a_closed | door_b_closed
        return masks

    # ------------------------------------------------------------------
    # Interactive object handlers
    # ------------------------------------------------------------------

    def _apply_switches_from_dest(
        self,
        grid: torch.Tensor,
        original_dest_val: torch.Tensor,
        moved_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Toggle doors if the agent stepped on a switch (using pre-move cell values)."""
        if not moved_mask.any():
            return grid
        for b in range(grid.shape[0]):
            if not moved_mask[b]:
                continue
            cell_val = int(original_dest_val[b].item())
            if cell_val == self.config.switch_a_value:
                grid = self._toggle_doors_in_batch(grid, b, "a")
            elif cell_val == self.config.switch_b_value:
                grid = self._toggle_doors_in_batch(grid, b, "b")
        return grid

    def _apply_teleporters_from_dest(
        self,
        grid: torch.Tensor,
        original_dest_val: torch.Tensor,
        moved_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Teleport agent if it stepped on a teleporter (using pre-move cell values).

        Blue (10) and Red (11) form a pair: stepping on blue teleports to
        red, and stepping on red teleports to blue.
        """
        grid = grid.clone()
        teleporter_pair = {
            self.config.teleporter_blue: self.config.teleporter_red,
            self.config.teleporter_red: self.config.teleporter_blue,
        }
        for b in range(grid.shape[0]):
            if not moved_mask[b]:
                continue
            cell_val = int(original_dest_val[b].item())
            if cell_val not in teleporter_pair:
                continue
            target_val = teleporter_pair[cell_val]
            # Find the destination position (the OTHER teleporter in the pair)
            positions = (grid[b] == target_val).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            agent_yx = self._agent_positions(grid[b : b + 1])[0]
            # Pick the first available target position
            dy, dx = int(positions[0][0].item()), int(positions[0][1].item())
            # Restore the teleporter tile at the source position
            grid[b, agent_yx[0], agent_yx[1]] = cell_val
            grid[b, dy, dx] = self.config.agent_value
        return grid

    def _apply_use(
        self,
        grid: torch.Tensor,
        agent_pos: torch.Tensor,
        use_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """USE action: activate the switch or teleporter the agent is standing on."""
        energy = torch.ones(grid.shape[0], device=self.device, dtype=torch.float32)
        grid = grid.clone()
        teleporter_pair = {
            self.config.teleporter_blue: self.config.teleporter_red,
            self.config.teleporter_red: self.config.teleporter_blue,
        }
        for b in range(grid.shape[0]):
            if not use_mask[b]:
                continue
            pos = agent_pos[b]
            cell_val = int(grid[b, pos[0], pos[1]].item())
            if cell_val == self.config.switch_a_value:
                grid = self._toggle_doors_in_batch(grid, b, "a")
                energy[b] = 0.3
            elif cell_val == self.config.switch_b_value:
                grid = self._toggle_doors_in_batch(grid, b, "b")
                energy[b] = 0.3
            elif cell_val in (self.config.teleporter_blue, self.config.teleporter_red):
                target_val = teleporter_pair.get(cell_val)
                if target_val is not None:
                    positions = (grid[b] == target_val).nonzero(as_tuple=False)
                    if positions.numel() > 0:
                        dy, dx = int(positions[0][0].item()), int(positions[0][1].item())
                        grid[b, pos[0], pos[1]] = cell_val  # preserve teleporter tile
                        grid[b, dy, dx] = self.config.agent_value
                energy[b] = 0.3
            else:
                energy[b] = 1.5
        return grid, energy

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

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

    def sokoban_win_energy(self, states: torch.Tensor) -> torch.Tensor:
        """Check if Sokoban puzzle is solved: all targets covered by boxes."""
        states = self.to_device(states)
        uncovered_targets = (states == self.config.target_value).float().flatten(1).sum(dim=1)
        covered_targets = (states == self.config.box_on_target).float().flatten(1).sum(dim=1)
        total_targets = uncovered_targets + covered_targets
        solved = (total_targets > 0) & (uncovered_targets == 0)
        return torch.where(solved, torch.zeros_like(total_targets, dtype=torch.float32), torch.ones_like(total_targets, dtype=torch.float32))

    def sokoban_boxes_on_targets(self, states: torch.Tensor) -> torch.Tensor:
        """Count how many boxes are on target cells."""
        states = self.to_device(states)
        box_on_target_mask = (states == self.config.box_on_target).float()
        return box_on_target_mask.flatten(1).sum(dim=1)
