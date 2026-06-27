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

# Direction deltas for the 4 movement actions
_DIRECTION_DELTAS = {
    1: (-1, 0),  # ACTION1 Up
    2: (+1, 0),  # ACTION2 Down
    3: (0, -1),  # ACTION3 Left
    4: (0, +1),  # ACTION4 Right
}


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


# ---------------------------------------------------------------------------
# Particle Filter for Hidden Player State Estimation
# ---------------------------------------------------------------------------

class ParticleFilter:
    """Estimates the hidden player position using a particle filter.

    At initialization, spawns particles across all walkable tiles. Each action
    updates every particle's position. Frame diffs (box movement) are used to
    collapse the particle cloud toward the true position.
    """

    def __init__(
        self,
        walkable_mask: torch.Tensor,
        num_particles: int = 200,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.walkable_mask = walkable_mask.to(self.device)
        H, W = walkable_mask.shape
        self.H, self.W = H, W
        self.num_particles = num_particles

        # Spawn particles uniformly across walkable tiles
        walkable_coords = torch.nonzero(walkable_mask, as_tuple=False).float()
        if walkable_coords.shape[0] == 0:
            # Fallback: center of grid
            walkable_coords = torch.tensor([[H // 2, W // 2]], dtype=torch.float)

        indices = torch.randint(0, walkable_coords.shape[0], (num_particles,), device=self.device)
        self.particles = walkable_coords[indices].clone()  # (N, 2) as (y, x)
        self.weights = torch.ones(num_particles, device=self.device) / num_particles

    def predict(self, action: int) -> None:
        """Move every particle according to the action. Particles that hit
        walls stay put."""
        if action not in _DIRECTION_DELTAS:
            return
        dy, dx = _DIRECTION_DELTAS[action]
        new_y = self.particles[:, 0].long() + dy
        new_x = self.particles[:, 1].long() + dx

        # Clamp to bounds
        new_y = torch.clamp(new_y, 0, self.H - 1)
        new_x = torch.clamp(new_x, 0, self.W - 1)

        # Check walkable
        can_move = self.walkable_mask[new_y, new_x]

        # Update particles that can move
        self.particles[:, 0] = torch.where(can_move, new_y.float(), self.particles[:, 0])
        self.particles[:, 1] = torch.where(can_move, new_x.float(), self.particles[:, 1])

    def correct_from_frame_diff(
        self,
        prev_grid: torch.Tensor,
        curr_grid: torch.Tensor,
    ) -> None:
        """Sensor correction: if a box moved, kill particles not adjacent to
        the old box position. This is the key insight — box movement implies
        the player was next to it."""
        # Ensure both grids are 2D and match particle filter dimensions
        prev_grid = prev_grid.squeeze()
        curr_grid = curr_grid.squeeze()
        if prev_grid.ndim != 2 or curr_grid.ndim != 2:
            return
        if prev_grid.shape[0] != self.H or prev_grid.shape[1] != self.W:
            return
        if curr_grid.shape[0] != self.H or curr_grid.shape[1] != self.W:
            return

        # Find cells that changed
        diff = prev_grid != curr_grid
        if not diff.any():
            return

        # Find boxes that moved (were box in prev, not box in curr)
        box_mask_prev = (prev_grid == 8)  # ARC box value
        box_mask_curr = (curr_grid == 8)
        box_moved_from = box_mask_prev & ~box_mask_curr  # cells that lost a box

        if not box_moved_from.any():
            # Kill particles that are in wall cells in the current grid
            wall_mask = torch.isin(curr_grid, torch.tensor(sorted(ARC_WALL_VALUES), device=self.device))
            in_wall = wall_mask[
                self.particles[:, 0].long().clamp(0, self.H - 1),
                self.particles[:, 1].long().clamp(0, self.W - 1),
            ]
            self.weights[in_wall] *= 0.01
            self._normalize_weights()
            return

        # Get the positions where boxes were pushed FROM
        old_box_positions = torch.nonzero(box_moved_from, as_tuple=False).float()  # (K, 2)

        # For each particle, check if it was adjacent to any old box position
        for bp in old_box_positions:
            adj = torch.tensor([
                [bp[0] - 1, bp[1]],
                [bp[0] + 1, bp[1]],
                [bp[0], bp[1] - 1],
                [bp[0], bp[1] + 1],
            ], device=self.device, dtype=torch.float)

            dists = torch.cdist(self.particles, adj, p=1)  # (N, 4)
            min_dist = dists.min(dim=1).values  # (N,)
            not_adjacent = min_dist > 0.5
            self.weights[not_adjacent] *= 0.001

        self._normalize_weights()

    def correct_from_wall_collision(
        self,
        action: int,
        prev_grid: torch.Tensor,
        curr_grid: torch.Tensor,
    ) -> None:
        """If the grid didn't change after a movement action, the player
        likely hit a wall. Kill particles that would have moved."""
        if action not in _DIRECTION_DELTAS:
            return
        # If grids are identical, player was blocked
        if torch.equal(prev_grid, curr_grid):
            dy, dx = _DIRECTION_DELTAS[action]
            # Particles that COULD have moved (were on walkable, new pos walkable)
            # are less likely if the player was blocked
            # Actually: if player was blocked, particles at the wall boundary
            # are MORE likely, not less. Skip correction here.
            pass

    def resample(self) -> None:
        """Systematic resampling to focus on high-weight particles."""
        indices = torch.multinomial(self.weights, self.num_particles, replacement=True)
        self.particles = self.particles[indices].clone()
        self.weights = torch.ones(self.num_particles, device=self.device) / self.num_particles

    def _normalize_weights(self) -> None:
        total = self.weights.sum()
        if total > 0:
            self.weights /= total
        else:
            # All particles killed — reinitialize
            walkable_coords = torch.nonzero(self.walkable_mask, as_tuple=False).float()
            indices = torch.randint(0, walkable_coords.shape[0], (self.num_particles,), device=self.device)
            self.particles = walkable_coords[indices].clone()
            self.weights = torch.ones(self.num_particles, device=self.device) / self.num_particles

    def estimate_position(self) -> tuple[int, int]:
        """Weighted average position of all particles."""
        weighted_y = (self.particles[:, 0] * self.weights).sum()
        weighted_x = (self.particles[:, 1] * self.weights).sum()
        return int(weighted_y.item()), int(weighted_x.item())

    def effective_particles(self) -> float:
        """Effective sample size — low means particles have collapsed."""
        return 1.0 / (self.weights ** 2).sum().item()

    def is_converged(self, threshold: float = 5.0) -> bool:
        """True when the particle cloud has collapsed to ~1 location."""
        return self.effective_particles() < threshold


# ---------------------------------------------------------------------------
# Frame Diff Analyzer
# ---------------------------------------------------------------------------

class FrameDiffAnalyzer:
    """Compares consecutive frames to detect what changed and infer
    player movement direction."""

    def __init__(self) -> None:
        self.prev_grid: torch.Tensor | None = None

    def update(self, grid: torch.Tensor) -> dict[str, Any]:
        """Analyze the diff between previous and current frame.
        Returns info about what changed."""
        if self.prev_grid is None:
            self.prev_grid = grid.clone()
            return {"changed": False, "boxes_moved": False, "diff_count": 0}

        diff = self.prev_grid != grid
        diff_count = int(diff.sum().item())

        # Detect box movement
        box_was = self.prev_grid == 8
        box_now = grid == 8
        boxes_lost = box_was & ~box_now  # cells that lost a box
        boxes_gained = ~box_was & box_now  # cells that gained a box

        boxes_moved = boxes_lost.any().item()
        box_positions = []
        if boxes_moved:
            box_positions = torch.nonzero(boxes_lost, as_tuple=False).tolist()

        # Detect target/9 changes
        target_was = self.prev_grid == 9
        target_now = grid == 9
        targets_changed = (target_was != target_now).any().item()

        self.prev_grid = grid.clone()

        return {
            "changed": diff_count > 0,
            "diff_count": diff_count,
            "boxes_moved": boxes_moved,
            "box_from_positions": box_positions,
            "targets_changed": targets_changed,
        }


# ---------------------------------------------------------------------------
# Main Agent
# ---------------------------------------------------------------------------

class Monarch_AI(Agent):
    """ARC-AGI-3 adapter class registered as Monarch_AI.

    Uses a three-phase strategy:
    Phase 1 (frames 0-19): Corner Calibration — force UP+LEFT to anchor position
    Phase 2 (frames 20-29): Particle Filter convergence — probe and observe
    Phase 3 (frames 30+): MCTS with estimated position from particle filter
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

        # Phase tracking
        self._frame_count = 0
        self._phase = "calibrate"  # calibrate -> converge -> play

        # Particle filter (initialized on first frame)
        self._particle_filter: ParticleFilter | None = None
        self._frame_analyzer = FrameDiffAnalyzer()

        # Calibration state
        self._calib_up_count = 0
        self._calib_left_count = 0
        self._calib_done = False

        # Estimated position (from particle filter)
        self._est_pos = [32, 20]

    def _reset_state(self) -> None:
        """Reset all internal state on game start/restart."""
        self._global_positions.clear()
        self._action_history.clear()
        self._frame_count = 0
        self._phase = "calibrate"
        self._particle_filter = None
        self._frame_analyzer = FrameDiffAnalyzer()
        self._calib_up_count = 0
        self._calib_left_count = 0
        self._calib_done = False
        self._est_pos = [32, 20]

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

    def _init_particle_filter(self, gray_grid: torch.Tensor) -> None:
        """Initialize the particle filter with walkable tiles from the grid."""
        # Walkable after remap: 0=empty, 2=agent, 3=goal, 12=box, 13=target
        walkable = torch.isin(gray_grid.squeeze(0), torch.tensor([0, 2, 3, 12, 13]))
        self._particle_filter = ParticleFilter(
            walkable_mask=walkable,
            num_particles=200,
            device=str(gray_grid.device),
        )

    def _phase_calibrate(
        self, gray_grid: torch.Tensor, available: list[int]
    ) -> int:
        """Phase 1: Force UP and LEFT to anchor position to a known corner.

        Strategy: 10x UP then 10x LEFT. After this, the hidden player is
        guaranteed to be at or near the top-left walkable corner.
        """
        ACTION_UP = 1
        ACTION_LEFT = 3

        # First 10 actions: UP
        if self._calib_up_count < 10:
            self._calib_up_count += 1
            if ACTION_UP in available:
                return ACTION_UP
            # If UP not available, skip to LEFT phase
            self._calib_up_count = 10

        # Next 10 actions: LEFT
        if self._calib_left_count < 10:
            self._calib_left_count += 1
            if ACTION_LEFT in available:
                return ACTION_LEFT
            # If LEFT not available, skip to convergence
            self._calib_left_count = 10

        # Calibration done — move to convergence phase
        self._calib_done = True
        self._phase = "converge"

        # Initialize particle filter at this point
        self._init_particle_filter(gray_grid)

        # After calibration, player should be at top-left corner
        # Find the top-left walkable cell
        walkable = torch.isin(gray_grid.squeeze(0), torch.tensor([0, 2, 3, 12, 13]))
        walkable_coords = torch.nonzero(walkable, as_tuple=False)
        if walkable_coords.shape[0] > 0:
            # Find the walkable cell closest to (0, 0)
            dists = walkable_coords.float().sum(dim=1)
            closest = walkable_coords[dists.argmin()]
            self._est_pos = [int(closest[0].item()), int(closest[1].item())]

        # Return the last LEFT action
        if ACTION_LEFT in available:
            return ACTION_LEFT
        return available[0] if available else 1

    def _phase_converge(
        self, gray_grid: torch.Tensor, available: list[int],
        prev_grid: torch.Tensor | None,
    ) -> int:
        """Phase 2: Probe the environment to collapse the particle filter.

        Move in known directions and observe frame diffs to narrow down
        the particle cloud. After ~10 probing moves, the filter should
        converge.
        """
        if self._particle_filter is None:
            self._init_particle_filter(gray_grid)

        pf = self._particle_filter

        # Analyze frame diff
        diff_info = self._frame_analyzer.update(gray_grid.squeeze(0))

        # Apply sensor correction if boxes moved
        if prev_grid is not None and diff_info["boxes_moved"]:
            pf.correct_from_frame_diff(prev_grid.squeeze(0), gray_grid.squeeze(0))
            # Resample after correction
            if pf.effective_particles() > 10:
                pf.resample()

        # Choose probing direction: spiral right then down
        probe_sequence = [4, 4, 4, 2, 2, 2, 3, 3, 1, 1]  # RRRDDDLLU
        step_in_converge = self._frame_count - 20  # steps since convergence started

        if step_in_converge < len(probe_sequence):
            action = probe_sequence[step_in_converge]
            if action in available:
                # Predict particle movement
                pf.predict(action)

                # Update estimated position
                self._est_pos = list(pf.estimate_position())

                # Check convergence
                if pf.is_converged(threshold=5.0) or step_in_converge >= 9:
                    self._phase = "play"

                return action

        # Fallback: try any available direction
        for action in [4, 2, 3, 1]:
            if action in available:
                pf.predict(action)
                self._est_pos = list(pf.estimate_position())
                return action

        return available[0] if available else 1

    def _phase_play(
        self, gray_grid: torch.Tensor, available: list[int],
        prev_grid: torch.Tensor | None,
    ) -> int:
        """Phase 3: Execute MCTS with the estimated position from the particle filter."""
        if self._particle_filter is not None:
            pf = self._particle_filter

            # Analyze frame diff for continuous correction
            diff_info = self._frame_analyzer.update(gray_grid.squeeze(0))
            if prev_grid is not None and diff_info["changed"]:
                pf.correct_from_frame_diff(prev_grid.squeeze(0), gray_grid.squeeze(0))
                if pf.effective_particles() > 20:
                    pf.resample()

            self._est_pos = list(pf.estimate_position())

        # Oscillation breakout
        if self._detect_oscillation() and len(available) > 2:
            last_two = set(self._action_history[-2:])
            unexplored = [a for a in available if a not in last_two]
            if unexplored:
                selected_id = self._rng.choice(unexplored)
                self._action_history.append(selected_id)
                self._global_positions.append(tuple(self._est_pos))
                action = GameAction.from_id(int(selected_id))
                if action.is_complex():
                    action.set_data({"x": self._est_pos[1], "y": self._est_pos[0], "game_id": getattr(self, "game_id", "")})
                action.reasoning = {
                    "agent": "Monarch_AI",
                    "architecture": "SOMA-Mythos-EHRA",
                    "strategy": "oscillation_breakout",
                    "est_pos": list(self._est_pos),
                }
                return action

        # MCTS with runtime
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

        self._global_positions.append(tuple(self._est_pos))
        self._action_history.append(selected_id)

        # Predict particle movement from chosen action
        if self._particle_filter is not None:
            self._particle_filter.predict(selected_id)
            self._est_pos = list(self._particle_filter.estimate_position())

        action = GameAction.from_id(int(selected_id))
        if action.is_complex():
            action.set_data({"x": self._est_pos[1], "y": self._est_pos[0], "game_id": getattr(self, "game_id", "")})
        action.reasoning = {
            "agent": "Monarch_AI",
            "architecture": "SOMA-Mythos-EHRA",
            "strategy": "mcts_with_particle_filter",
            "est_pos": list(self._est_pos),
            "effective_particles": self._particle_filter.effective_particles() if self._particle_filter else 0,
            "telemetry": str(result.telemetry_path),
        }
        return action

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self._reset_state()
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

        # Get previous grid for diff analysis
        prev_grid = self._frame_analyzer.prev_grid

        # Route to the appropriate phase
        if self._phase == "calibrate":
            selected_id = self._phase_calibrate(gray_grid, available)
        elif self._phase == "converge":
            selected_id = self._phase_converge(gray_grid, available, prev_grid)
        else:
            result = self._phase_play(gray_grid, available, prev_grid)
            if isinstance(result, GameAction):
                return result
            selected_id = result

        self._frame_count += 1

        action = GameAction.from_id(int(selected_id))
        if action.is_complex():
            action.set_data({"x": self._est_pos[1], "y": self._est_pos[0], "game_id": getattr(self, "game_id", "")})
        action.reasoning = {
            "agent": "Monarch_AI",
            "architecture": "SOMA-Mythos-EHRA",
            "phase": self._phase,
            "frame": self._frame_count,
            "est_pos": list(self._est_pos),
        }
        return action
