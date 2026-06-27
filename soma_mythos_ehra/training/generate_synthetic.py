"""Procedural grid physics data generator with ARC-style objects.

Generates large-scale synthetic (state, action, next_state) transitions
by running random action sequences on the TensorGridSimulator with
randomized grid configurations. Teaches JEPA core physics plus
interactive object semantics:

  - Movement, wall collision, boundary containment
  - Switch activation toggling doors (pair A and pair B)
  - Door blocking when closed, passability when open
  - Teleporter pair teleportation

Usage:
    python -m soma_mythos_ehra.training.generate_synthetic \\
        --output-dir synthetic_recordings \\
        --num-sequences 2000 \\
        --steps-per-seq 50
"""
from __future__ import annotations

import argparse
import json
import os
import uuid

import torch

from soma_mythos_ehra.soma.gpu_simulator import SimulatorConfig, TensorGridSimulator
from soma_mythos_ehra.types import CellType


def generate_grid(
    h: int,
    w: int,
    wall_density: float,
    include_objects: bool,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """Generate a random grid with walls, agent, goal, and optional ARC objects.

    When *include_objects* is True the grid may contain:
      - Switch A (4) / Door A closed (6) pairs
      - Switch B (5) / Door B closed (8) pairs
      - Teleporter Blue (10) / Teleporter Red (11) pairs

    Returns:
        grid: (H, W) tensor
        agent_pos: (y, x) position
        goal_pos: (y, x) position
    """
    grid = torch.zeros(h, w, dtype=torch.long, device=device)

    # Place walls randomly
    wall_mask = torch.rand(h, w, device=device) < wall_density
    grid[wall_mask] = int(CellType.WALL)

    # Place agent at random empty cell
    empty_cells = torch.nonzero(grid == int(CellType.EMPTY), as_tuple=False)
    if len(empty_cells) == 0:
        grid[h // 2, w // 2] = int(CellType.EMPTY)
        empty_cells = torch.nonzero(grid == int(CellType.EMPTY), as_tuple=False)

    agent_idx = torch.randint(0, len(empty_cells), (1,)).item()
    agent_pos = tuple(empty_cells[agent_idx].tolist())
    grid[agent_pos[0], agent_pos[1]] = int(CellType.AGENT)

    # Place goal at random empty cell (not agent position)
    empty_cells = torch.nonzero(grid == int(CellType.EMPTY), as_tuple=False)
    if len(empty_cells) > 0:
        goal_idx = torch.randint(0, len(empty_cells), (1,)).item()
        goal_pos = tuple(empty_cells[goal_idx].tolist())
        grid[goal_pos[0], goal_pos[1]] = int(CellType.GOAL)
    else:
        goal_pos = agent_pos

    if not include_objects:
        return grid, agent_pos, goal_pos

    # --- Place interactive objects ---
    # Reserve space: we need at least 3 empty cells for a meaningful pair
    empty_cells = torch.nonzero(grid == int(CellType.EMPTY), as_tuple=False)
    if len(empty_cells) < 6:
        return grid, agent_pos, goal_pos

    # Teleporter pair (Blue -> Red)
    perm = torch.randperm(len(empty_cells))
    t1_idx, t2_idx = int(perm[0].item()), int(perm[1].item())
    t1 = tuple(empty_cells[t1_idx].tolist())
    t2 = tuple(empty_cells[t2_idx].tolist())
    grid[t1[0], t1[1]] = int(CellType.TELEPORTER_BLUE)
    grid[t2[0], t2[1]] = int(CellType.TELEPORTER_RED)

    # Switch A + Door A pair
    empty_cells = torch.nonzero(grid == int(CellType.EMPTY), as_tuple=False)
    if len(empty_cells) >= 4:
        perm = torch.randperm(len(empty_cells))
        s_idx, d_idx = int(perm[0].item()), int(perm[1].item())
        s_pos = tuple(empty_cells[s_idx].tolist())
        d_pos = tuple(empty_cells[d_idx].tolist())
        grid[s_pos[0], s_pos[1]] = int(CellType.SWITCH_A)
        grid[d_pos[0], d_pos[1]] = int(CellType.DOOR_A_CLOSED)

        # Switch B + Door B pair (if room)
        empty_cells = torch.nonzero(grid == int(CellType.EMPTY), as_tuple=False)
        if len(empty_cells) >= 4:
            perm = torch.randperm(len(empty_cells))
            s2_idx, d2_idx = int(perm[0].item()), int(perm[1].item())
            s2_pos = tuple(empty_cells[s2_idx].tolist())
            d2_pos = tuple(empty_cells[d2_idx].tolist())
            grid[s2_pos[0], s2_pos[1]] = int(CellType.SWITCH_B)
            grid[d2_pos[0], d2_pos[1]] = int(CellType.DOOR_B_CLOSED)

    return grid, agent_pos, goal_pos


def grid_to_multichannel(grid: torch.Tensor) -> torch.Tensor:
    """Convert a single-channel grid to 4-channel multi-channel representation.

    Channel 0: Walls (value 1)
    Channel 1: Interactive (switches, doors, teleporters)
    Channel 2: Dynamic (agent=2, goal=3)
    Channel 3: Floor (empty=0)
    """
    H, W = grid.shape[-2], grid.shape[-1]
    out = torch.zeros(4, H, W, dtype=torch.long, device=grid.device)
    out[0] = (grid == int(CellType.WALL)).long()
    interactive = torch.zeros(H, W, dtype=torch.long, device=grid.device)
    for v in (int(CellType.SWITCH_A), int(CellType.SWITCH_B),
              int(CellType.DOOR_A_CLOSED), int(CellType.DOOR_A_OPEN),
              int(CellType.DOOR_B_CLOSED), int(CellType.DOOR_B_OPEN),
              int(CellType.TELEPORTER_BLUE), int(CellType.TELEPORTER_RED)):
        interactive = interactive | (grid == v)
    out[1] = interactive.long()
    out[2] = ((grid == int(CellType.AGENT)) | (grid == int(CellType.GOAL))).long()
    out[3] = (grid == int(CellType.EMPTY)).long()
    return out


def generate_sequence(
    grid: torch.Tensor,
    agent_pos: tuple[int, int],
    steps: int,
    device: torch.device,
) -> list[dict]:
    """Run random actions on grid and record transitions.

    Returns list of {"grid": list, "action": int, "next_grid": list}
    """
    config = SimulatorConfig()
    sim = TensorGridSimulator(grid, config=config, device=device)

    transitions = []
    current_grid = grid.clone()
    current_pos = agent_pos

    for _ in range(steps):
        # Mix directional moves (1-4) with USE action (5)
        action = torch.randint(1, 6, (1,)).item()

        state_before = current_grid.clone()
        next_grid, energy = sim.step_batch(
            current_grid.unsqueeze(0),
            torch.tensor([action], device=device),
        )
        next_grid = next_grid.squeeze(0)

        # Check if agent actually moved
        next_pos = _find_agent(next_grid)
        if next_pos is not None:
            current_grid = next_grid
            current_pos = next_pos
        # If no movement (wall/boundary), still record the transition

        transitions.append({
            "grid": state_before.cpu().tolist(),
            "action": int(action),
            "next_grid": current_grid.cpu().tolist(),
            "mc_grid": grid_to_multichannel(state_before).cpu().tolist(),
            "mc_next_grid": grid_to_multichannel(current_grid).cpu().tolist(),
            "energy": float(energy[0].item()),
        })

    return transitions


def _find_agent(grid: torch.Tensor) -> tuple[int, int] | None:
    """Find agent position (value 2) in grid."""
    positions = (grid == int(CellType.AGENT)).nonzero(as_tuple=False)
    if len(positions) == 0:
        return None
    return tuple(positions[0].tolist())


def generate_bulk(
    output_dir: str,
    num_sequences: int,
    steps_per_seq: int,
    min_size: int,
    max_size: int,
    wall_density_range: tuple[float, float],
    object_ratio: float,
) -> None:
    """Generate bulk synthetic physics data on GPU."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating synthetic grid physics on {device}...")

    total_transitions = 0
    all_transitions = []

    for seq_idx in range(num_sequences):
        h = torch.randint(min_size, max_size + 1, (1,)).item()
        w = torch.randint(min_size, max_size + 1, (1,)).item()
        wall_density = torch.empty(1).uniform_(*wall_density_range).item()
        include_objects = torch.rand(1).item() < object_ratio

        grid, agent_pos, goal_pos = generate_grid(h, w, wall_density, include_objects, device)
        transitions = generate_sequence(grid, agent_pos, steps_per_seq, device)
        all_transitions.extend(transitions)
        total_transitions += len(transitions)

        if (seq_idx + 1) % 100 == 0:
            print(f"  [{seq_idx + 1}/{num_sequences}] {total_transitions} transitions generated...")

    # Write all transitions to a single JSONL file
    output_path = os.path.join(output_dir, f"synthetic_{uuid.uuid4().hex[:8]}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in all_transitions:
            f.write(json.dumps(entry) + "\n")

    print(f"Done: {total_transitions} transitions written to {output_path}")
    print(f"Grid sizes: {min_size}x{min_size} to {max_size}x{max_size}")
    print(f"Wall density: {wall_density_range[0]:.1f}-{wall_density_range[1]:.1f}")
    print(f"Object ratio: {object_ratio:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic grid physics data")
    parser.add_argument("--output-dir", type=str, default="synthetic_recordings", help="Output directory")
    parser.add_argument("--num-sequences", type=int, default=2000, help="Number of random grid sequences")
    parser.add_argument("--steps-per-seq", type=int, default=50, help="Steps per sequence")
    parser.add_argument("--min-size", type=int, default=32, help="Minimum grid dimension")
    parser.add_argument("--max-size", type=int, default=64, help="Maximum grid dimension")
    parser.add_argument("--wall-density-min", type=float, default=0.05, help="Minimum wall density")
    parser.add_argument("--wall-density-max", type=float, default=0.3, help="Maximum wall density")
    parser.add_argument("--object-ratio", type=float, default=0.5, help="Fraction of grids with interactive objects")
    args = parser.parse_args()

    generate_bulk(
        output_dir=args.output_dir,
        num_sequences=args.num_sequences,
        steps_per_seq=args.steps_per_seq,
        min_size=args.min_size,
        max_size=args.max_size,
        wall_density_range=(args.wall_density_min, args.wall_density_max),
        object_ratio=args.object_ratio,
    )


if __name__ == "__main__":
    main()
