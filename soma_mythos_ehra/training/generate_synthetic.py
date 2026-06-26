"""Procedural grid physics data generator.

Generates large-scale synthetic (state, action, next_state) transitions
by running random action sequences on the TensorGridSimulator with
randomized grid configurations. This teaches JEPA core physics:
movement, wall collision, boundary containment — independent of any level.

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


def generate_grid(
    h: int,
    w: int,
    wall_density: float,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """Generate a random grid with walls, agent, and goal.

    Returns:
        grid: (H, W) tensor
        agent_pos: (y, x) position
        goal_pos: (y, x) position
    """
    grid = torch.zeros(h, w, dtype=torch.long, device=device)

    # Place walls randomly
    wall_mask = torch.rand(h, w, device=device) < wall_density
    grid[wall_mask] = 1

    # Place agent at random empty cell
    empty_cells = torch.nonzero(grid == 0, as_tuple=False)
    if len(empty_cells) == 0:
        grid[h // 2, w // 2] = 0
        empty_cells = torch.nonzero(grid == 0, as_tuple=False)

    agent_idx = torch.randint(0, len(empty_cells), (1,)).item()
    agent_pos = tuple(empty_cells[agent_idx].tolist())
    grid[agent_pos[0], agent_pos[1]] = 2

    # Place goal at random empty cell (not agent position)
    empty_cells = torch.nonzero(grid == 0, as_tuple=False)
    if len(empty_cells) > 0:
        goal_idx = torch.randint(0, len(empty_cells), (1,)).item()
        goal_pos = tuple(empty_cells[goal_idx].tolist())
        grid[goal_pos[0], goal_pos[1]] = 3
    else:
        goal_pos = agent_pos

    return grid, agent_pos, goal_pos


def generate_sequence(
    grid: torch.Tensor,
    agent_pos: tuple[int, int],
    steps: int,
    device: torch.device,
) -> list[dict]:
    """Run random actions on grid and record transitions.

    Returns list of {"grid": list, "action": int, "next_grid": list}
    """
    config = SimulatorConfig(wall_value=1, agent_value=2, goal_value=3, empty_value=0)
    sim = TensorGridSimulator(grid, config=config, device=device)

    transitions = []
    current_grid = grid.clone()
    current_pos = agent_pos

    for _ in range(steps):
        action = torch.randint(1, 5, (1,)).item()  # 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT

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
        # The "next_grid" is the same as current_grid — teaches collision physics

        transitions.append({
            "grid": state_before.cpu().tolist(),
            "action": int(action),
            "next_grid": current_grid.cpu().tolist(),
            "energy": float(energy[0].item()),
        })

    return transitions


def _find_agent(grid: torch.Tensor) -> tuple[int, int] | None:
    """Find agent position (value 2) in grid."""
    positions = (grid == 2).nonzero(as_tuple=False)
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

        grid, agent_pos, goal_pos = generate_grid(h, w, wall_density, device)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic grid physics data")
    parser.add_argument("--output-dir", type=str, default="synthetic_recordings", help="Output directory")
    parser.add_argument("--num-sequences", type=int, default=2000, help="Number of random grid sequences")
    parser.add_argument("--steps-per-seq", type=int, default=50, help="Steps per sequence")
    parser.add_argument("--min-size", type=int, default=8, help="Minimum grid dimension")
    parser.add_argument("--max-size", type=int, default=32, help="Maximum grid dimension")
    parser.add_argument("--wall-density-min", type=float, default=0.05, help="Minimum wall density")
    parser.add_argument("--wall-density-max", type=float, default=0.3, help="Maximum wall density")
    args = parser.parse_args()

    generate_bulk(
        output_dir=args.output_dir,
        num_sequences=args.num_sequences,
        steps_per_seq=args.steps_per_seq,
        min_size=args.min_size,
        max_size=args.max_size,
        wall_density_range=(args.wall_density_min, args.wall_density_max),
    )


if __name__ == "__main__":
    main()
