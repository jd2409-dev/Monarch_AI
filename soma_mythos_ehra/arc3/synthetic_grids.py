"""Synthetic Grid Generator — creates random structured grids for training.

Generates grids with:
- Random shapes (3x3 to 15x15)
- 1-5 colored objects (connected components)
- Holes, dense blocks, hollow rings
- Structured patterns (lines, corners, L-shapes)
"""
from __future__ import annotations

import random

import torch


def generate_random_grid(
    min_size: int = 3,
    max_size: int = 15,
    num_colors: int = 5,
    num_objects: int = 3,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate a random grid with structured objects.

    Returns:
        (H, W) tensor with integer cell values (0 = background).
    """
    if seed is not None:
        random.seed(seed)

    H = random.randint(min_size, max_size)
    W = random.randint(min_size, max_size)
    grid = torch.zeros(H, W, dtype=torch.long)

    for _ in range(num_objects):
        color = random.randint(1, num_colors)
        _place_random_object(grid, color)

    return grid


def _place_random_object(grid: torch.Tensor, color: int) -> None:
    """Place a random shaped object onto the grid."""
    H, W = grid.shape
    shape_type = random.choice(["block", "line_h", "line_v", "corner", "l_shape", "dot", "ring"])

    if shape_type == "block":
        bh = random.randint(1, min(4, H))
        bw = random.randint(1, min(4, W))
        ry = random.randint(0, H - bh)
        rx = random.randint(0, W - bw)
        grid[ry:ry+bh, rx:rx+bw] = color

    elif shape_type == "line_h":
        length = random.randint(2, min(6, W))
        ry = random.randint(0, H - 1)
        rx = random.randint(0, W - length)
        grid[ry, rx:rx+length] = color

    elif shape_type == "line_v":
        length = random.randint(2, min(6, H))
        ry = random.randint(0, H - length)
        rx = random.randint(0, W - 1)
        grid[ry:ry+length, rx] = color

    elif shape_type == "corner":
        arm_h = random.randint(1, min(3, H))
        arm_w = random.randint(1, min(3, W))
        ry = random.randint(0, H - arm_h)
        rx = random.randint(0, W - arm_w)
        grid[ry, rx:rx+arm_w] = color
        grid[ry:ry+arm_h, rx] = color

    elif shape_type == "l_shape":
        arm_h = random.randint(2, min(4, H))
        arm_w = random.randint(2, min(4, W))
        ry = random.randint(0, H - arm_h)
        rx = random.randint(0, W - arm_w)
        grid[ry:ry+arm_h, rx] = color
        grid[ry+arm_h-1, rx:rx+arm_w] = color

    elif shape_type == "dot":
        ry = random.randint(0, H - 1)
        rx = random.randint(0, W - 1)
        grid[ry, rx] = color

    elif shape_type == "ring":
        size = random.randint(3, min(5, H, W))
        ry = random.randint(0, H - size)
        rx = random.randint(0, W - size)
        for i in range(size):
            for j in range(size):
                if i == 0 or i == size-1 or j == 0 or j == size-1:
                    grid[ry+i, rx+j] = color


def generate_grid_pair(
    transform_type: str = "random",
    min_size: int = 3,
    max_size: int = 10,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate an (input, output) pair using a known transform.

    Args:
        transform_type: One of "rotate", "flip", "scale", "recolor", "shift", "compose".
        min_size: Minimum grid dimension.
        max_size: Maximum grid dimension.

    Returns:
        (input_grid, output_grid) tuple.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    inp = generate_random_grid(min_size, max_size)

    if transform_type == "random":
        transform_type = random.choice(["rotate", "flip", "scale", "recolor", "shift"])

    if transform_type == "rotate":
        k = random.choice([1, 2, 3])
        out = torch.rot90(inp, k=k, dims=(0, 1))

    elif transform_type == "flip":
        axis = random.choice([0, 1])
        out = inp.flip(axis)

    elif transform_type == "scale":
        factor = random.choice([2, 3])
        out = inp.repeat_interleave(factor, dim=0).repeat_interleave(factor, dim=1)

    elif transform_type == "recolor":
        src = random.randint(1, 5)
        dst = random.randint(1, 5)
        out = inp.clone()
        out[inp == src] = dst

    elif transform_type == "shift":
        dy = random.choice([-2, -1, 1, 2])
        dx = random.choice([-2, -1, 1, 2])
        out = torch.roll(inp, shifts=(dy, dx), dims=(0, 1))

    else:
        out = inp.clone()

    return inp, out
