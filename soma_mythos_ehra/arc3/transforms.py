"""Grid Transformation Macros — the action space for ARC-AGI puzzle solving.

Each macro is a deterministic, reversible transformation that can be applied
to a grid tensor. MCTS searches sequences of these macros to find the
transformation that maps train inputs to train outputs.
"""
from __future__ import annotations

import torch
from enum import IntEnum


class TransformType(IntEnum):
    """Available grid transformation macros."""
    COLOR_MAP = 0       # Remap one color to another
    ROTATE_90 = 1       # Rotate 90 degrees clockwise
    ROTATE_180 = 2      # Rotate 180 degrees
    ROTATE_270 = 3      # Rotate 270 degrees clockwise
    FLIP_H = 4          # Flip horizontally
    FLIP_V = 5          # Flip vertically
    TRANSPOSE = 6       # Transpose (swap rows/cols)
    FLOOD_FILL = 7      # Flood fill connected region
    SHIFT_OBJECTS = 8   # Shift all non-background objects
    SCALE_UP = 9        # Scale grid by integer factor
    TILE = 10           # Tile/repeat pattern
    INVERT_COLORS = 11  # Invert all non-zero colors
    WRAP_AROUND = 12    # Cyclic shift of rows/cols


# Total action space size
NUM_TRANSFORMS = 13


def apply_color_map(grid: torch.Tensor, src: int, dst: int) -> torch.Tensor:
    """Remap all cells of color `src` to color `dst`."""
    out = grid.clone()
    out[grid == src] = dst
    return out


def apply_rotate_90(grid: torch.Tensor) -> torch.Tensor:
    """Rotate grid 90 degrees clockwise."""
    return torch.rot90(grid, k=-1, dims=(0, 1))


def apply_rotate_180(grid: torch.Tensor) -> torch.Tensor:
    """Rotate grid 180 degrees."""
    return torch.rot90(grid, k=2, dims=(0, 1))


def apply_rotate_270(grid: torch.Tensor) -> torch.Tensor:
    """Rotate grid 270 degrees clockwise (= 90 counter-clockwise)."""
    return torch.rot90(grid, k=1, dims=(0, 1))


def apply_flip_h(grid: torch.Tensor) -> torch.Tensor:
    """Flip grid horizontally (left-right)."""
    return grid.flip(1)


def apply_flip_v(grid: torch.Tensor) -> torch.Tensor:
    """Flip grid vertically (top-bottom)."""
    return grid.flip(0)


def apply_transpose(grid: torch.Tensor) -> torch.Tensor:
    """Transpose grid (swap rows and columns)."""
    return grid.t()


def apply_flood_fill(grid: torch.Tensor, seed_y: int, seed_x: int, new_color: int) -> torch.Tensor:
    """Flood fill connected region of same color starting from (seed_y, seed_x)."""
    H, W = grid.shape
    if seed_y < 0 or seed_y >= H or seed_x < 0 or seed_x >= W:
        return grid
    out = grid.clone()
    target_color = int(grid[seed_y, seed_x].item())
    if target_color == new_color:
        return out
    # BFS flood fill
    stack = [(seed_y, seed_x)]
    visited = set()
    while stack:
        y, x = stack.pop()
        if (y, x) in visited:
            continue
        if y < 0 or y >= H or x < 0 or x >= W:
            continue
        if int(out[y, x].item()) != target_color:
            continue
        visited.add((y, x))
        out[y, x] = new_color
        stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
    return out


def apply_shift_objects(grid: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Shift all non-zero objects by (dy, dx). Background (0) stays fixed."""
    out = torch.zeros_like(grid)
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            val = int(grid[r, c].item())
            if val != 0:
                nr, nc = r + dy, c + dx
                if 0 <= nr < H and 0 <= nc < W:
                    out[nr, nc] = val
    return out


def apply_scale_up(grid: torch.Tensor, factor: int = 3) -> torch.Tensor:
    """Scale grid up by an integer factor (each cell becomes factor x factor block)."""
    H, W = grid.shape
    out = grid.repeat_interleave(factor, dim=0).repeat_interleave(factor, dim=1)
    return out


def apply_tile(grid: torch.Tensor, reps_h: int = 3, reps_w: int = 3) -> torch.Tensor:
    """Tile/repeat the grid pattern."""
    return grid.repeat(reps_h, reps_w)


def apply_invert_colors(grid: torch.Tensor) -> torch.Tensor:
    """Invert all non-zero colors (1->max, 2->max-1, etc.)."""
    out = grid.clone()
    max_val = int(grid.max().item())
    for v in range(1, max_val + 1):
        out[grid == v] = max_val - v + 1
    return out


def apply_wrap_around(grid: torch.Tensor, shift: int = 1, axis: int = 0) -> torch.Tensor:
    """Cyclic shift of rows (axis=0) or columns (axis=1)."""
    return torch.roll(grid, shifts=shift, dims=axis)


# ---------------------------------------------------------------------------
# Unified apply function
# ---------------------------------------------------------------------------

def apply_transform(
    grid: torch.Tensor,
    transform: TransformType,
    **kwargs: int,
) -> torch.Tensor:
    """Apply a transformation macro to a grid tensor.

    Args:
        grid: Input grid (H, W) with integer cell values.
        transform: The transformation type to apply.
        **kwargs: Additional parameters for specific transforms.
            - For COLOR_MAP: src (int), dst (int)
            - For FLOOD_FILL: seed_y (int), seed_x (int), new_color (int)
            - For SHIFT_OBJECTS: dy (int), dx (int)
    """
    if transform == TransformType.COLOR_MAP:
        src = kwargs.get("src", 0)
        dst = kwargs.get("dst", 0)
        return apply_color_map(grid, src, dst)
    elif transform == TransformType.ROTATE_90:
        return apply_rotate_90(grid)
    elif transform == TransformType.ROTATE_180:
        return apply_rotate_180(grid)
    elif transform == TransformType.ROTATE_270:
        return apply_rotate_270(grid)
    elif transform == TransformType.FLIP_H:
        return apply_flip_h(grid)
    elif transform == TransformType.FLIP_V:
        return apply_flip_v(grid)
    elif transform == TransformType.TRANSPOSE:
        return apply_transpose(grid)
    elif transform == TransformType.FLOOD_FILL:
        seed_y = kwargs.get("seed_y", 0)
        seed_x = kwargs.get("seed_x", 0)
        new_color = kwargs.get("new_color", 0)
        return apply_flood_fill(grid, seed_y, seed_x, new_color)
    elif transform == TransformType.SHIFT_OBJECTS:
        dy = kwargs.get("dy", 0)
        dx = kwargs.get("dx", 0)
        return apply_shift_objects(grid, dy, dx)
    elif transform == TransformType.SCALE_UP:
        factor = kwargs.get("factor", 3)
        return apply_scale_up(grid, factor)
    elif transform == TransformType.TILE:
        reps_h = kwargs.get("reps_h", 3)
        reps_w = kwargs.get("reps_w", 3)
        return apply_tile(grid, reps_h, reps_w)
    elif transform == TransformType.INVERT_COLORS:
        return apply_invert_colors(grid)
    elif transform == TransformType.WRAP_AROUND:
        shift = kwargs.get("shift", 1)
        axis = kwargs.get("axis", 0)
        return apply_wrap_around(grid, shift, axis)
    else:
        raise ValueError(f"Unknown transform: {transform}")


# ---------------------------------------------------------------------------
# Sequence application
# ---------------------------------------------------------------------------

def apply_sequence(grid: torch.Tensor, sequence: list[dict]) -> torch.Tensor:
    """Apply a sequence of transformations to a grid.

    Each element in `sequence` is a dict with:
        - "transform": TransformType
        - Additional kwargs for the specific transform.
        - Special case: "color_map" dict for atomic color remapping.
    """
    state = grid.clone()
    for step in sequence:
        transform = step["transform"]
        # Handle atomic color map (applied simultaneously to avoid circular deps)
        if "color_map" in step:
            old = state.clone()
            for src, dst in step["color_map"].items():
                state[old == src] = dst
            continue
        kwargs = {k: v for k, v in step.items() if k != "transform"}
        state = apply_transform(state, transform, **kwargs)
    return state
