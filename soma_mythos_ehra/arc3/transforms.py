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
    MOVE_OBJECT = 13    # Move a specific object by (dy, dx)
    FILL_HOLES = 14     # Fill interior holes in objects
    SORT_OBJECTS = 15   # Sort objects by position/size
    RECOLOR_BY_SIZE = 16    # Recolor objects based on size ordering
    SORT_BY_DENSITY = 17    # Sort objects by density (solidness)
    SORT_BY_CENTROID = 18   # Sort objects by centroid position


# Total action space size
NUM_TRANSFORMS = 19


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


def apply_move_object(grid: torch.Tensor, obj_label: int, dy: int, dx: int) -> torch.Tensor:
    """Move a specific connected component by (dy, dx).

    Uses connected component labeling to find the object with the given label,
    then shifts all its pixels.
    """
    from soma_mythos_ehra.arc3.objects import connected_component_labeling
    H, W = grid.shape
    labels = connected_component_labeling(grid)
    mask = (labels == obj_label)
    if not mask.any():
        return grid

    out = grid.clone()
    # Clear old position
    out[mask] = 0
    # Paint new position
    pixel_coords = mask.nonzero(as_tuple=False)
    for p in pixel_coords:
        r, c = p[0].item(), p[1].item()
        nr, nc = r + dy, c + dx
        if 0 <= nr < H and 0 <= nc < W:
            out[nr, nc] = grid[r, c]
    return out


def apply_fill_holes(grid: torch.Tensor) -> torch.Tensor:
    """Fill interior holes (background pockets fully enclosed by non-background).

    Uses flood fill from edges to find all reachable background cells,
    then fills unreachable background cells with the surrounding color.
    """
    H, W = grid.shape
    background = 0

    # BFS from all edge background cells
    reachable = torch.zeros(H, W, dtype=torch.bool)
    stack = []
    for r in range(H):
        for c in [0, W - 1]:
            if int(grid[r, c].item()) == background:
                stack.append((r, c))
    for c in range(W):
        for r in [0, H - 1]:
            if int(grid[r, c].item()) == background:
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        if r < 0 or r >= H or c < 0 or c >= W:
            continue
        if reachable[r, c]:
            continue
        if int(grid[r, c].item()) != background:
            continue
        reachable[r, c] = True
        stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])

    # Fill unreachable background with nearest non-background color
    out = grid.clone()
    for r in range(H):
        for c in range(W):
            if int(grid[r, c].item()) == background and not reachable[r, c]:
                # Find nearest non-background neighbor
                for radius in range(1, max(H, W)):
                    found = False
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            if abs(dr) != radius and abs(dc) != radius:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                val = int(grid[nr, nc].item())
                                if val != background:
                                    out[r, c] = val
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
    return out


def apply_sort_objects(grid: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Sort objects by their position along an axis and rearrange them.

    Objects are extracted, sorted by centroid position, and placed back
    in a grid with even spacing.
    """
    from soma_mythos_ehra.arc3.objects import extract_objects
    H, W = grid.shape
    objects = extract_objects(grid)
    if not objects:
        return grid

    # Sort by centroid along specified axis
    objects.sort(key=lambda o: o.centroid[axis])

    out = torch.zeros_like(grid)
    # Place objects with even spacing
    total_height = sum(o.bbox[1] - o.bbox[0] + 1 for o in objects)
    total_width = max(o.bbox[3] - o.bbox[2] + 1 for o in objects) if objects else 0

    if axis == 0:  # Sort vertically
        y_offset = max(0, (H - total_height) // (len(objects) + 1)) if objects else 0
        current_y = max(0, (H - total_height) // 2)
        for obj in objects:
            obj_h = obj.bbox[1] - obj.bbox[0] + 1
            obj_w = obj.bbox[3] - obj.bbox[2] + 1
            x_offset = max(0, (W - obj_w) // 2)
            for py, px in obj.pixels:
                new_y = current_y + (py - obj.bbox[0])
                new_x = x_offset + (px - obj.bbox[2])
                if 0 <= new_y < H and 0 <= new_x < W:
                    out[new_y, new_x] = grid[py, px]
            current_y += obj_h + max(1, (H - total_height) // (len(objects) + 1))
    else:  # Sort horizontally
        current_x = max(0, (W - total_width) // 2)
        for obj in objects:
            obj_h = obj.bbox[1] - obj.bbox[0] + 1
            obj_w = obj.bbox[3] - obj.bbox[2] + 1
            y_offset = max(0, (H - obj_h) // 2)
            for py, px in obj.pixels:
                new_y = y_offset + (py - obj.bbox[0])
                new_x = current_x + (px - obj.bbox[2])
                if 0 <= new_y < H and 0 <= new_x < W:
                    out[new_y, new_x] = grid[py, px]
            current_x += obj_w + max(1, (W - total_width) // (len(objects) + 1))

    return out


def apply_recolor_by_size(grid: torch.Tensor) -> torch.Tensor:
    """Recolor objects based on their size ordering.

    Largest object gets color 1, second largest gets color 2, etc.
    """
    from soma_mythos_ehra.arc3.objects import extract_objects
    objects = extract_objects(grid)
    if not objects:
        return grid

    # Sort by area (descending)
    objects.sort(key=lambda o: o.area, reverse=True)

    out = grid.clone()
    for i, obj in enumerate(objects):
        new_color = i + 1
        for py, px in obj.pixels:
            out[py, px] = new_color

    return out


def apply_sort_by_density(grid: torch.Tensor) -> torch.Tensor:
    """Sort objects by density (solidness) and rearrange them.

    Densest objects go to top-left, least dense to bottom-right.
    """
    from soma_mythos_ehra.arc3.objects import extract_objects
    H, W = grid.shape
    objects = extract_objects(grid)
    if not objects:
        return grid

    # Sort by density (descending)
    objects.sort(key=lambda o: o.density, reverse=True)

    out = torch.zeros_like(grid)
    current_y = 0
    for obj in objects:
        obj_h = obj.bbox[1] - obj.bbox[0] + 1
        obj_w = obj.bbox[3] - obj.bbox[2] + 1
        if current_y + obj_h > H:
            current_y = 0
        for py, px in obj.pixels:
            new_y = current_y + (py - obj.bbox[0])
            new_x = (px - obj.bbox[2])
            if 0 <= new_y < H and 0 <= new_x < W:
                out[new_y, new_x] = grid[py, px]
        current_y += obj_h + 1

    return out


def apply_sort_by_centroid(grid: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Sort objects by centroid position and rearrange them.

    axis=0: sort by row (top to bottom)
    axis=1: sort by column (left to right)
    """
    from soma_mythos_ehra.arc3.objects import extract_objects
    H, W = grid.shape
    objects = extract_objects(grid)
    if not objects:
        return grid

    # Sort by centroid along axis
    objects.sort(key=lambda o: o.centroid[axis])

    out = torch.zeros_like(grid)
    if axis == 0:  # Vertical sort
        current_y = 0
        for obj in objects:
            obj_h = obj.bbox[1] - obj.bbox[0] + 1
            obj_w = obj.bbox[3] - obj.bbox[2] + 1
            x_offset = max(0, (W - obj_w) // 2)
            for py, px in obj.pixels:
                new_y = current_y + (py - obj.bbox[0])
                new_x = x_offset + (px - obj.bbox[2])
                if 0 <= new_y < H and 0 <= new_x < W:
                    out[new_y, new_x] = grid[py, px]
            current_y += obj_h + 1
    else:  # Horizontal sort
        current_x = 0
        for obj in objects:
            obj_h = obj.bbox[1] - obj.bbox[0] + 1
            obj_w = obj.bbox[3] - obj.bbox[2] + 1
            y_offset = max(0, (H - obj_h) // 2)
            for py, px in obj.pixels:
                new_y = y_offset + (py - obj.bbox[0])
                new_x = current_x + (px - obj.bbox[2])
                if 0 <= new_y < H and 0 <= new_x < W:
                    out[new_y, new_x] = grid[py, px]
            current_x += obj_w + 1

    return out


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
    elif transform == TransformType.MOVE_OBJECT:
        obj_label = kwargs.get("obj_label", 1)
        dy = kwargs.get("dy", 0)
        dx = kwargs.get("dx", 0)
        return apply_move_object(grid, obj_label, dy, dx)
    elif transform == TransformType.FILL_HOLES:
        return apply_fill_holes(grid)
    elif transform == TransformType.SORT_OBJECTS:
        axis = kwargs.get("axis", 0)
        return apply_sort_objects(grid, axis)
    elif transform == TransformType.RECOLOR_BY_SIZE:
        return apply_recolor_by_size(grid)
    elif transform == TransformType.SORT_BY_DENSITY:
        return apply_sort_by_density(grid)
    elif transform == TransformType.SORT_BY_CENTROID:
        axis = kwargs.get("axis", 0)
        return apply_sort_by_centroid(grid, axis)
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
            cm = step["color_map"]
            # Special case: component labeling (compute labels dynamically)
            if step.get("type") == "component_labeling":
                from soma_mythos_ehra.arc3.objects import connected_component_labeling
                state = connected_component_labeling(state)
            # Special case: DSL program execution
            elif step.get("type") == "dsl" and "dsl_program" in step:
                from soma_mythos_ehra.arc3.dsl_kernel import DSLKernel
                kernel = DSLKernel(background=0)
                result = kernel.execute(step["dsl_program"], state)
                if result is not None:
                    state = result
            else:
                old = state.clone()
                for src, dst in cm.items():
                    state[old == src] = dst
            continue
        kwargs = {k: v for k, v in step.items() if k != "transform"}
        state = apply_transform(state, transform, **kwargs)
    return state
