"""Expanded Grammar — spatial relations, tessellation, mirror, and scene graph primitives.

Extends the base recursive grammar with higher-level structural operations
needed to express the complex spatial compositions in ARC puzzles.
"""
from __future__ import annotations

import random
from soma_mythos_ehra.arc3.recursive_grammar import (
    ALL_PRIMITIVES,
    ASTNode,
    NodeType,
    PRIMITIVE_MAP,
)


# ---------------------------------------------------------------------------
# New Higher-Level Primitives
# ---------------------------------------------------------------------------

# Spatial relational primitives
SPATIAL_RELATIONS = [
    "mirror_h",           # Mirror grid horizontally across center axis
    "mirror_v",           # Mirror grid vertically across center axis
    "mirror_diag",        # Mirror across main diagonal
    "tessellate_2x2",     # Tile input into 2x2 pattern
    "tessellate_3x3",     # Tile input into 3x3 pattern
    "quadrant_fill",      # Fill each quadrant with a transformed version
    "border_frame",       # Add a border of background around the grid
    "center_crop",        # Crop to center region
    "diagonal_copy",      # Copy content along diagonal
    "row_repeat",         # Repeat each row N times
    "col_repeat",         # Repeat each column N times
    "invert_mask",        # Invert non-zero and zero regions
    "object_shift_up",    # Shift all objects up by 1
    "object_shift_down",  # Shift all objects down by 1
    "object_shift_left",  # Shift all objects left by 1
    "object_shift_right", # Shift all objects right by 1
    "grow_objects",       # Dilate all objects by 1 pixel
    "shrink_objects",     # Erode all objects by 1 pixel
    "fill_interior",      # Fill interior of each object
    "outline_objects",    # Keep only border pixels of objects
]

# Combine with base primitives
EXTENDED_PRIMITIVES = ALL_PRIMITIVES + SPATIAL_RELATIONS

# Extended filter set
EXTENDED_FILTERS = [
    "all", "by_area_max", "by_area_min",
    "by_density_solid", "by_density_hollow",
    "by_color_1", "by_color_2", "by_color_3",
    "by_position_top", "by_position_bottom",
    "by_position_left", "by_position_right",
    "by_position_center",
]

# Extended condition set
EXTENDED_CONDITIONS = [
    "is_largest", "is_smallest", "is_solid", "is_hollow",
    "has_area_gt_5", "has_area_lt_5",
    "has_objects_above_3", "has_objects_below_3",
    "grid_is_square", "grid_is_wide", "grid_is_tall",
]

# Extended composition operators
EXTENDED_COMPOSITIONS = [
    "compose", "apply_to_objects", "branch",
    "map_each_object",    # Apply transform to each object independently
    "sequence",           # Execute list of transforms sequentially
]

# Full extended vocabulary
EXTENDED_TOKEN_VOCAB = (
    EXTENDED_PRIMITIVES +
    EXTENDED_FILTERS +
    EXTENDED_CONDITIONS +
    EXTENDED_COMPOSITIONS
)

EXTENDED_TOKEN_TO_IDX = {tok: i for i, tok in enumerate(EXTENDED_TOKEN_VOCAB)}
EXTENDED_NUM_TOKENS = len(EXTENDED_TOKEN_VOCAB)


# ---------------------------------------------------------------------------
# Execution Functions for New Primitives
# ---------------------------------------------------------------------------

def execute_mirror_h(grid):
    """Mirror grid horizontally (left-right flip)."""
    return grid.flip(1)

def execute_mirror_v(grid):
    """Mirror grid vertically (top-bottom flip)."""
    return grid.flip(0)

def execute_mirror_diag(grid):
    """Mirror across main diagonal (transpose)."""
    return grid.t()

def execute_tessellate_2x2(grid):
    """Tile into 2x2 pattern."""
    return grid.repeat(2, 2)

def execute_tessellate_3x3(grid):
    """Tile into 3x3 pattern."""
    return grid.repeat(3, 3)

def execute_quadrant_fill(grid):
    """Fill each quadrant: top-left=original, others=rotated versions."""
    H, W = grid.shape
    h, w = H // 2, W // 2
    if h == 0 or w == 0:
        return grid
    q = grid[:h, :w]
    out = grid.clone()
    out[:h, w:] = torch.rot90(q, -1)    # top-right: rotate 90
    out[h:, :w] = torch.rot90(q, 1)     # bottom-left: rotate 270
    out[h:, w:] = torch.rot90(q, 2)     # bottom-right: rotate 180
    return out

def execute_border_frame(grid):
    """Add 1-pixel border of background around the grid."""
    H, W = grid.shape
    out = torch.zeros(H + 2, W + 2, dtype=grid.dtype)
    out[1:H+1, 1:W+1] = grid
    return out

def execute_center_crop(grid):
    """Crop to center 50% region."""
    H, W = grid.shape
    h, w = H // 4, W // 4
    return grid[h:H-h, w:W-w] if h > 0 and w > 0 else grid

def execute_diagonal_copy(grid):
    """Copy grid content along the diagonal."""
    H, W = grid.shape
    out = grid.clone()
    S = min(H, W)
    for i in range(S):
        for j in range(S):
            if i != j:
                out[i, j] = grid[i, i] if i < H and i < W else out[i, j]
    return out

def execute_row_repeat(grid):
    """Repeat each row 2 times."""
    return grid.repeat_interleave(2, dim=0)

def execute_col_repeat(grid):
    """Repeat each column 2 times."""
    return grid.repeat_interleave(2, dim=1)

def execute_invert_mask(grid):
    """Invert: zero becomes max+1, non-zero becomes 0."""
    out = grid.clone()
    max_val = int(grid.max().item()) + 1
    out[grid == 0] = max_val
    out[grid != 0] = 0
    return out

def execute_object_shift_up(grid):
    """Shift all non-background pixels up by 1."""
    out = torch.zeros_like(grid)
    H, W = grid.shape
    for r in range(1, H):
        for c in range(W):
            if grid[r, c] != 0:
                out[r-1, c] = grid[r, c]
    return out

def execute_object_shift_down(grid):
    """Shift all non-background pixels down by 1."""
    out = torch.zeros_like(grid)
    H, W = grid.shape
    for r in range(H-1):
        for c in range(W):
            if grid[r, c] != 0:
                out[r+1, c] = grid[r, c]
    return out

def execute_object_shift_left(grid):
    """Shift all non-background pixels left by 1."""
    out = torch.zeros_like(grid)
    H, W = grid.shape
    for r in range(H):
        for c in range(1, W):
            if grid[r, c] != 0:
                out[r, c-1] = grid[r, c]
    return out

def execute_object_shift_right(grid):
    """Shift all non-background pixels right by 1."""
    out = torch.zeros_like(grid)
    H, W = grid.shape
    for r in range(H):
        for c in range(W-1):
            if grid[r, c] != 0:
                out[r, c+1] = grid[r, c]
    return out

def execute_grow_objects(grid):
    """Dilate: expand each non-background pixel to its 4-neighbors."""
    out = grid.clone()
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 0:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and out[nr, nc] == 0:
                        out[nr, nc] = grid[r, c]
    return out

def execute_shrink_objects(grid):
    """Erode: remove pixels that have any background neighbor."""
    out = grid.clone()
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 0:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if nr < 0 or nr >= H or nc < 0 or nc >= W or grid[nr, nc] == 0:
                        out[r, c] = 0
                        break
    return out

def execute_fill_interior(grid):
    """Fill interior holes of each connected component."""
    from soma_mythos_ehra.arc3.transforms import apply_fill_holes
    return apply_fill_holes(grid)

def execute_outline_objects(grid):
    """Keep only border pixels of objects (remove interior)."""
    out = torch.zeros_like(grid)
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 0:
                is_border = False
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if nr < 0 or nr >= H or nc < 0 or nc >= W or grid[nr, nc] == 0:
                        is_border = True
                        break
                if is_border:
                    out[r, c] = grid[r, c]
    return out


# Map new primitives to execution functions
EXTENDED_EXECUTORS = {
    "mirror_h": execute_mirror_h,
    "mirror_v": execute_mirror_v,
    "mirror_diag": execute_mirror_diag,
    "tessellate_2x2": execute_tessellate_2x2,
    "tessellate_3x3": execute_tessellate_3x3,
    "quadrant_fill": execute_quadrant_fill,
    "border_frame": execute_border_frame,
    "center_crop": execute_center_crop,
    "diagonal_copy": execute_diagonal_copy,
    "row_repeat": execute_row_repeat,
    "col_repeat": execute_col_repeat,
    "invert_mask": execute_invert_mask,
    "object_shift_up": execute_object_shift_up,
    "object_shift_down": execute_object_shift_down,
    "object_shift_left": execute_object_shift_left,
    "object_shift_right": execute_object_shift_right,
    "grow_objects": execute_grow_objects,
    "shrink_objects": execute_shrink_objects,
    "fill_interior": execute_fill_interior,
    "outline_objects": execute_outline_objects,
}


# ---------------------------------------------------------------------------
# Extended Random Sampler
# ---------------------------------------------------------------------------

def sample_extended_program(max_depth: int = 3, depth: int = 0) -> ASTNode:
    """Sample a random program from the extended grammar."""
    if depth >= max_depth or (depth > 0 and random.random() < 0.35):
        # Leaf: primitive from extended set
        name = random.choice(EXTENDED_PRIMITIVES)
        if name in PRIMITIVE_MAP:
            prim, params = PRIMITIVE_MAP[name]
            if prim == "recolor":
                params = {"src": random.randint(1, 5), "dst": random.randint(1, 5)}
            elif prim == "flood_fill":
                params = {"y": random.randint(0, 7), "x": random.randint(0, 7), "color": random.randint(1, 5)}
            elif prim == "shift":
                params = {"dy": random.choice([-2, -1, 1, 2]), "dx": random.choice([-2, -1, 1, 2])}
            elif prim == "wrap":
                params = {"shift": random.randint(1, 3), "axis": random.choice([0, 1])}
            return ASTNode(NodeType.PRIMITIVE, name=prim, params=params)
        else:
            return ASTNode(NodeType.PRIMITIVE, name=name, params={})

    choice = random.choices(
        ["primitive", "compose", "apply_to_objects", "branch", "map_each_object"],
        weights=[2, 3, 3, 2, 1],
        k=1,
    )[0]

    if choice == "primitive":
        name = random.choice(EXTENDED_PRIMITIVES)
        if name in PRIMITIVE_MAP:
            prim, params = PRIMITIVE_MAP[name]
            if prim == "recolor":
                params = {"src": random.randint(1, 5), "dst": random.randint(1, 5)}
            return ASTNode(NodeType.PRIMITIVE, name=prim, params=params)
        return ASTNode(NodeType.PRIMITIVE, name=name, params={})

    elif choice == "compose":
        left = sample_extended_program(max_depth, depth + 1)
        right = sample_extended_program(max_depth, depth + 1)
        return ASTNode(NodeType.COMPOSE, name="compose", children=[left, right])

    elif choice == "apply_to_objects":
        filt = random.choice(EXTENDED_FILTERS)
        action = sample_extended_program(max_depth, depth + 1)
        return ASTNode(NodeType.APPLY_TO_OBJECTS, name="apply_to_objects",
                       children=[ASTNode(NodeType.FILTER, name=filt), action])

    elif choice == "branch":
        cond = random.choice(EXTENDED_CONDITIONS)
        true_branch = sample_extended_program(max_depth, depth + 1)
        false_branch = sample_extended_program(max_depth, depth + 1)
        return ASTNode(NodeType.BRANCH, name="branch",
                       children=[ASTNode(NodeType.CONDITION, name=cond),
                                 true_branch, false_branch])

    else:  # map_each_object
        action = sample_extended_program(max_depth, depth + 1)
        return ASTNode(NodeType.APPLY_TO_OBJECTS, name="map_each_object",
                       children=[ASTNode(NodeType.FILTER, name="all"), action])
