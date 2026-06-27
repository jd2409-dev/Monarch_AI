"""ARC Template Library — 100+ pre-composed program templates.

Covers common ARC puzzle patterns including:
- Color transformations
- Spatial rearrangements
- Object manipulation
- Pattern generation
- Shape morphing
"""
from __future__ import annotations

from soma_mythos_ehra.arc3.dsl_grammar import DSLNode


def build_template_library() -> list[tuple[str, DSLNode]]:
    """Build a library of 100+ program templates.

    Returns list of (name, program_ast) tuples.
    """
    templates = []

    # === COLOR TRANSFORMATIONS (20 templates) ===

    # Simple recolor
    for src in range(10):
        for dst in range(10):
            if src != dst and len(templates) < 10:
                templates.append((
                    f"recolor_{src}_to_{dst}",
                    DSLNode(primitive="recolor", params={"src": src, "dst": dst}),
                ))

    # Recolor by size
    templates.append(("recolor_by_size", DSLNode(
        primitive="compose",
        children=[
            DSLNode(primitive="objects"),
            DSLNode(primitive="recolor_by_size"),
        ],
    )))

    # Recolor by density
    templates.append(("recolor_by_density", DSLNode(
        primitive="compose",
        children=[
            DSLNode(primitive="objects"),
            DSLNode(primitive="recolor_by_density"),
        ],
    )))

    # Component labeling
    templates.append(("component_labeling", DSLNode(
        primitive="compose",
        children=[
            DSLNode(primitive="objects"),
            DSLNode(primitive="recolor_by_size"),
        ],
    )))

    # Fill holes
    templates.append(("fill_holes", DSLNode(primitive="fill_holes")))

    # Invert colors
    for max_color in [3, 5, 9]:
        templates.append((
            f"invert_colors_{max_color}",
            DSLNode(primitive="compose", children=[
                DSLNode(primitive="recolor", params={"src": 0, "dst": max_color}),
                DSLNode(primitive="recolor", params={"src": 1, "dst": max_color - 1}),
            ]),
        ))

    # === SPATIAL TRANSFORMATIONS (25 templates) ===

    # Rotations
    for angle in [90, 180, 270]:
        templates.append((
            f"rotate_{angle}",
            DSLNode(primitive="rotate", params={"angle": angle}),
        ))

    # Flips
    for axis in ["h", "v"]:
        templates.append((
            f"flip_{axis}",
            DSLNode(primitive="flip", params={"axis": axis}),
        ))

    # Transpose
    templates.append(("transpose", DSLNode(primitive="transpose")))

    # Rotate + recolor
    for angle in [90, 180, 270]:
        templates.append((
            f"rotate_{angle}_recolor_by_size",
            DSLNode(primitive="compose", children=[
                DSLNode(primitive="rotate", params={"angle": angle}),
                DSLNode(primitive="objects"),
                DSLNode(primitive="recolor_by_size"),
            ]),
        ))

    # Flip + recolor
    for axis in ["h", "v"]:
        templates.append((
            f"flip_{axis}_recolor_by_size",
            DSLNode(primitive="compose", children=[
                DSLNode(primitive="flip", params={"axis": axis}),
                DSLNode(primitive="objects"),
                DSLNode(primitive="recolor_by_size"),
            ]),
        ))

    # Transpose + recolor
    templates.append(("transpose_recolor_by_size", DSLNode(
        primitive="compose",
        children=[
            DSLNode(primitive="transpose"),
            DSLNode(primitive="objects"),
            DSLNode(primitive="recolor_by_size"),
        ],
    )))

    # === SCALING AND TILING (20 templates) ===

    # Scale up
    for factor in [2, 3, 4]:
        templates.append((
            f"scale_{factor}",
            DSLNode(primitive="scale", params={"factor": factor}),
        ))

    # Tile
    for r in [2, 3, 4]:
        templates.append((
            f"tile_{r}x{r}",
            DSLNode(primitive="tile", params={"reps_h": r, "reps_w": r}),
        ))

    # Scale + recolor
    for factor in [2, 3]:
        templates.append((
            f"scale_{factor}_recolor_by_size",
            DSLNode(primitive="compose", children=[
                DSLNode(primitive="scale", params={"factor": factor}),
                DSLNode(primitive="objects"),
                DSLNode(primitive="recolor_by_size"),
            ]),
        ))

    # Tile + recolor
    for r in [2, 3]:
        templates.append((
            f"tile_{r}x{r}_recolor_by_size",
            DSLNode(primitive="compose", children=[
                DSLNode(primitive="tile", params={"reps_h": r, "reps_w": r}),
                DSLNode(primitive="objects"),
                DSLNode(primitive="recolor_by_size"),
            ]),
        ))

    # === OBJECT MANIPULATION (25 templates) ===

    # Filter by area + recolor
    for mode in ["max", "min"]:
        for color in range(1, 6):
            templates.append((
                f"filter_area_{mode}_recolor_{color}",
                DSLNode(primitive="apply_to_objects", children=[
                    DSLNode(primitive="filter_by_area", params={"mode": mode}),
                    DSLNode(primitive="recolor_objects", params={"color": color}),
                ]),
            ))

    # Filter by density + recolor
    for mode in ["max", "min"]:
        for color in range(1, 4):
            templates.append((
                f"filter_density_{mode}_recolor_{color}",
                DSLNode(primitive="apply_to_objects", children=[
                    DSLNode(primitive="filter_by_density", params={"mode": mode}),
                    DSLNode(primitive="recolor_objects", params={"color": color}),
                ]),
            ))

    # Sort by position + recolor by size
    for axis in [0, 1]:
        templates.append((
            f"sort_pos_{axis}_recolor_by_size",
            DSLNode(primitive="compose", children=[
                DSLNode(primitive="objects"),
                DSLNode(primitive="sort_by_position", params={"axis": axis}),
                DSLNode(primitive="recolor_by_size"),
            ]),
        ))

    # Sort by area + recolor by size
    for reverse in [True, False]:
        templates.append((
            f"sort_area_{reverse}_recolor_by_size",
            DSLNode(primitive="compose", children=[
                DSLNode(primitive="objects"),
                DSLNode(primitive="sort_by_area", params={"reverse": reverse}),
                DSLNode(primitive="recolor_by_size"),
            ]),
        ))

    # Filter by color + recolor
    for color in range(1, 5):
        templates.append((
            f"filter_color_{color}_recolor_by_size",
            DSLNode(primitive="compose", children=[
                DSLNode(primitive="filter_by_color", params={"color": color}),
                DSLNode(primitive="recolor_by_size"),
            ]),
        ))

    # Filter by size + recolor
    for mode in ["largest", "smallest"]:
        for color in range(1, 4):
            templates.append((
                f"filter_size_{mode}_recolor_{color}",
                DSLNode(primitive="apply_to_objects", children=[
                    DSLNode(primitive="filter_by_size", params={"mode": mode}),
                    DSLNode(primitive="recolor_objects", params={"color": color}),
                ]),
            ))

    # Filter by position + recolor
    for region in ["top", "bottom", "left", "right", "center"]:
        for color in range(1, 4):
            templates.append((
                f"filter_pos_{region}_recolor_{color}",
                DSLNode(primitive="apply_to_objects", children=[
                    DSLNode(primitive="filter_by_position", params={"region": region}),
                    DSLNode(primitive="recolor_objects", params={"color": color}),
                ]),
            ))

    # === COMPLEX COMPOSITIONS (25 templates) ===

    # Rotate + filter + recolor
    for angle in [90, 180, 270]:
        for mode in ["max", "min"]:
            templates.append((
                f"rotate_{angle}_filter_area_{mode}",
                DSLNode(primitive="compose", children=[
                    DSLNode(primitive="rotate", params={"angle": angle}),
                    DSLNode(primitive="objects"),
                    DSLNode(primitive="filter_by_area", params={"mode": mode}),
                    DSLNode(primitive="recolor_by_size"),
                ]),
            ))

    # Flip + filter + recolor
    for axis in ["h", "v"]:
        for mode in ["max", "min"]:
            templates.append((
                f"flip_{axis}_filter_area_{mode}",
                DSLNode(primitive="compose", children=[
                    DSLNode(primitive="flip", params={"axis": axis}),
                    DSLNode(primitive="objects"),
                    DSLNode(primitive="filter_by_area", params={"mode": mode}),
                    DSLNode(primitive="recolor_by_size"),
                ]),
            ))

    # Scale + filter + recolor
    for factor in [2, 3]:
        templates.append((
            f"scale_{factor}_filter_area_max",
            DSLNode(primitive="compose", children=[
                DSLNode(primitive="scale", params={"factor": factor}),
                DSLNode(primitive="objects"),
                DSLNode(primitive="filter_by_area", params={"mode": "max"}),
                DSLNode(primitive="recolor_by_size"),
            ]),
        ))

    # Tile + filter + recolor
    for r in [2, 3]:
        templates.append((
            f"tile_{r}x{r}_filter_area_max",
            DSLNode(primitive="compose", children=[
                DSLNode(primitive="tile", params={"reps_h": r, "reps_w": r}),
                DSLNode(primitive="objects"),
                DSLNode(primitive="filter_by_area", params={"mode": "max"}),
                DSLNode(primitive="recolor_by_size"),
            ]),
        ))

    # Fill holes + recolor
    templates.append(("fill_holes_recolor_by_size", DSLNode(
        primitive="compose",
        children=[
            DSLNode(primitive="fill_holes"),
            DSLNode(primitive="objects"),
            DSLNode(primitive="recolor_by_size"),
        ],
    )))

    # Rotate + fill holes
    for angle in [90, 180, 270]:
        templates.append((
            f"rotate_{angle}_fill_holes",
            DSLNode(primitive="compose", children=[
                DSLNode(primitive="rotate", params={"angle": angle}),
                DSLNode(primitive="fill_holes"),
            ]),
        ))

    # Flip + fill holes
    for axis in ["h", "v"]:
        templates.append((
            f"flip_{axis}_fill_holes",
            DSLNode(primitive="compose", children=[
                DSLNode(primitive="flip", params={"axis": axis}),
                DSLNode(primitive="fill_holes"),
            ]),
        ))

    return templates
