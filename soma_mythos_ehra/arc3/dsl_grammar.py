"""ARC Domain-Specific Language (DSL) for Program Synthesis.

Defines atomic primitives and a grammar for composing transformation programs.
MCTS searches over program trees built from these primitives to solve ARC puzzles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import torch


# ---------------------------------------------------------------------------
# DSL Primitive Types
# ---------------------------------------------------------------------------

class PrimitiveType(IntEnum):
    """Categories of DSL primitives."""
    OBJECT = 0       # Object extraction and querying
    FILTER = 1       # Filtering objects by attributes
    TRANSFORM = 2    # Grid transformations
    COLOR = 3        # Color operations
    SPATIAL = 4      # Spatial relationships
    COMPOSITION = 5  # Program composition


@dataclass
class DSLPrimitive:
    """A single DSL primitive with its signature."""
    name: str
    ptype: PrimitiveType
    params: dict[str, Any] = field(default_factory=dict)
    input_type: str = "grid"     # "grid", "objects", "mask"
    output_type: str = "grid"    # "grid", "objects", "mask", "int", "bool"
    description: str = ""


# ---------------------------------------------------------------------------
# Core Grammar: 30 Atomic Primitives
# ---------------------------------------------------------------------------

PRIMITIVES = {
    # === OBJECT PRIMITIVES ===
    "objects": DSLPrimitive(
        name="objects",
        ptype=PrimitiveType.OBJECT,
        input_type="grid",
        output_type="objects",
        description="Extract all connected components from grid",
    ),
    "objects_of_color": DSLPrimitive(
        name="objects_of_color",
        ptype=PrimitiveType.OBJECT,
        params={"color": "int"},
        input_type="grid",
        output_type="objects",
        description="Extract objects with specific color",
    ),
    "largest_object": DSLPrimitive(
        name="largest_object",
        ptype=PrimitiveType.OBJECT,
        input_type="objects",
        output_type="object",
        description="Select object with maximum area",
    ),
    "smallest_object": DSLPrimitive(
        name="smallest_object",
        ptype=PrimitiveType.OBJECT,
        input_type="objects",
        output_type="object",
        description="Select object with minimum area",
    ),
    "object_at": DSLPrimitive(
        name="object_at",
        ptype=PrimitiveType.OBJECT,
        params={"position": "str"},  # "top_left", "bottom_right", "center"
        input_type="objects",
        output_type="object",
        description="Select object by spatial position",
    ),

    # === FILTER PRIMITIVES ===
    "filter_by_area": DSLPrimitive(
        name="filter_by_area",
        ptype=PrimitiveType.FILTER,
        params={"mode": "str"},  # "max", "min", "range"
        input_type="objects",
        output_type="objects",
        description="Filter objects by area attribute",
    ),
    "filter_by_density": DSLPrimitive(
        name="filter_by_density",
        ptype=PrimitiveType.FILTER,
        params={"mode": "str"},
        input_type="objects",
        output_type="objects",
        description="Filter objects by density (solidness)",
    ),
    "filter_by_color": DSLPrimitive(
        name="filter_by_color",
        ptype=PrimitiveType.FILTER,
        params={"color": "int"},
        input_type="objects",
        output_type="objects",
        description="Filter objects by color",
    ),
    "filter_by_size": DSLPrimitive(
        name="filter_by_size",
        ptype=PrimitiveType.FILTER,
        params={"mode": "str"},  # "largest", "smallest"
        input_type="objects",
        output_type="objects",
        description="Filter objects by relative size",
    ),
    "filter_by_position": DSLPrimitive(
        name="filter_by_position",
        ptype=PrimitiveType.FILTER,
        params={"region": "str"},  # "top", "bottom", "left", "right", "center"
        input_type="objects",
        output_type="objects",
        description="Filter objects by spatial region",
    ),
    "take_n": DSLPrimitive(
        name="take_n",
        ptype=PrimitiveType.FILTER,
        params={"n": "int", "order": "str"},  # "first", "last"
        input_type="objects",
        output_type="objects",
        description="Take first or last N objects",
    ),

    # === TRANSFORM PRIMITIVES ===
    "rotate": DSLPrimitive(
        name="rotate",
        ptype=PrimitiveType.TRANSFORM,
        params={"angle": "int"},  # 90, 180, 270
        input_type="grid",
        output_type="grid",
        description="Rotate grid by angle degrees",
    ),
    "flip": DSLPrimitive(
        name="flip",
        ptype=PrimitiveType.TRANSFORM,
        params={"axis": "str"},  # "h", "v"
        input_type="grid",
        output_type="grid",
        description="Flip grid horizontally or vertically",
    ),
    "transpose": DSLPrimitive(
        name="transpose",
        ptype=PrimitiveType.TRANSFORM,
        input_type="grid",
        output_type="grid",
        description="Transpose grid (swap rows/cols)",
    ),
    "crop": DSLPrimitive(
        name="crop",
        ptype=PrimitiveType.TRANSFORM,
        params={"region": "str"},  # "top_left", "center", "bbox"
        input_type="grid",
        output_type="grid",
        description="Crop grid to region",
    ),
    "pad": DSLPrimitive(
        name="pad",
        ptype=PrimitiveType.TRANSFORM,
        params={"size": "int", "color": "int"},
        input_type="grid",
        output_type="grid",
        description="Pad grid with background color",
    ),
    "scale": DSLPrimitive(
        name="scale",
        ptype=PrimitiveType.TRANSFORM,
        params={"factor": "int"},
        input_type="grid",
        output_type="grid",
        description="Scale grid by integer factor",
    ),
    "tile": DSLPrimitive(
        name="tile",
        ptype=PrimitiveType.TRANSFORM,
        params={"reps_h": "int", "reps_w": "int"},
        input_type="grid",
        output_type="grid",
        description="Tile grid pattern",
    ),
    "shift": DSLPrimitive(
        name="shift",
        ptype=PrimitiveType.TRANSFORM,
        params={"dy": "int", "dx": "int"},
        input_type="grid",
        output_type="grid",
        description="Shift all objects by offset",
    ),
    "wrap": DSLPrimitive(
        name="wrap",
        ptype=PrimitiveType.TRANSFORM,
        params={"shift": "int", "axis": "int"},
        input_type="grid",
        output_type="grid",
        description="Cyclic shift of rows/cols",
    ),

    # === COLOR PRIMITIVES ===
    "recolor": DSLPrimitive(
        name="recolor",
        ptype=PrimitiveType.COLOR,
        params={"src": "int", "dst": "int"},
        input_type="grid",
        output_type="grid",
        description="Remap one color to another",
    ),
    "recolor_objects": DSLPrimitive(
        name="recolor_objects",
        ptype=PrimitiveType.COLOR,
        params={"color": "int"},
        input_type="objects",
        output_type="grid",
        description="Recolor all objects to same color",
    ),
    "recolor_by_size": DSLPrimitive(
        name="recolor_by_size",
        ptype=PrimitiveType.COLOR,
        input_type="objects",
        output_type="grid",
        description="Recolor objects by size ordering",
    ),
    "recolor_by_density": DSLPrimitive(
        name="recolor_by_density",
        ptype=PrimitiveType.COLOR,
        input_type="objects",
        output_type="grid",
        description="Recolor objects by density ordering",
    ),
    "fill_holes": DSLPrimitive(
        name="fill_holes",
        ptype=PrimitiveType.COLOR,
        input_type="grid",
        output_type="grid",
        description="Fill interior holes in objects",
    ),
    "flood_fill": DSLPrimitive(
        name="flood_fill",
        ptype=PrimitiveType.COLOR,
        params={"y": "int", "x": "int", "color": "int"},
        input_type="grid",
        output_type="grid",
        description="Flood fill from position",
    ),

    # === SPATIAL PRIMITIVES ===
    "sort_by_position": DSLPrimitive(
        name="sort_by_position",
        ptype=PrimitiveType.SPATIAL,
        params={"axis": "int"},
        input_type="objects",
        output_type="objects",
        description="Sort objects by centroid position",
    ),
    "sort_by_area": DSLPrimitive(
        name="sort_by_area",
        ptype=PrimitiveType.SPATIAL,
        params={"reverse": "bool"},
        input_type="objects",
        output_type="objects",
        description="Sort objects by area",
    ),
    "distance_between": DSLPrimitive(
        name="distance_between",
        ptype=PrimitiveType.SPATIAL,
        input_type="objects",
        output_type="int",
        description="Compute distance between two objects",
    ),

    # === COMPOSITION PRIMITIVES ===
    "compose": DSLPrimitive(
        name="compose",
        ptype=PrimitiveType.COMPOSITION,
        params={"programs": "list"},
        description="Compose multiple programs sequentially",
    ),
    "branch": DSLPrimitive(
        name="branch",
        ptype=PrimitiveType.COMPOSITION,
        params={"condition": "str", "true_prog": "program", "false_prog": "program"},
        description="Conditional branching",
    ),
    "apply_to_objects": DSLPrimitive(
        name="apply_to_objects",
        ptype=PrimitiveType.COMPOSITION,
        params={"filter_prog": "program", "transform_prog": "program"},
        description="Apply transform to filtered objects",
    ),
}


# ---------------------------------------------------------------------------
# DSL Program AST
# ---------------------------------------------------------------------------

@dataclass
class DSLNode:
    """A node in a DSL program AST."""
    primitive: str
    children: list["DSLNode"] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        if self.children:
            child_strs = ", ".join(c.to_string(indent + 1) for c in self.children)
            return f"{prefix}{self.primitive}({child_strs}, {self.params})"
        return f"{prefix}{self.primitive}({self.params})"

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)


# ---------------------------------------------------------------------------
# Grammar Rules for MCTS Program Generation
# ---------------------------------------------------------------------------

GRAMMAR_RULES = {
    # Start: a program is a composition of transforms
    "program": ["transform", "compose(transform, transform)", "apply_to_objects(filter, transform)"],

    # Transform can be applied to grid or objects
    "transform": [
        "rotate", "flip", "transpose", "crop", "pad", "scale", "tile",
        "shift", "wrap", "recolor", "fill_holes", "flood_fill",
        "recolor_objects", "recolor_by_size", "recolor_by_density",
        "apply_to_objects(filter, color_transform)",
    ],

    # Filter can select objects
    "filter": [
        "objects", "objects_of_color", "filter_by_area", "filter_by_density",
        "filter_by_color", "filter_by_size", "filter_by_position",
        "largest_object", "smallest_object", "take_n",
    ],

    # Color transforms work on objects
    "color_transform": [
        "recolor", "recolor_objects", "recolor_by_size", "recolor_by_density",
    ],
}


def get_legal_expansions(node: DSLNode) -> list[str]:
    """Get legal primitive expansions for a DSL node during MCTS."""
    if not node.children:
        # Leaf node: expand based on primitive type
        prim = PRIMITIVES.get(node.primitive)
        if prim is None:
            return list(PRIMITIVES.keys())
        if prim.output_type == "objects":
            return [p for p, v in PRIMITIVES.items() if v.input_type == "objects"]
        if prim.output_type == "grid":
            return [p for p, v in PRIMITIVES.items() if v.input_type == "grid"]
        return list(PRIMITIVES.keys())
    # Non-leaf: expand last child
    return get_legal_expansions(node.children[-1])
