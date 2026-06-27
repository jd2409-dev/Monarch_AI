"""Recursive Grammar AST — generates random valid DSL programs for synthetic training.

Defines a context-free grammar with primitives, compositions, and conditionals.
Programs are sampled randomly and executed on synthetic grids to produce
(input, output) pairs with multi-hot target labels for JEPA training.

BNF Grammar:
  <Program>     ::= <Expression>
  <Expression>  ::= <Primitive> | <Composition> | <Conditional>
  <Primitive>   ::= rotate | flip | transpose | scale | recolor | fill_holes | shift | ...
  <Composition> ::= compose(<Expr>, <Expr>) | apply_to_objects(<Filter>, <Expr>)
  <Conditional> ::= branch(<Condition>, <Expr>, <Expr>)
  <Filter>      ::= all | by_area(max|min) | by_density(hollow|solid) | by_color(Int)
  <Condition>   ::= has_size(greater_than|less_than, Int) | is_color(Int) | is_hollow | is_largest
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# AST Node Types
# ---------------------------------------------------------------------------

class NodeType(Enum):
    PRIMITIVE = auto()
    COMPOSE = auto()
    APPLY_TO_OBJECTS = auto()
    BRANCH = auto()
    FILTER = auto()
    CONDITION = auto()


# Available primitives grouped by category
GRID_TRANSFORMS = ["rotate_90", "rotate_180", "rotate_270", "flip_h", "flip_v", "transpose", "scale_2", "scale_3"]
COLOR_OPS = ["recolor_map", "fill_holes", "flood_fill"]
SPATIAL_OPS = ["shift_down", "shift_up", "shift_left", "shift_right", "wrap_h", "wrap_v"]
ALL_PRIMITIVES = GRID_TRANSFORMS + COLOR_OPS + SPATIAL_OPS

FILTERS = ["all", "by_area_max", "by_area_min", "by_density_solid", "by_density_hollow"]
CONDITIONS = ["is_largest", "is_smallest", "is_solid", "is_hollow", "has_area_gt_5", "has_area_lt_5"]

# Map primitive names to DSL kernel primitives for execution
PRIMITIVE_MAP = {
    "rotate_90": ("rotate", {"angle": 90}),
    "rotate_180": ("rotate", {"angle": 180}),
    "rotate_270": ("rotate", {"angle": 270}),
    "flip_h": ("flip", {"axis": "h"}),
    "flip_v": ("flip", {"axis": "v"}),
    "transpose": ("transpose", {}),
    "scale_2": ("scale", {"factor": 2}),
    "scale_3": ("scale", {"factor": 3}),
    "recolor_map": ("recolor", {"src": 1, "dst": 2}),
    "fill_holes": ("fill_holes", {}),
    "flood_fill": ("flood_fill", {"y": 0, "x": 0, "color": 3}),
    "shift_down": ("shift", {"dy": 1, "dx": 0}),
    "shift_up": ("shift", {"dy": -1, "dx": 0}),
    "shift_left": ("shift", {"dy": 0, "dx": -1}),
    "shift_right": ("shift", {"dy": 0, "dx": 1}),
    "wrap_h": ("wrap", {"shift": 1, "axis": 1}),
    "wrap_v": ("wrap", {"shift": 1, "axis": 0}),
}

# Map filter names to DSL filter operations
FILTER_MAP = {
    "all": ("filter_by_area", {"mode": "max"}),  # passthrough: keep all
    "by_area_max": ("filter_by_area", {"mode": "max"}),
    "by_area_min": ("filter_by_area", {"mode": "min"}),
    "by_density_solid": ("filter_by_density", {"mode": "max"}),
    "by_density_hollow": ("filter_by_density", {"mode": "min"}),
}

# Map condition names to DSL kernel checks
CONDITION_MAP = {
    "is_largest": "area_max",
    "is_smallest": "area_min",
    "is_solid": "density_max",
    "is_hollow": "density_min",
    "has_area_gt_5": "area_gt_5",
    "has_area_lt_5": "area_lt_5",
}


# ---------------------------------------------------------------------------
# AST Node Classes
# ---------------------------------------------------------------------------

@dataclass
class ASTNode:
    """Base node for recursive grammar AST."""
    node_type: NodeType
    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    children: list[ASTNode] = field(default_factory=list)

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        if self.children:
            child_strs = ", ".join(c.to_string(indent + 1) for c in self.children)
            return f"{prefix}{self.name}({child_strs}, {self.params})"
        return f"{prefix}{self.name}({self.params})"


# ---------------------------------------------------------------------------
# Multi-Hot Token Encoder
# ---------------------------------------------------------------------------

# All tokens in the grammar (used for multi-hot encoding)
TOKEN_VOCAB = (
    ALL_PRIMITIVES +
    FILTERS +
    CONDITIONS +
    ["compose", "apply_to_objects", "branch"]
)

TOKEN_TO_IDX = {tok: i for i, tok in enumerate(TOKEN_VOCAB)}
NUM_TOKENS = len(TOKEN_VOCAB)


def encode_multi_hot(node: ASTNode) -> list[float]:
    """Flatten an AST into a multi-hot binary vector over all grammar tokens."""
    hits = [0.0] * NUM_TOKENS

    def _walk(n: ASTNode):
        if n.name in TOKEN_TO_IDX:
            hits[TOKEN_TO_IDX[n.name]] = 1.0
        if n.node_type == NodeType.COMPOSE:
            hits[TOKEN_TO_IDX["compose"]] = 1.0
        elif n.node_type == NodeType.APPLY_TO_OBJECTS:
            hits[TOKEN_TO_IDX["apply_to_objects"]] = 1.0
        elif n.node_type == NodeType.BRANCH:
            hits[TOKEN_TO_IDX["branch"]] = 1.0
        for c in n.children:
            _walk(c)

    _walk(node)
    return hits


# ---------------------------------------------------------------------------
# Random Program Sampler
# ---------------------------------------------------------------------------

def _sample_primitive() -> ASTNode:
    """Sample a random leaf primitive."""
    name = random.choice(ALL_PRIMITIVES)
    prim, params = PRIMITIVE_MAP[name]
    # Randomize color params
    if prim == "recolor":
        params = {"src": random.randint(1, 5), "dst": random.randint(1, 5)}
    elif prim == "flood_fill":
        params = {"y": random.randint(0, 7), "x": random.randint(0, 7), "color": random.randint(1, 5)}
    elif prim == "shift":
        params = {"dy": random.choice([-2, -1, 1, 2]), "dx": random.choice([-2, -1, 1, 2])}
    elif prim == "wrap":
        params = {"shift": random.randint(1, 3), "axis": random.choice([0, 1])}
    return ASTNode(NodeType.PRIMITIVE, name=prim, params=params)


def _sample_filter() -> ASTNode:
    """Sample a random filter."""
    name = random.choice(FILTERS)
    return ASTNode(NodeType.FILTER, name=name)


def _sample_condition() -> ASTNode:
    """Sample a random condition."""
    name = random.choice(CONDITIONS)
    return ASTNode(NodeType.CONDITION, name=name)


def sample_program(max_depth: int = 3, depth: int = 0) -> ASTNode:
    """Sample a random valid program from the recursive grammar.

    Args:
        max_depth: Maximum nesting depth.
        depth: Current depth.

    Returns:
        A random ASTNode representing a valid program.
    """
    # At max depth or randomly, produce a leaf
    if depth >= max_depth or (depth > 0 and random.random() < 0.4):
        return _sample_primitive()

    # Weight composition types
    choice = random.choices(
        ["primitive", "compose", "apply_to_objects", "branch"],
        weights=[2, 3, 3, 2],
        k=1,
    )[0]

    if choice == "primitive":
        return _sample_primitive()

    elif choice == "compose":
        left = sample_program(max_depth, depth + 1)
        right = sample_program(max_depth, depth + 1)
        return ASTNode(NodeType.COMPOSE, name="compose", children=[left, right])

    elif choice == "apply_to_objects":
        filt = _sample_filter()
        action = sample_program(max_depth, depth + 1)
        return ASTNode(NodeType.APPLY_TO_OBJECTS, name="apply_to_objects",
                       children=[filt, action])

    else:  # branch
        cond = _sample_condition()
        true_branch = sample_program(max_depth, depth + 1)
        false_branch = sample_program(max_depth, depth + 1)
        return ASTNode(NodeType.BRANCH, name="branch",
                       children=[cond, true_branch, false_branch])
