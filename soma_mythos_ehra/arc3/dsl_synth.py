"""DSL Program Synthesis MCTS — searches over program trees to solve ARC puzzles.

Instead of searching fixed transform sequences, this solver synthesizes
custom programs from the DSL grammar to solve each puzzle.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

from soma_mythos_ehra.arc3.adapter import ARC3Task
from soma_mythos_ehra.arc3.dsl_grammar import DSLNode, PRIMITIVES, get_legal_expansions
from soma_mythos_ehra.arc3.dsl_kernel import DSLKernel


@dataclass
class SynthConfig:
    max_depth: int = 5
    max_program_size: int = 15
    simulations: int = 1024
    exploration: float = 2.0
    seed: int = 42


@dataclass
class SynthNode:
    """A node in the MCTS tree for program synthesis."""
    program: DSLNode = field(default_factory=lambda: DSLNode(primitive="compose"))
    parent: SynthNode | None = None
    action_from_parent: str | None = None
    visits: int = 0
    total_value: float = 0.0
    energy: float = float("inf")
    children: dict[str, "SynthNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.total_value / max(self.visits, 1)


class DSLSynthesizer:
    """MCTS-based program synthesizer using the ARC DSL."""

    def __init__(self, config: SynthConfig | None = None) -> None:
        self.config = config or SynthConfig()
        self.rng = random.Random(self.config.seed)
        self.kernel = DSLKernel(background=0)

    def synthesize(self, task: ARC3Task) -> DSLNode | None:
        """Search for a DSL program that solves the task.

        Returns the best program AST, or None if no solution found.
        """
        train_inputs = task.get_train_inputs()
        train_outputs = task.get_train_outputs()
        if not train_inputs:
            return None

        # Try common program templates first (fast path)
        template = self._try_templates(train_inputs, train_outputs)
        if template is not None:
            return template

        # MCTS over program space
        return self._mcts_search(train_inputs, train_outputs)

    def _try_templates(
        self,
        inputs: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> DSLNode | None:
        """Try common program templates that cover many ARC puzzles."""
        templates = self._generate_templates()
        for prog in templates:
            correct, _ = self.kernel.execute_on_pairs(prog, inputs, targets)
            if correct == len(inputs):
                return prog
        return None

    def _generate_templates(self) -> list[DSLNode]:
        """Generate a library of common program templates."""
        templates = []

        # Template 1: Component labeling by size
        templates.append(DSLNode(
            primitive="compose",
            children=[
                DSLNode(primitive="objects"),
                DSLNode(primitive="recolor_by_size"),
            ],
        ))

        # Template 2: Rotate + single recolor (limited colors)
        for angle in [90, 180, 270]:
            templates.append(DSLNode(
                primitive="compose",
                children=[
                    DSLNode(primitive="rotate", params={"angle": angle}),
                    DSLNode(primitive="recolor", params={"src": 0, "dst": 1}),
                ],
            ))

        # Template 3: Flip + single recolor
        for axis in ["h", "v"]:
            templates.append(DSLNode(
                primitive="compose",
                children=[
                    DSLNode(primitive="flip", params={"axis": axis}),
                    DSLNode(primitive="recolor", params={"src": 0, "dst": 1}),
                ],
            ))

        # Template 4: Scale
        for factor in [2, 3]:
            templates.append(DSLNode(
                primitive="compose",
                children=[
                    DSLNode(primitive="scale", params={"factor": factor}),
                ],
            ))

        # Template 5: Tile
        for r in [2, 3]:
            templates.append(DSLNode(
                primitive="compose",
                children=[
                    DSLNode(primitive="tile", params={"reps_h": r, "reps_w": r}),
                ],
            ))

        # Template 6: Fill holes
        templates.append(DSLNode(primitive="fill_holes"))

        # Template 7: Objects -> filter by area -> recolor
        for color in range(1, 5):
            templates.append(DSLNode(
                primitive="apply_to_objects",
                children=[
                    DSLNode(primitive="filter_by_area", params={"mode": "max"}),
                    DSLNode(primitive="recolor_objects", params={"color": color}),
                ],
            ))

        # Template 8: Sort by position -> recolor by size
        for axis in [0, 1]:
            templates.append(DSLNode(
                primitive="compose",
                children=[
                    DSLNode(primitive="objects"),
                    DSLNode(primitive="sort_by_position", params={"axis": axis}),
                    DSLNode(primitive="recolor_by_size"),
                ],
            ))

        return templates

    def _mcts_search(
        self,
        inputs: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> DSLNode | None:
        """MCTS search over program space."""
        import time
        start_time = time.time()
        timeout = 5.0  # 5 second timeout per task

        root = SynthNode()
        best_program = None
        best_energy = float("inf")

        for _ in range(self.config.simulations):
            if time.time() - start_time > timeout:
                break

            # Selection
            leaf = self._select(root)

            # Expansion
            if leaf.program.depth() < self.config.max_depth and leaf.program.size() < self.config.max_program_size:
                children = self._expand(leaf, inputs, targets)
                if children:
                    leaf = min(children, key=lambda c: c.energy)

            # Evaluation (already computed during expansion)
            value = leaf.energy

            # Backpropagation
            self._backprop(leaf, value)

            # Track best
            if leaf.energy < best_energy:
                best_energy = leaf.energy
                best_program = leaf.program

            # Early termination
            if best_energy == 0:
                break

        if best_energy == 0 and best_program is not None:
            return best_program
        return None

    def _select(self, node: SynthNode) -> SynthNode:
        """Select a leaf node using UCB1."""
        while node.children:
            log_parent = math.log(max(node.visits, 1))
            node = min(
                node.children.values(),
                key=lambda child: child.value
                - self.config.exploration * math.sqrt(log_parent / max(child.visits, 1)),
            )
        return node

    def _expand(
        self,
        node: SynthNode,
        inputs: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> list[SynthNode]:
        """Expand a node by trying all legal primitive additions."""
        children = []
        legal = self._get_legal_primitives(node)

        for prim_name in legal:
            if prim_name in node.children:
                continue

            # Create new program by adding this primitive
            new_program = self._add_primitive(node.program, prim_name)
            if new_program is None:
                continue

            # Evaluate on all train pairs
            correct, _ = self.kernel.execute_on_pairs(new_program, inputs, targets)
            energy = len(inputs) - correct  # 0 = perfect

            child = SynthNode(
                program=new_program,
                parent=node,
                action_from_parent=prim_name,
                energy=energy,
            )
            node.children[prim_name] = child
            children.append(child)

        return children

    def _get_legal_primitives(self, node: SynthNode) -> list[str]:
        """Get legal primitive names for expansion."""
        prog = node.program
        if not prog.children:
            # Root: can add any transform
            return [p for p, v in PRIMITIVES.items()
                    if v.ptype.value in [2, 3, 4]]  # TRANSFORM, COLOR, SPATIAL
        # Non-root: can add transforms or composition
        return [p for p, v in PRIMITIVES.items()
                if v.ptype.value in [2, 3, 4, 5]]  # + COMPOSITION

    def _add_primitive(self, program: DSLNode, prim_name: str) -> DSLNode | None:
        """Add a primitive to the program tree."""
        prim = PRIMITIVES.get(prim_name)
        if prim is None:
            return None

        # Create new node
        new_node = DSLNode(primitive=prim_name, params=self._default_params(prim_name))

        # If program is empty (compose with no children), make it the first child
        if program.primitive == "compose" and not program.children:
            return DSLNode(
                primitive="compose",
                children=[new_node],
            )

        # If program is a single transform, wrap in compose
        if program.primitive != "compose":
            return DSLNode(
                primitive="compose",
                children=[program, new_node],
            )

        # Program is a compose: add as new child
        return DSLNode(
            primitive="compose",
            children=program.children + [new_node],
        )

    def _default_params(self, prim_name: str) -> dict:
        """Get default parameters for a primitive."""
        defaults = {
            "rotate": {"angle": 90},
            "flip": {"axis": "h"},
            "scale": {"factor": 2},
            "tile": {"reps_h": 2, "reps_w": 2},
            "shift": {"dy": 1, "dx": 0},
            "wrap": {"shift": 1, "axis": 0},
            "recolor": {"src": 0, "dst": 1},
            "recolor_objects": {"color": 1},
            "flood_fill": {"y": 0, "x": 0, "color": 1},
            "filter_by_area": {"mode": "max"},
            "filter_by_density": {"mode": "max"},
            "filter_by_color": {"color": 1},
            "filter_by_size": {"mode": "largest"},
            "filter_by_position": {"region": "center"},
            "take_n": {"n": 1, "order": "first"},
            "sort_by_position": {"axis": 0},
            "sort_by_area": {"reverse": True},
            "object_at": {"position": "center"},
        }
        return defaults.get(prim_name, {})

    def _backprop(self, node: SynthNode, value: float) -> None:
        """Backpropagate value up the tree."""
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent


def synthesize_task(task: ARC3Task, config: SynthConfig | None = None) -> dict:
    """Synthesize a DSL program to solve an ARC task.

    Returns a dict with:
        - "program": the synthesized DSL program AST
        - "program_str": string representation of the program
        - "test_output": the predicted test output grid
        - "train_accuracy": fraction of train pairs solved
    """
    synth = DSLSynthesizer(config)
    train_inputs = task.get_train_inputs()
    train_outputs = task.get_train_outputs()

    program = synth.synthesize(task)

    result = {
        "task_id": task.task_id,
        "program": program,
        "program_str": program.to_string() if program else None,
        "test_output": None,
        "train_accuracy": 0.0,
    }

    if program is not None:
        correct, _ = synth.kernel.execute_on_pairs(program, train_inputs, train_outputs)
        result["train_accuracy"] = correct / len(train_inputs) if train_inputs else 0.0

        test_input = task.get_test_input()
        if test_input is not None:
            test_output = synth.kernel.execute(program, test_input)
            if test_output is not None:
                result["test_output"] = test_output.tolist()

    return result
