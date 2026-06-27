"""ARC-AGI 3 Puzzle Solver — MCTS search over transformation sequences."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

from soma_mythos_ehra.arc3.adapter import ARC3Task
from soma_mythos_ehra.arc3.transforms import (
    NUM_TRANSFORMS,
    TransformType,
    apply_sequence,
)


@dataclass
class ARC3Config:
    max_depth: int = 6
    simulations: int = 512
    exploration: float = 3.0
    seed: int = 42


@dataclass
class ARC3Node:
    sequence: list[dict] = field(default_factory=list)
    parent: ARC3Node | None = None
    action_from_parent: dict | None = None
    visits: int = 0
    total_value: float = 0.0
    energy: float = 0.0
    children: dict[int, "ARC3Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.total_value / max(self.visits, 1)


def _compute_energy(current: list[torch.Tensor], target: list[torch.Tensor]) -> float:
    total = 0
    for cur, tgt in zip(current, target):
        if cur.shape == tgt.shape:
            total += int((cur != tgt).sum().item())
    return float(total)


def _hash_transform(step: dict) -> int:
    items = sorted((k, v) for k, v in step.items() if k != "transform")
    return hash((step["transform"], tuple(items)))


class ARC3Solver:
    def __init__(self, config: ARC3Config | None = None) -> None:
        self.config = config or ARC3Config()
        self.rng = random.Random(self.config.seed)

    def solve(self, task: ARC3Task) -> list[dict] | None:
        train_inputs = task.get_train_inputs()
        train_outputs = task.get_train_outputs()
        if not train_inputs:
            return None

        # Try color map first
        color_map = self._learn_color_map(train_inputs, train_outputs)
        if color_map:
            mapped = [self._apply_color_map(inp, color_map) for inp in train_inputs]
            if _compute_energy(mapped, train_outputs) == 0:
                return [{"transform": TransformType.COLOR_MAP, "color_map": color_map}]

        # Try MCTS
        result = self._search(train_inputs, train_outputs)
        if result is not None:
            return result

        # Fallback: return partial color map
        if color_map:
            return [{"transform": TransformType.COLOR_MAP, "color_map": color_map}]
        return None

    def _learn_color_map(self, inputs: list[torch.Tensor], outputs: list[torch.Tensor]) -> dict[int, int] | None:
        mapping: dict[int, int] = {}
        for inp, out in zip(inputs, outputs):
            if inp.shape != out.shape:
                return None
            for iv, ov in zip(inp.flatten().tolist(), out.flatten().tolist()):
                if iv in mapping:
                    if mapping[iv] != ov:
                        return None
                else:
                    mapping[iv] = ov
        return mapping if mapping else None

    def _apply_color_map(self, grid: torch.Tensor, mapping: dict[int, int]) -> torch.Tensor:
        out = grid.clone()
        for src, dst in mapping.items():
            out[grid == src] = dst
        return out

    def _search(self, inputs: list[torch.Tensor], targets: list[torch.Tensor]) -> list[dict] | None:
        root = ARC3Node()
        best_seq = None
        best_energy = float("inf")

        for _ in range(self.config.simulations):
            leaf = self._select(root)
            if len(leaf.sequence) < self.config.max_depth:
                children = self._expand(leaf, inputs, targets)
                if children:
                    leaf = min(children, key=lambda c: c.energy)
            value = leaf.energy + 0.1 * len(leaf.sequence)
            self._backprop(leaf, value)
            if leaf.energy < best_energy:
                best_energy = leaf.energy
                best_seq = leaf.sequence
            if best_energy == 0:
                break

        return best_seq if best_energy == 0 else None

    def _select(self, node: ARC3Node) -> ARC3Node:
        while node.children and len(node.children) >= NUM_TRANSFORMS:
            log_p = math.log(max(node.visits, 1))
            node = min(
                node.children.values(),
                key=lambda c: c.value - self.config.exploration * math.sqrt(log_p / max(c.visits, 1)),
            )
        return node

    def _expand(self, node: ARC3Node, inputs: list[torch.Tensor], targets: list[torch.Tensor]) -> list[ARC3Node]:
        children = []
        for t in range(NUM_TRANSFORMS):
            for params in self._params(t, inputs):
                step = {"transform": TransformType(t), **params}
                h = _hash_transform(step)
                if h in node.children:
                    continue
                grids = [apply_sequence(inp, node.sequence + [step]) for inp in inputs]
                energy = _compute_energy(grids, targets)
                child = ARC3Node(sequence=node.sequence + [step], parent=node, action_from_parent=step, energy=energy)
                node.children[h] = child
                children.append(child)
        return children

    def _params(self, t: int, inputs: list[torch.Tensor]) -> list[dict]:
        if t == TransformType.COLOR_MAP or t == TransformType.FLOOD_FILL:
            return []
        if t == TransformType.SHIFT_OBJECTS:
            return [{"dy": dy, "dx": dx} for dy in [-1, 0, 1] for dx in [-1, 0, 1] if dy or dx]
        if t == TransformType.SCALE_UP:
            return [{"factor": f} for f in [2, 3]]
        if t == TransformType.TILE:
            return [{"reps_h": r, "reps_w": r} for r in [2, 3]]
        if t == TransformType.WRAP_AROUND:
            return [{"shift": s, "axis": a} for s in [1, -1] for a in [0, 1]]
        return [{}]

    def _backprop(self, node: ARC3Node, value: float) -> None:
        while node:
            node.visits += 1
            node.total_value += value
            node = node.parent


def solve_task(task: ARC3Task, config: ARC3Config | None = None) -> dict:
    solver = ARC3Solver(config)
    train_inputs = task.get_train_inputs()
    train_outputs = task.get_train_outputs()

    solution = solver.solve(task)
    result = {"task_id": task.task_id, "solution": solution, "test_output": None, "train_accuracy": 0.0}

    if solution is not None:
        correct = sum(
            1 for inp, tgt in zip(train_inputs, train_outputs)
            if torch.equal(apply_sequence(inp, solution), tgt)
        )
        result["train_accuracy"] = correct / len(train_inputs) if train_inputs else 0.0
        test_input = task.get_test_input()
        if test_input is not None:
            result["test_output"] = apply_sequence(test_input, solution).tolist()
    else:
        # Fallback color map
        color_map = solver._learn_color_map(train_inputs, train_outputs)
        if color_map:
            test_input = task.get_test_input()
            if test_input is not None:
                mapped = test_input.clone()
                for src, dst in color_map.items():
                    mapped[test_input == src] = dst
                result["test_output"] = mapped.tolist()
                result["solution"] = [{"transform": TransformType.COLOR_MAP, "color_map": color_map}]

    return result
