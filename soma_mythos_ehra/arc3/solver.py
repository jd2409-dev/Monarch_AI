"""ARC-AGI 3 Puzzle Solver — MCTS search over transformation sequences."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

from soma_mythos_ehra.arc3.adapter import ARC3Task
from soma_mythos_ehra.arc3.objects import extract_objects, connected_component_labeling
from soma_mythos_ehra.arc3.transforms import (
    NUM_TRANSFORMS,
    TransformType,
    apply_sequence,
    apply_fill_holes,
    apply_tile,
    apply_recolor_by_size,
    apply_sort_by_density,
    apply_sort_by_centroid,
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

        # Heuristic 1: Color map
        color_map = self._learn_color_map(train_inputs, train_outputs)
        if color_map:
            mapped = [self._apply_color_map(inp, color_map) for inp in train_inputs]
            if _compute_energy(mapped, train_outputs) == 0:
                return [{"transform": TransformType.COLOR_MAP, "color_map": color_map}]

        # Heuristic 2: Scale/resize pattern
        scale_result = self._try_scale_heuristic(train_inputs, train_outputs)
        if scale_result is not None:
            return scale_result

        # Heuristic 3: Rotation sequence
        rot_result = self._try_rotation_heuristic(train_inputs, train_outputs)
        if rot_result is not None:
            return rot_result

        # Heuristic 4: Object sorting
        sort_result = self._try_sort_heuristic(train_inputs, train_outputs)
        if sort_result is not None:
            return sort_result

        # Heuristic 5: Hole filling
        fill_result = self._try_fill_holes_heuristic(train_inputs, train_outputs)
        if fill_result is not None:
            return fill_result

        # Heuristic 6: Tiling pattern
        tile_result = self._try_tile_heuristic(train_inputs, train_outputs)
        if tile_result is not None:
            return tile_result

        # Heuristic 7: Component coloring
        comp_result = self._try_component_coloring_heuristic(train_inputs, train_outputs)
        if comp_result is not None:
            return comp_result

        # Heuristic 8: Recolor by size
        size_result = self._try_recolor_by_size_heuristic(train_inputs, train_outputs)
        if size_result is not None:
            return size_result

        # Heuristic 9: Sort by density
        density_result = self._try_sort_by_density_heuristic(train_inputs, train_outputs)
        if density_result is not None:
            return density_result

        # Heuristic 10: Sort by centroid
        centroid_result = self._try_sort_by_centroid_heuristic(train_inputs, train_outputs)
        if centroid_result is not None:
            return centroid_result

        # Full MCTS search
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

    def _try_scale_heuristic(self, inputs: list[torch.Tensor], targets: list[torch.Tensor]) -> list[dict] | None:
        """Try scale up/down patterns."""
        for inp, tgt in zip(inputs, targets):
            ih, iw = inp.shape
            th, tw = tgt.shape
            if ih == th and iw == tw:
                continue
            # Check if target is a scaled version of input
            if th >= ih and tw >= iw:
                fy, fx = th // ih, tw // iw
                if fy * ih == th and fx * iw == tw:
                    # Try tiling the input
                    from soma_mythos_ehra.arc3.transforms import apply_tile
                    tiled = apply_tile(inp, fy, fx)
                    if torch.equal(tiled, tgt):
                        return [{"transform": TransformType.TILE, "reps_h": fy, "reps_w": fx}]
            # Check if input is scaled version of target
            if ih <= th and iw <= tw:
                fy, fx = th // ih, tw // iw
                if fy * ih == th and fx * iw == tw:
                    from soma_mythos_ehra.arc3.transforms import apply_tile
                    tiled = apply_tile(inp, fy, fx)
                    if torch.equal(tiled, tgt):
                        return [{"transform": TransformType.TILE, "reps_h": fy, "reps_w": fx}]
        return None

    def _try_rotation_heuristic(self, inputs: list[torch.Tensor], targets: list[torch.Tensor]) -> list[dict] | None:
        """Try rotation/flip combinations."""
        from soma_mythos_ehra.arc3.transforms import apply_rotate_90, apply_rotate_180, apply_rotate_270, apply_flip_h, apply_flip_v, apply_transpose
        transforms = [
            ([{"transform": TransformType.ROTATE_90}], apply_rotate_90),
            ([{"transform": TransformType.ROTATE_180}], apply_rotate_180),
            ([{"transform": TransformType.ROTATE_270}], apply_rotate_270),
            ([{"transform": TransformType.FLIP_H}], apply_flip_h),
            ([{"transform": TransformType.FLIP_V}], apply_flip_v),
            ([{"transform": TransformType.TRANSPOSE}], apply_transpose),
            ([{"transform": TransformType.FLIP_H}, {"transform": TransformType.ROTATE_90}],
             lambda g: apply_rotate_90(apply_flip_h(g))),
            ([{"transform": TransformType.FLIP_V}, {"transform": TransformType.ROTATE_90}],
             lambda g: apply_rotate_90(apply_flip_v(g))),
        ]
        for seq, fn in transforms:
            all_match = True
            for inp, tgt in zip(inputs, targets):
                if inp.shape != tgt.shape:
                    all_match = False
                    break
                pred = fn(inp)
                if not torch.equal(pred, tgt):
                    all_match = False
                    break
            if all_match:
                return seq
        return None

    def _try_sort_heuristic(self, inputs: list[torch.Tensor], targets: list[torch.Tensor]) -> list[dict] | None:
        """Try sorting objects by position."""
        for axis in [0, 1]:
            all_match = True
            for inp, tgt in zip(inputs, targets):
                if inp.shape != tgt.shape:
                    all_match = False
                    break
                from soma_mythos_ehra.arc3.transforms import apply_sort_objects
                pred = apply_sort_objects(inp, axis)
                if not torch.equal(pred, tgt):
                    all_match = False
                    break
            if all_match:
                return [{"transform": TransformType.SORT_OBJECTS, "axis": axis}]
        return None

    def _try_fill_holes_heuristic(self, inputs: list[torch.Tensor], targets: list[torch.Tensor]) -> list[dict] | None:
        """Try filling interior holes."""
        all_match = True
        for inp, tgt in zip(inputs, targets):
            if inp.shape != tgt.shape:
                all_match = False
                break
            pred = apply_fill_holes(inp)
            if not torch.equal(pred, tgt):
                all_match = False
                break
        if all_match:
            return [{"transform": TransformType.FILL_HOLES}]
        return None

    def _try_tile_heuristic(self, inputs: list[torch.Tensor], targets: list[torch.Tensor]) -> list[dict] | None:
        """Try tiling a small pattern across the grid.

        Detects if the output is a tiled version of a corner/subregion of the input.
        """
        for inp, tgt in zip(inputs, targets):
            ih, iw = inp.shape
            th, tw = tgt.shape
            if th < ih or tw < iw:
                continue
            # Check all possible tile sizes
            for tile_h in range(1, ih + 1):
                for tile_w in range(1, iw + 1):
                    if th % tile_h != 0 or tw % tile_w != 0:
                        continue
                    # Extract tile from top-left of input
                    tile = inp[:tile_h, :tile_w]
                    # Check if target is tiled version
                    reps_h = th // tile_h
                    reps_w = tw // tile_w
                    tiled = apply_tile(tile, reps_h, reps_w)
                    if torch.equal(tiled, tgt):
                        return [{"transform": TransformType.TILE, "reps_h": reps_h, "reps_w": reps_w}]
        return None

    def _try_component_coloring_heuristic(self, inputs: list[torch.Tensor], targets: list[torch.Tensor]) -> list[dict] | None:
        """Try coloring each connected component with its label number.

        This handles puzzles where objects of the same color get different colors
        based on their connected component identity (label = output color).
        """
        for inp, tgt in zip(inputs, targets):
            if inp.shape != tgt.shape:
                continue
            labels = connected_component_labeling(inp)
            num_components = int(labels.max().item())
            if num_components == 0:
                continue

            # Check if output colors match labels exactly
            match = True
            for r in range(tgt.shape[0]):
                for c in range(tgt.shape[1]):
                    lbl = int(labels[r, c].item())
                    out_val = int(tgt[r, c].item())
                    if lbl != out_val:
                        match = False
                        break
                if not match:
                    break

            if match:
                return [{"transform": TransformType.COLOR_MAP, "color_map": labels, "type": "component_labeling"}]

        return None

    def _try_recolor_by_size_heuristic(self, inputs: list[torch.Tensor], targets: list[torch.Tensor]) -> list[dict] | None:
        """Try recoloring objects based on their size ordering.

        Largest object gets color 1, second largest gets color 2, etc.
        """
        all_match = True
        for inp, tgt in zip(inputs, targets):
            if inp.shape != tgt.shape:
                all_match = False
                break
            pred = apply_recolor_by_size(inp)
            if not torch.equal(pred, tgt):
                all_match = False
                break
        if all_match:
            return [{"transform": TransformType.RECOLOR_BY_SIZE}]
        return None

    def _try_sort_by_density_heuristic(self, inputs: list[torch.Tensor], targets: list[torch.Tensor]) -> list[dict] | None:
        """Try sorting objects by density (solidness)."""
        all_match = True
        for inp, tgt in zip(inputs, targets):
            if inp.shape != tgt.shape:
                all_match = False
                break
            pred = apply_sort_by_density(inp)
            if not torch.equal(pred, tgt):
                all_match = False
                break
        if all_match:
            return [{"transform": TransformType.SORT_BY_DENSITY}]
        return None

    def _try_sort_by_centroid_heuristic(self, inputs: list[torch.Tensor], targets: list[torch.Tensor]) -> list[dict] | None:
        """Try sorting objects by centroid position."""
        for axis in [0, 1]:
            all_match = True
            for inp, tgt in zip(inputs, targets):
                if inp.shape != tgt.shape:
                    all_match = False
                    break
                pred = apply_sort_by_centroid(inp, axis)
                if not torch.equal(pred, tgt):
                    all_match = False
                    break
            if all_match:
                return [{"transform": TransformType.SORT_BY_CENTROID, "axis": axis}]
        return None

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
        if t == TransformType.MOVE_OBJECT:
            params = []
            for inp in inputs[:1]:
                objects = extract_objects(inp)
                for obj in objects[:3]:
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        params.append({"obj_label": obj.label, "dy": dy, "dx": dx})
            return params[:12]
        if t == TransformType.FILL_HOLES:
            return [{}]
        if t == TransformType.SORT_OBJECTS:
            return [{"axis": 0}, {"axis": 1}]
        if t == TransformType.RECOLOR_BY_SIZE:
            return [{}]
        if t == TransformType.SORT_BY_DENSITY:
            return [{}]
        if t == TransformType.SORT_BY_CENTROID:
            return [{"axis": 0}, {"axis": 1}]
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
