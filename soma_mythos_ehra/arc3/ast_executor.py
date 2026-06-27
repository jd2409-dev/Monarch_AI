"""AST Executor — runs recursive grammar programs on grids via DSLKernel.

Translates the recursive grammar AST into DSLKernel calls, handling
compositions, object-scoped operations, and conditional branches.
"""
from __future__ import annotations

import torch

from soma_mythos_ehra.arc3.dsl_grammar import DSLNode
from soma_mythos_ehra.arc3.dsl_kernel import DSLKernel
from soma_mythos_ehra.arc3.objects import extract_objects
from soma_mythos_ehra.arc3.recursive_grammar import (
    ASTNode,
    NodeType,
    FILTER_MAP,
    PRIMITIVE_MAP,
)


class ASTExecutor:
    """Executes recursive grammar ASTs on grid tensors."""

    def __init__(self, background: int = 0) -> None:
        self.kernel = DSLKernel(background=background)
        self.background = background

    def execute(self, program: ASTNode, grid: torch.Tensor) -> torch.Tensor | None:
        """Execute an AST program on a grid.

        Returns the transformed grid, or None if execution fails.
        """
        try:
            result = self._eval(program, grid.clone())
            if isinstance(result, torch.Tensor) and result.shape == grid.shape:
                return result
            return None
        except Exception:
            return None

    def execute_on_pairs(
        self,
        program: ASTNode,
        inputs: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> tuple[int, list[torch.Tensor]]:
        """Execute on all (input, target) pairs, return count of correct solutions."""
        predictions = []
        correct = 0
        for inp, tgt in zip(inputs, targets):
            pred = self.execute(program, inp)
            if pred is not None and torch.equal(pred, tgt):
                correct += 1
            predictions.append(pred if pred is not None else inp.clone())
        return correct, predictions

    def _eval(self, node: ASTNode, grid: torch.Tensor) -> torch.Tensor:
        """Recursively evaluate an AST node."""
        if node.node_type == NodeType.PRIMITIVE:
            return self._eval_primitive(node, grid)

        elif node.node_type == NodeType.COMPOSE:
            return self._eval_compose(node, grid)

        elif node.node_type == NodeType.APPLY_TO_OBJECTS:
            return self._eval_apply_to_objects(node, grid)

        elif node.node_type == NodeType.BRANCH:
            return self._eval_branch(node, grid)

        return grid

    def _eval_primitive(self, node: ASTNode, grid: torch.Tensor) -> torch.Tensor:
        """Execute a leaf primitive."""
        prim_name = node.name
        params = node.params

        # Rotate
        if prim_name == "rotate":
            angle = params.get("angle", 90)
            k = {90: -1, 180: 2, 270: 1}.get(angle, -1)
            return torch.rot90(grid, k=k, dims=(0, 1))

        # Flip
        if prim_name == "flip":
            axis = params.get("axis", "h")
            return grid.flip(1 if axis == "h" else 0)

        # Transpose
        if prim_name == "transpose":
            return grid.t()

        # Scale
        if prim_name == "scale":
            factor = params.get("factor", 2)
            return grid.repeat_interleave(factor, dim=0).repeat_interleave(factor, dim=1)

        # Recolor
        if prim_name == "recolor":
            src = params.get("src", 1)
            dst = params.get("dst", 2)
            out = grid.clone()
            out[grid == src] = dst
            return out

        # Fill holes
        if prim_name == "fill_holes":
            dsl_node = DSLNode(primitive="fill_holes")
            result = self.kernel.execute(dsl_node, grid)
            return result if result is not None else grid

        # Flood fill
        if prim_name == "flood_fill":
            y = params.get("y", 0)
            x = params.get("x", 0)
            color = params.get("color", 3)
            H, W = grid.shape
            if 0 <= y < H and 0 <= x < W:
                target = int(grid[y, x].item())
                if target != color:
                    out = grid.clone()
                    stack = [(y, x)]
                    visited = set()
                    while stack:
                        r, c = stack.pop()
                        if (r, c) in visited or r < 0 or r >= H or c < 0 or c >= W:
                            continue
                        if int(out[r, c].item()) != target:
                            continue
                        visited.add((r, c))
                        out[r, c] = color
                        stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
                    return out
            return grid

        # Shift
        if prim_name == "shift":
            dy = params.get("dy", 0)
            dx = params.get("dx", 0)
            return torch.roll(grid, shifts=(dy, dx), dims=(0, 1))

        # Wrap
        if prim_name == "wrap":
            shift = params.get("shift", 1)
            axis = params.get("axis", 0)
            return torch.roll(grid, shifts=shift, dims=axis)

        return grid

    def _eval_compose(self, node: ASTNode, grid: torch.Tensor) -> torch.Tensor:
        """Execute compose: apply left then right."""
        if len(node.children) >= 2:
            mid = self._eval(node.children[0], grid)
            return self._eval(node.children[1], mid)
        return grid

    def _eval_apply_to_objects(self, node: ASTNode, grid: torch.Tensor) -> torch.Tensor:
        """Execute apply_to_objects: filter objects, then apply action to each."""
        if len(node.children) < 2:
            return grid

        filt_node = node.children[0]
        action_node = node.children[1]

        # Extract objects
        objects = extract_objects(grid, self.background)
        if not objects:
            return grid

        # Apply filter
        filtered = self._apply_filter(filt_node, objects)
        if not filtered:
            return grid

        # Apply action to each filtered object
        out = grid.clone()
        for obj in filtered:
            # Create a sub-grid with just this object
            obj_grid = torch.zeros_like(grid)
            for py, px in obj.pixels:
                obj_grid[py, px] = grid[py, px]

            # Execute action on sub-grid
            action_result = self._eval(action_node, obj_grid)

            # Merge result back
            if action_result is not None:
                for py, px in obj.pixels:
                    if 0 <= py < action_result.shape[0] and 0 <= px < action_result.shape[1]:
                        out[py, px] = action_result[py, px]

        return out

    def _eval_branch(self, node: ASTNode, grid: torch.Tensor) -> torch.Tensor:
        """Execute branch: evaluate condition, take true or false path."""
        if len(node.children) < 3:
            return grid

        cond_node = node.children[0]
        true_node = node.children[1]
        false_node = node.children[2]

        # Evaluate condition
        cond_result = self._eval_condition(cond_node, grid)

        if cond_result:
            return self._eval(true_node, grid)
        else:
            return self._eval(false_node, grid)

    def _apply_filter(self, node: ASTNode, objects: list) -> list:
        """Apply a filter to a list of objects."""
        name = node.name

        if name == "all":
            return objects

        if name == "by_area_max":
            if not objects:
                return []
            max_area = max(o.area for o in objects)
            return [o for o in objects if o.area == max_area]

        if name == "by_area_min":
            if not objects:
                return []
            min_area = min(o.area for o in objects)
            return [o for o in objects if o.area == min_area]

        if name == "by_density_solid":
            if not objects:
                return []
            max_d = max(o.density for o in objects)
            return [o for o in objects if o.density == max_d]

        if name == "by_density_hollow":
            if not objects:
                return []
            min_d = min(o.density for o in objects)
            return [o for o in objects if o.density == min_d]

        return objects

    def _eval_condition(self, node: ASTNode, grid: torch.Tensor) -> bool:
        """Evaluate a condition on the current grid."""
        name = node.name
        objects = extract_objects(grid, self.background)

        if name == "is_largest":
            if not objects:
                return False
            max_area = max(o.area for o in objects)
            return any(o.area == max_area for o in objects)

        if name == "is_smallest":
            if not objects:
                return False
            min_area = min(o.area for o in objects)
            return any(o.area == min_area for o in objects)

        if name == "is_solid":
            if not objects:
                return False
            return any(o.density > 0.8 for o in objects)

        if name == "is_hollow":
            if not objects:
                return False
            return any(o.density < 0.5 for o in objects)

        if name == "has_area_gt_5":
            return any(o.area > 5 for o in objects)

        if name == "has_area_lt_5":
            return any(o.area < 5 for o in objects)

        return False
