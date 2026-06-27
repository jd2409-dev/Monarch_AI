"""ARC DSL Execution Kernel — runs synthesized programs on GPU tensors.

Provides a safe, vectorized execution environment for DSL program ASTs.
Each primitive maps to a concrete tensor operation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from soma_mythos_ehra.arc3.dsl_grammar import DSLNode, PRIMITIVES
from soma_mythos_ehra.arc3.objects import (
    ARCObject,
    connected_component_labeling,
    extract_objects,
)


@dataclass
class ExecutionContext:
    """Mutable context passed through program execution."""
    grid: torch.Tensor
    objects: list[ARCObject] | None = None
    masks: list[torch.Tensor] | None = None
    metadata: dict[str, Any] | None = None


class DSLExecutionError(Exception):
    """Raised when a DSL program fails to execute."""
    pass


class DSLKernel:
    """Execution kernel for DSL program ASTs."""

    def __init__(self, background: int = 0) -> None:
        self.background = background

    def execute(self, program: DSLNode, grid: torch.Tensor) -> torch.Tensor | None:
        """Execute a DSL program on a grid tensor.

        Returns the transformed grid, or None if execution fails.
        """
        try:
            ctx = ExecutionContext(grid=grid.clone())
            result = self._eval(program, ctx)
            if isinstance(result, torch.Tensor) and result.shape == grid.shape:
                return result
            return None
        except Exception:
            return None

    def execute_on_pairs(
        self,
        program: DSLNode,
        inputs: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> tuple[int, list[torch.Tensor]]:
        """Execute program on all train pairs.

        Returns (num_correct, list_of_predictions).
        """
        predictions = []
        correct = 0
        for inp, tgt in zip(inputs, targets):
            pred = self.execute(program, inp)
            if pred is not None and torch.equal(pred, tgt):
                correct += 1
            predictions.append(pred if pred is not None else inp.clone())
        return correct, predictions

    def _eval(self, node: DSLNode, ctx: ExecutionContext) -> Any:
        """Recursively evaluate a DSL node."""
        prim = node.primitive
        params = node.params

        # === OBJECT PRIMITIVES ===
        if prim == "objects":
            ctx.objects = extract_objects(ctx.grid, self.background)
            return ctx.objects

        if prim == "objects_of_color":
            color = params.get("color", 1)
            all_objects = extract_objects(ctx.grid, self.background)
            ctx.objects = [o for o in all_objects if o.color == color]
            return ctx.objects

        if prim == "largest_object":
            if ctx.objects:
                return max(ctx.objects, key=lambda o: o.area)
            return None

        if prim == "smallest_object":
            if ctx.objects:
                return min(ctx.objects, key=lambda o: o.area)
            return None

        if prim == "object_at":
            if not ctx.objects:
                return None
            position = params.get("position", "center")
            if position == "top_left":
                return min(ctx.objects, key=lambda o: o.centroid[0] + o.centroid[1])
            elif position == "bottom_right":
                return max(ctx.objects, key=lambda o: o.centroid[0] + o.centroid[1])
            elif position == "center":
                H, W = ctx.grid.shape
                return min(ctx.objects, key=lambda o: abs(o.centroid[0] - H/2) + abs(o.centroid[1] - W/2))
            return ctx.objects[0]

        # === FILTER PRIMITIVES ===
        if prim == "filter_by_area":
            if not ctx.objects:
                return []
            mode = params.get("mode", "max")
            areas = [o.area for o in ctx.objects]
            if mode == "max":
                max_area = max(areas)
                return [o for o in ctx.objects if o.area == max_area]
            elif mode == "min":
                min_area = min(areas)
                return [o for o in ctx.objects if o.area == min_area]
            return ctx.objects

        if prim == "filter_by_density":
            if not ctx.objects:
                return []
            mode = params.get("mode", "max")
            densities = [o.density for o in ctx.objects]
            if mode == "max":
                max_d = max(densities)
                return [o for o in ctx.objects if o.density == max_d]
            elif mode == "min":
                min_d = min(densities)
                return [o for o in ctx.objects if o.density == min_d]
            return ctx.objects

        if prim == "filter_by_color":
            color = params.get("color", 1)
            if not ctx.objects:
                return []
            return [o for o in ctx.objects if o.color == color]

        if prim == "filter_by_size":
            if not ctx.objects:
                return []
            mode = params.get("mode", "largest")
            if mode == "largest":
                max_a = max(o.area for o in ctx.objects)
                return [o for o in ctx.objects if o.area == max_a]
            elif mode == "smallest":
                min_a = min(o.area for o in ctx.objects)
                return [o for o in ctx.objects if o.area == min_a]
            return ctx.objects

        if prim == "filter_by_position":
            if not ctx.objects:
                return []
            region = params.get("region", "center")
            H, W = ctx.grid.shape
            filtered = []
            for o in ctx.objects:
                cy, cx = o.centroid
                if region == "top" and cy < H / 3:
                    filtered.append(o)
                elif region == "bottom" and cy > 2 * H / 3:
                    filtered.append(o)
                elif region == "left" and cx < W / 3:
                    filtered.append(o)
                elif region == "right" and cx > 2 * W / 3:
                    filtered.append(o)
                elif region == "center":
                    if H/3 <= cy <= 2*H/3 and W/3 <= cx <= 2*W/3:
                        filtered.append(o)
            return filtered

        if prim == "take_n":
            n = params.get("n", 1)
            order = params.get("order", "first")
            if not ctx.objects:
                return []
            if order == "last":
                return ctx.objects[-n:]
            return ctx.objects[:n]

        # === TRANSFORM PRIMITIVES ===
        if prim == "rotate":
            angle = params.get("angle", 90)
            k = {90: -1, 180: 2, 270: 1}.get(angle, -1)
            ctx.grid = torch.rot90(ctx.grid, k=k, dims=(0, 1))
            return ctx.grid

        if prim == "flip":
            axis = params.get("axis", "h")
            if axis == "h":
                ctx.grid = ctx.grid.flip(1)
            else:
                ctx.grid = ctx.grid.flip(0)
            return ctx.grid

        if prim == "transpose":
            ctx.grid = ctx.grid.t()
            return ctx.grid

        if prim == "crop":
            region = params.get("region", "center")
            H, W = ctx.grid.shape
            if region == "top_left":
                h, w = H // 2, W // 2
                ctx.grid = ctx.grid[:h, :w]
            elif region == "center":
                h, w = H // 2, W // 2
                y0, x0 = H // 4, W // 4
                ctx.grid = ctx.grid[y0:y0+h, x0:x0+w]
            return ctx.grid

        if prim == "pad":
            size = params.get("size", 1)
            color = params.get("color", self.background)
            H, W = ctx.grid.shape
            new_grid = torch.full((H + 2*size, W + 2*size), color, dtype=ctx.grid.dtype)
            new_grid[size:size+H, size:size+W] = ctx.grid
            ctx.grid = new_grid
            return ctx.grid

        if prim == "scale":
            factor = params.get("factor", 2)
            ctx.grid = ctx.grid.repeat_interleave(factor, dim=0).repeat_interleave(factor, dim=1)
            return ctx.grid

        if prim == "tile":
            reps_h = params.get("reps_h", 2)
            reps_w = params.get("reps_w", 2)
            ctx.grid = ctx.grid.repeat(reps_h, reps_w)
            return ctx.grid

        if prim == "shift":
            dy = params.get("dy", 0)
            dx = params.get("dx", 0)
            ctx.grid = torch.roll(ctx.grid, shifts=(dy, dx), dims=(0, 1))
            return ctx.grid

        if prim == "wrap":
            shift = params.get("shift", 1)
            axis = params.get("axis", 0)
            ctx.grid = torch.roll(ctx.grid, shifts=shift, dims=axis)
            return ctx.grid

        # === COLOR PRIMITIVES ===
        if prim == "recolor":
            src = params.get("src", 0)
            dst = params.get("dst", 1)
            ctx.grid[ctx.grid == src] = dst
            return ctx.grid

        if prim == "recolor_objects":
            color = params.get("color", 1)
            if ctx.objects:
                out = ctx.grid.clone()
                for obj in ctx.objects:
                    for py, px in obj.pixels:
                        out[py, px] = color
                ctx.grid = out
            return ctx.grid

        if prim == "recolor_by_size":
            if ctx.objects:
                sorted_objs = sorted(ctx.objects, key=lambda o: o.area, reverse=True)
                out = ctx.grid.clone()
                for i, obj in enumerate(sorted_objs):
                    for py, px in obj.pixels:
                        out[py, px] = i + 1
                ctx.grid = out
            return ctx.grid

        if prim == "recolor_by_density":
            if ctx.objects:
                sorted_objs = sorted(ctx.objects, key=lambda o: o.density, reverse=True)
                out = ctx.grid.clone()
                for i, obj in enumerate(sorted_objs):
                    for py, px in obj.pixels:
                        out[py, px] = i + 1
                ctx.grid = out
            return ctx.grid

        if prim == "fill_holes":
            H, W = ctx.grid.shape
            reachable = torch.zeros(H, W, dtype=torch.bool)
            stack = []
            for r in range(H):
                for c in [0, W - 1]:
                    if int(ctx.grid[r, c].item()) == self.background:
                        stack.append((r, c))
            for c in range(W):
                for r in [0, H - 1]:
                    if int(ctx.grid[r, c].item()) == self.background:
                        stack.append((r, c))
            while stack:
                r, c = stack.pop()
                if 0 <= r < H and 0 <= c < W and not reachable[r, c]:
                    if int(ctx.grid[r, c].item()) == self.background:
                        reachable[r, c] = True
                        stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
            out = ctx.grid.clone()
            for r in range(H):
                for c in range(W):
                    if int(ctx.grid[r, c].item()) == self.background and not reachable[r, c]:
                        for dr in range(-1, 2):
                            for dc in range(-1, 2):
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < H and 0 <= nc < W:
                                    val = int(ctx.grid[nr, nc].item())
                                    if val != self.background:
                                        out[r, c] = val
                                        break
                            else:
                                continue
                            break
            ctx.grid = out
            return ctx.grid

        if prim == "flood_fill":
            y = params.get("y", 0)
            x = params.get("x", 0)
            color = params.get("color", 1)
            H, W = ctx.grid.shape
            if 0 <= y < H and 0 <= x < W:
                target = int(ctx.grid[y, x].item())
                if target != color:
                    stack = [(y, x)]
                    visited = set()
                    while stack:
                        r, c = stack.pop()
                        if (r, c) in visited or r < 0 or r >= H or c < 0 or c >= W:
                            continue
                        if int(ctx.grid[r, c].item()) != target:
                            continue
                        visited.add((r, c))
                        ctx.grid[r, c] = color
                        stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
            return ctx.grid

        # === SPATIAL PRIMITIVES ===
        if prim == "sort_by_position":
            axis = params.get("axis", 0)
            if ctx.objects:
                ctx.objects.sort(key=lambda o: o.centroid[axis])
            return ctx.objects

        if prim == "sort_by_area":
            reverse = params.get("reverse", True)
            if ctx.objects:
                ctx.objects.sort(key=lambda o: o.area, reverse=reverse)
            return ctx.objects

        # === COMPOSITION PRIMITIVES ===
        if prim == "compose":
            children = node.children
            for child in children:
                result = self._eval(child, ctx)
            return ctx.grid

        if prim == "apply_to_objects":
            if len(node.children) >= 2:
                self._eval(node.children[0], ctx)  # Filter
                self._eval(node.children[1], ctx)  # Transform
            return ctx.grid

        # Unknown primitive: pass through
        return ctx.grid
