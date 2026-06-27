"""Spatial Diff Classifier — predicts DSL primitive probabilities from grid pairs.

Analyzes structural differences between input and output grids to predict
which DSL primitives are likely needed, enabling execution-guided pruning.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpatialDiffClassifier(nn.Module):
    """Lightweight classifier that predicts primitive probabilities from grid pairs.

    Takes input/output grid pairs and outputs a probability vector over DSL primitives.
    Uses handcrafted features for speed (no training required).
    """

    # Map primitive names to indices
    PRIMITIVE_NAMES = [
        "objects", "objects_of_color", "largest_object", "smallest_object", "object_at",
        "filter_by_area", "filter_by_density", "filter_by_color", "filter_by_size", "filter_by_position",
        "take_n", "rotate", "flip", "transpose", "crop", "pad", "scale", "tile", "shift", "wrap",
        "recolor", "recolor_objects", "recolor_by_size", "recolor_by_density", "fill_holes", "flood_fill",
        "sort_by_position", "sort_by_area", "distance_between", "compose", "branch", "apply_to_objects",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.name_to_idx = {name: i for i, name in enumerate(self.PRIMITIVE_NAMES)}

    @torch.no_grad()
    def predict(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        """Predict primitive probabilities from a single input/output pair.

        Returns a tensor of shape (32,) with probabilities for each primitive.
        """
        features = self._extract_features(input_grid, output_grid)
        return self._features_to_probs(features)

    @torch.no_grad()
    def predict_batch(
        self,
        inputs: list[torch.Tensor],
        outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Predict averaged primitive probabilities from multiple train pairs.

        Returns a tensor of shape (32,) with averaged probabilities.
        """
        all_probs = []
        for inp, out in zip(inputs, outputs):
            probs = self.predict(inp, out)
            all_probs.append(probs)
        return torch.stack(all_probs).mean(dim=0)

    def _extract_features(self, inp: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
        """Extract structural features from input/output pair."""
        ih, iw = inp.shape
        oh, ow = out.shape

        # Shape features
        shape_changed = (ih != oh) or (iw != ow)
        height_ratio = oh / ih if ih > 0 else 1.0
        width_ratio = ow / iw if iw > 0 else 1.0
        area_ratio = (oh * ow) / (ih * iw) if (ih * iw) > 0 else 1.0
        is_scaled = (oh % ih == 0) and (ow % iw == 0) and (height_ratio == width_ratio)

        # Color features
        inp_colors = set(inp.flatten().tolist())
        out_colors = set(out.flatten().tolist())
        num_inp_colors = len(inp_colors)
        num_out_colors = len(out_colors)
        color_count_changed = num_inp_colors != num_out_colors
        new_colors = out_colors - inp_colors
        lost_colors = inp_colors - out_colors
        has_new_colors = len(new_colors) > 0
        has_lost_colors = len(lost_colors) > 0

        # Check if color mapping is consistent
        mapping = {}
        consistent_mapping = True
        min_h, min_w = min(ih, oh), min(iw, ow)
        for r in range(min_h):
            for c in range(min_w):
                iv, ov = int(inp[r, c].item()), int(out[r, c].item())
                if iv in mapping:
                    if mapping[iv] != ov:
                        consistent_mapping = False
                else:
                    mapping[iv] = ov

        # Object features (from input)
        inp_objects = self._count_objects(inp)
        out_objects = self._count_objects(out)
        object_count_changed = inp_objects != out_objects

        # Spatial features
        inp_density = self._compute_density(inp)
        out_density = self._compute_density(out)
        density_changed = abs(inp_density - out_density) > 0.1

        # Rotation detection
        is_rotated = False
        if not shape_changed:
            for k in [1, 2, 3]:
                rotated = torch.rot90(inp, k=k, dims=(0, 1))
                if rotated.shape == out.shape and torch.equal(rotated, out):
                    is_rotated = True
                    break

        # Flip detection
        is_flipped = False
        if not shape_changed:
            if torch.equal(inp.flip(1), out) or torch.equal(inp.flip(0), out):
                is_flipped = True

        # Fill holes detection
        inp_holes = self._count_interior_holes(inp)
        out_holes = self._count_interior_holes(out)
        has_holes = inp_holes > 0

        return {
            "shape_changed": float(shape_changed),
            "height_ratio": height_ratio,
            "width_ratio": width_ratio,
            "area_ratio": area_ratio,
            "is_scaled": float(is_scaled),
            "color_count_changed": float(color_count_changed),
            "has_new_colors": float(has_new_colors),
            "has_lost_colors": float(has_lost_colors),
            "consistent_mapping": float(consistent_mapping),
            "num_inp_colors": num_inp_colors,
            "num_out_colors": num_out_colors,
            "object_count_changed": float(object_count_changed),
            "inp_objects": inp_objects,
            "out_objects": out_objects,
            "density_changed": float(density_changed),
            "is_rotated": float(is_rotated),
            "is_flipped": float(is_flipped),
            "has_holes": float(has_holes),
        }

    def _features_to_probs(self, features: dict[str, float]) -> torch.Tensor:
        """Convert extracted features to primitive probabilities."""
        probs = torch.full((len(self.PRIMITIVE_NAMES),), 0.05)  # Base probability

        shape_changed = features["shape_changed"]
        is_scaled = features["is_scaled"]
        color_count_changed = features["color_count_changed"]
        consistent_mapping = features["consistent_mapping"]
        has_new_colors = features["has_new_colors"]
        is_rotated = features["is_rotated"]
        is_flipped = features["is_flipped"]
        has_holes = features["has_holes"]
        object_count_changed = features["object_count_changed"]
        inp_objects = features["inp_objects"]

        # If shape changed and is scaled -> scale/tile
        if shape_changed and is_scaled:
            probs[self.name_to_idx["scale"]] = 0.4
            probs[self.name_to_idx["tile"]] = 0.4
            probs[self.name_to_idx["crop"]] = 0.2

        # If shape changed but not scaled -> crop/pad
        elif shape_changed:
            probs[self.name_to_idx["crop"]] = 0.3
            probs[self.name_to_idx["pad"]] = 0.3

        # If consistent color mapping -> recolor
        if consistent_mapping and not shape_changed:
            probs[self.name_to_idx["recolor"]] = 0.5

        # If new colors appeared -> component labeling / recolor_by_size
        if has_new_colors and not shape_changed:
            probs[self.name_to_idx["objects"]] = 0.3
            probs[self.name_to_idx["recolor_by_size"]] = 0.3
            probs[self.name_to_idx["recolor_by_density"]] = 0.2

        # If colors lost -> filter / fill
        if features["has_lost_colors"] and not shape_changed:
            probs[self.name_to_idx["fill_holes"]] = 0.2
            probs[self.name_to_idx["filter_by_color"]] = 0.2

        # If rotated -> rotate
        if is_rotated:
            probs[self.name_to_idx["rotate"]] = 0.6

        # If flipped -> flip
        if is_flipped:
            probs[self.name_to_idx["flip"]] = 0.6

        # If has holes -> fill_holes
        if has_holes:
            probs[self.name_to_idx["fill_holes"]] = 0.4

        # If object count changed -> object primitives
        if object_count_changed:
            probs[self.name_to_idx["objects"]] = 0.3
            probs[self.name_to_idx["filter_by_area"]] = 0.2
            probs[self.name_to_idx["filter_by_size"]] = 0.2

        # If multiple objects exist -> sort/filter
        if inp_objects > 1 and not shape_changed:
            probs[self.name_to_idx["sort_by_position"]] = 0.2
            probs[self.name_to_idx["sort_by_area"]] = 0.2
            probs[self.name_to_idx["filter_by_area"]] = 0.2

        # Always include composition
        probs[self.name_to_idx["compose"]] = 0.3
        probs[self.name_to_idx["apply_to_objects"]] = 0.2

        # Normalize
        probs = probs / probs.sum()
        return probs

    def _count_objects(self, grid: torch.Tensor) -> int:
        """Count connected components in grid."""
        from soma_mythos_ehra.arc3.objects import connected_component_labeling
        labels = connected_component_labeling(grid)
        return int(labels.max().item())

    def _compute_density(self, grid: torch.Tensor) -> float:
        """Compute density of non-background pixels."""
        total = grid.numel()
        non_bg = (grid != 0).sum().item()
        return non_bg / total if total > 0 else 0.0

    def _count_interior_holes(self, grid: torch.Tensor) -> int:
        """Count interior holes (background pockets enclosed by non-background)."""
        H, W = grid.shape
        reachable = torch.zeros(H, W, dtype=torch.bool)
        stack = []
        for r in range(H):
            for c in [0, W - 1]:
                if int(grid[r, c].item()) == 0:
                    stack.append((r, c))
        for c in range(W):
            for r in [0, H - 1]:
                if int(grid[r, c].item()) == 0:
                    stack.append((r, c))
        while stack:
            r, c = stack.pop()
            if 0 <= r < H and 0 <= c < W and not reachable[r, c]:
                if int(grid[r, c].item()) == 0:
                    reachable[r, c] = True
                    stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
        holes = 0
        for r in range(H):
            for c in range(W):
                if int(grid[r, c].item()) == 0 and not reachable[r, c]:
                    holes += 1
        return holes
