"""Object Detection and Spatial Feature Maps for ARC-AGI puzzle solving.

Provides connected component labeling, bounding box extraction, and
multi-channel spatial feature maps for object-centric reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ARCObject:
    """A detected object in the grid with geometric metadata."""
    color: int
    label: int
    bbox: tuple[int, int, int, int]  # ymin, ymax, xmin, xmax
    centroid: tuple[float, float]
    mask: torch.Tensor
    area: int
    pixels: list[tuple[int, int]]
    # Geometric invariants
    density: float = 0.0      # area / bbox_area (solidness)
    aspect_ratio: float = 1.0  # height / width
    bbox_area: int = 0         # height * width
    height: int = 0
    width: int = 0


def connected_component_labeling(grid: torch.Tensor, background: int = 0) -> torch.Tensor:
    """BFS-based connected component labeling on a (H, W) grid.

    Returns a label map where each connected region of the same non-background
    color gets a unique integer label (1-indexed). Background stays 0.
    """
    H, W = grid.shape
    labels = torch.zeros_like(grid, dtype=torch.long)
    current_label = 0

    for r in range(H):
        for c in range(W):
            val = int(grid[r, c].item())
            if val == background or labels[r, c].item() != 0:
                continue
            # BFS flood fill for this component
            current_label += 1
            stack = [(r, c)]
            while stack:
                y, x = stack.pop()
                if y < 0 or y >= H or x < 0 or x >= W:
                    continue
                if int(labels[y, x].item()) != 0:
                    continue
                if int(grid[y, x].item()) != val:
                    continue
                labels[y, x] = current_label
                stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])

    return labels


def extract_objects(grid: torch.Tensor, background: int = 0) -> list[ARCObject]:
    """Extract all discrete objects from a grid tensor.

    Returns a list of ARCObject with bounding boxes, centroids, masks,
    area, and geometric invariants (density, aspect_ratio).
    """
    H, W = grid.shape
    labels = connected_component_labeling(grid, background)
    unique_labels = torch.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    objects = []
    for lbl in unique_labels:
        mask = (labels == lbl)
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        if not torch.any(rows):
            continue

        row_indices = torch.where(rows)[0]
        col_indices = torch.where(cols)[0]
        ymin, ymax = row_indices[0].item(), row_indices[-1].item()
        xmin, xmax = col_indices[0].item(), col_indices[-1].item()
        area = int(mask.sum().item())

        # Bounding box dimensions
        height = ymax - ymin + 1
        width = xmax - xmin + 1
        bbox_area = height * width

        # Geometric invariants
        density = area / bbox_area if bbox_area > 0 else 0.0
        aspect_ratio = height / width if width > 0 else 1.0

        # Get color from the original grid
        color_vals = grid[mask]
        color = int(color_vals[0].item()) if color_vals.numel() > 0 else 0

        # Centroid
        cy = (ymin + ymax) / 2.0
        cx = (xmin + xmax) / 2.0

        # Pixel coordinates
        pixel_coords = mask.nonzero(as_tuple=False)
        pixels = [(p[0].item(), p[1].item()) for p in pixel_coords]

        objects.append(ARCObject(
            color=color,
            label=int(lbl.item()),
            bbox=(ymin, ymax, xmin, xmax),
            centroid=(cy, cx),
            mask=mask[ymin:ymax+1, xmin:xmax+1],
            area=area,
            pixels=pixels,
            density=density,
            aspect_ratio=aspect_ratio,
            bbox_area=bbox_area,
            height=height,
            width=width,
        ))

    return objects


def compute_boundary_mask(grid: torch.Tensor, background: int = 0) -> torch.Tensor:
    """Compute a boundary/edge map of the grid.

    Returns a tensor of shape (H, W) where boundary pixels are 1 and interior/background are 0.
    """
    H, W = grid.shape
    boundary = torch.zeros(H, W, dtype=torch.long)

    for r in range(H):
        for c in range(W):
            val = int(grid[r, c].item())
            if val == background:
                continue
            # Check 4 neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    boundary[r, c] = 1
                    break
                if int(grid[nr, nc].item()) != val:
                    boundary[r, c] = 1
                    break

    return boundary


def compute_background_mask(grid: torch.Tensor, background: int = 0) -> torch.Tensor:
    """Returns a mask where background pixels are 1 and foreground are 0."""
    return (grid == background).long()


def compute_centroid_map(grid: torch.Tensor, background: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-pixel centroid offset maps.

    For each non-background pixel, stores the offset to its object's centroid.
    Returns (centroid_y, centroid_x) tensors of shape (H, W).
    """
    H, W = grid.shape
    objects = extract_objects(grid, background)

    cy_map = torch.zeros(H, W, dtype=torch.float32)
    cx_map = torch.zeros(H, W, dtype=torch.float32)

    for obj in objects:
        for py, px in obj.pixels:
            cy_map[py, px] = obj.centroid[0] - py
            cx_map[py, px] = obj.centroid[1] - px

    return cy_map, cx_map


def build_feature_map(grid: torch.Tensor, background: int = 0) -> torch.Tensor:
    """Build a 4-channel spatial feature map from a raw grid.

    Channel 0: Raw grid palette (normalized)
    Channel 1: Boundary/edge map
    Channel 2: Background mask
    Channel 3: Object centroid offset (Euclidean distance)

    Returns: (4, H, W) float tensor.
    """
    H, W = grid.shape
    ch0 = grid.float() / 9.0  # Normalize to [0, 1] (ARC colors 0-9)
    ch1 = compute_boundary_mask(grid, background).float()
    ch2 = compute_background_mask(grid, background).float()
    cy_map, cx_map = compute_centroid_map(grid, background)
    ch3 = torch.sqrt(cy_map ** 2 + cx_map ** 2)

    return torch.stack([ch0, ch1, ch2, ch3], dim=0)
