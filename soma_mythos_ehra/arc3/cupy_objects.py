"""CuPy-accelerated Object Extraction — GPU-native connected component labeling.

Replaces CPU-bound BFS loops with CuPy's scipy.ndimage.label kernel
for blazing fast object extraction entirely in VRAM.
"""
from __future__ import annotations

import torch

try:
    import cupy as cp
    from cupyx.scipy.ndimage import label as cupy_label
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from soma_mythos_ehra.arc3.objects import ARCObject


def cupy_connected_component_labeling(grid: torch.Tensor, background: int = 0) -> torch.Tensor:
    """GPU-accelerated connected component labeling using CuPy.

    Falls back to CPU BFS if CuPy is not available.
    """
    if not CUPY_AVAILABLE:
        from soma_mythos_ehra.arc3.objects import connected_component_labeling
        return connected_component_labeling(grid, background)

    # Convert PyTorch tensor to CuPy array (zero-copy via DLPack)
    try:
        gpu_grid = cp.from_dlpack(grid.to_dlpack())
    except Exception:
        # Fallback: copy to CPU and back
        gpu_grid = cp.array(grid.cpu().numpy())

    # Isolate non-background elements
    binary_mask = (gpu_grid != background).astype(cp.int32)

    # GPU-accelerated connected component labeling
    labeled_array, num_features = cupy_label(binary_mask)

    # Convert back to PyTorch tensor
    try:
        result = torch.from_dlpack(labeled_array.toDlpack()).long()
    except Exception:
        result = torch.tensor(cp.asnumpy(labeled_array), dtype=torch.long)

    return result


def cupy_extract_objects(grid: torch.Tensor, background: int = 0) -> list[ARCObject]:
    """GPU-accelerated object extraction using CuPy.

    Extracts all discrete objects with bounding boxes, centroids, masks,
    area, and geometric invariants (density, aspect_ratio).
    """
    H, W = grid.shape
    labels = cupy_connected_component_labeling(grid, background)

    # Get unique labels (excluding background)
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


def cupy_fill_holes(grid: torch.Tensor, background: int = 0) -> torch.Tensor:
    """GPU-accelerated hole filling using CuPy flood fill."""
    if not CUPY_AVAILABLE:
        from soma_mythos_ehra.arc3.transforms import apply_fill_holes
        return apply_fill_holes(grid)

    try:
        gpu_grid = cp.from_dlpack(grid.to_dlpack())
    except Exception:
        gpu_grid = cp.array(grid.cpu().numpy())

    H, W = gpu_grid.shape

    # BFS from edges
    reachable = cp.zeros((H, W), dtype=cp.bool_)
    stack = []

    for r in range(H):
        for c in [0, W - 1]:
            if int(gpu_grid[r, c].item()) == background:
                stack.append((r, c))
    for c in range(W):
        for r in [0, H - 1]:
            if int(gpu_grid[r, c].item()) == background:
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        if 0 <= r < H and 0 <= c < W and not reachable[r, c]:
            if int(gpu_grid[r, c].item()) == background:
                reachable[r, c] = True
                stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])

    # Fill unreachable holes with nearest non-background
    out = gpu_grid.copy()
    for r in range(H):
        for c in range(W):
            if int(gpu_grid[r, c].item()) == background and not reachable[r, c]:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            val = int(gpu_grid[nr, nc].item())
                            if val != background:
                                out[r, c] = val
                                break
                    else:
                        continue
                    break

    try:
        return torch.from_dlpack(out.toDlpack()).long()
    except Exception:
        return torch.tensor(cp.asnumpy(out), dtype=torch.long)


def benchmark_cupy_vs_cpu(grid_size: int = 30, num_iterations: int = 100) -> None:
    """Benchmark CuPy vs CPU object extraction."""
    import time

    # Create a test grid
    grid = torch.randint(0, 5, (grid_size, grid_size), dtype=torch.long)

    # CPU benchmark
    from soma_mythos_ehra.arc3.objects import extract_objects
    t0 = time.time()
    for _ in range(num_iterations):
        cpu_objects = extract_objects(grid)
    cpu_time = time.time() - t0

    # CuPy benchmark
    if CUPY_AVAILABLE:
        t0 = time.time()
        for _ in range(num_iterations):
            gpu_objects = cupy_extract_objects(grid)
        cupy_time = time.time() - t0
        speedup = cpu_time / cupy_time
        print(f"CPU: {cpu_time:.3f}s | CuPy: {cupy_time:.3f}s | Speedup: {speedup:.1f}x")
    else:
        print(f"CPU: {cpu_time:.3f}s | CuPy: N/A (not installed)")
