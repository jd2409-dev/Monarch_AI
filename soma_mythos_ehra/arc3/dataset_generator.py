"""ARC Training Dataset Generator — creates (features, template_labels) pairs.

For each ARC puzzle, executes all templates and records which ones solve it,
then extracts structural features for training the JEPA structure predictor.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from soma_mythos_ehra.arc3.dsl_kernel import DSLKernel
from soma_mythos_ehra.arc3.template_library import build_template_library


def extract_structural_features(inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Extract fixed-size structural features from input/output pair.

    Returns a feature vector of size 32.
    """
    ih, iw = inp.shape
    oh, ow = out.shape

    # Shape features
    height_ratio = oh / ih if ih > 0 else 1.0
    width_ratio = ow / iw if iw > 0 else 1.0
    area_ratio = (oh * ow) / (ih * iw) if (ih * iw) > 0 else 1.0
    shape_changed = float((ih != oh) or (iw != ow))
    is_scaled = float((oh % ih == 0) and (ow % iw == 0) and (height_ratio == width_ratio))

    # Color features
    inp_colors = set(inp.flatten().tolist())
    out_colors = set(out.flatten().tolist())
    num_inp_colors = len(inp_colors)
    num_out_colors = len(out_colors)
    new_colors = len(out_colors - inp_colors)
    lost_colors = len(inp_colors - out_colors)

    # Check consistent mapping
    mapping = {}
    consistent = True
    min_h, min_w = min(ih, oh), min(iw, ow)
    for r in range(min_h):
        for c in range(min_w):
            iv, ov = int(inp[r, c].item()), int(out[r, c].item())
            if iv in mapping:
                if mapping[iv] != ov:
                    consistent = False
            else:
                mapping[iv] = ov

    # Object features
    inp_objects = count_objects(inp)
    out_objects = count_objects(out)

    # Density
    inp_density = (inp != 0).float().mean().item()
    out_density = (out != 0).float().mean().item()

    # Rotation detection
    is_rotated = 0.0
    if not shape_changed:
        for k in [1, 2, 3]:
            rotated = torch.rot90(inp, k=k, dims=(0, 1))
            if rotated.shape == out.shape and torch.equal(rotated, out):
                is_rotated = 1.0
                break

    # Flip detection
    is_flipped = 0.0
    if not shape_changed:
        if torch.equal(inp.flip(1), out) or torch.equal(inp.flip(0), out):
            is_flipped = 1.0

    # Holes
    inp_holes = count_holes(inp)

    # Position variance (are objects scattered?)
    inp_obj = extract_object_positions(inp)
    out_obj = extract_object_positions(out)
    inp_spread = position_spread(inp_obj)
    out_spread = position_spread(out_obj)

    features = torch.tensor([
        height_ratio, width_ratio, area_ratio, shape_changed, is_scaled,
        float(num_inp_colors), float(num_out_colors), float(new_colors), float(lost_colors),
        float(consistent), float(inp_objects), float(out_objects),
        inp_density, out_density, is_rotated, is_flipped,
        float(inp_holes), inp_spread, out_spread,
        # Padding to 32
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=torch.float32)

    return features


def count_objects(grid: torch.Tensor) -> int:
    """Count connected components."""
    H, W = grid.shape
    visited = torch.zeros(H, W, dtype=torch.bool)
    count = 0
    for r in range(H):
        for c in range(W):
            if int(grid[r, c].item()) != 0 and not visited[r, c]:
                count += 1
                stack = [(r, c)]
                while stack:
                    y, x = stack.pop()
                    if 0 <= y < H and 0 <= x < W and not visited[y, x]:
                        if int(grid[y, x].item()) == int(grid[r, c].item()):
                            visited[y, x] = True
                            stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
    return count


def count_holes(grid: torch.Tensor) -> int:
    """Count interior holes."""
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


def extract_object_positions(grid: torch.Tensor) -> list[tuple[float, float]]:
    """Extract centroids of objects."""
    H, W = grid.shape
    visited = torch.zeros(H, W, dtype=torch.bool)
    positions = []
    for r in range(H):
        for c in range(W):
            if int(grid[r, c].item()) != 0 and not visited[r, c]:
                color = int(grid[r, c].item())
                pixels = []
                stack = [(r, c)]
                while stack:
                    y, x = stack.pop()
                    if 0 <= y < H and 0 <= x < W and not visited[y, x]:
                        if int(grid[y, x].item()) == color:
                            visited[y, x] = True
                            pixels.append((y, x))
                            stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
                if pixels:
                    cy = sum(p[0] for p in pixels) / len(pixels)
                    cx = sum(p[1] for p in pixels) / len(pixels)
                    positions.append((cy, cx))
    return positions


def position_spread(positions: list[tuple[float, float]]) -> float:
    """Compute spread of object positions."""
    if len(positions) < 2:
        return 0.0
    cy_mean = sum(p[0] for p in positions) / len(positions)
    cx_mean = sum(p[1] for p in positions) / len(positions)
    variance = sum((p[0] - cy_mean)**2 + (p[1] - cx_mean)**2 for p in positions) / len(positions)
    return variance ** 0.5


def generate_dataset(data_dir: str | Path, limit: int = 400) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate training dataset from ARC puzzles.

    Returns:
        X: (N, 32) feature vectors
        Y: (N, T) template label vectors where T = number of templates
    """
    data_dir = Path(data_dir)
    json_files = sorted(data_dir.glob("*.json"))[:limit]

    templates = build_template_library()
    kernel = DSLKernel(background=0)
    num_templates = len(templates)

    print(f"Generating dataset from {len(json_files)} puzzles with {num_templates} templates")

    X_list = []
    Y_list = []

    start_time = time.time()
    for i, json_file in enumerate(json_files):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(json_files)}] {elapsed:.1f}s elapsed")

        try:
            with open(json_file) as f:
                data = json.load(f)

            train_pairs = data.get("train", [])
            if not train_pairs:
                continue

            # Use first train pair for features
            pair = train_pairs[0]
            inp = torch.tensor(pair["input"], dtype=torch.long)
            out = torch.tensor(pair["output"], dtype=torch.long)

            # Extract features
            features = extract_structural_features(inp, out)

            # Test all templates
            labels = torch.zeros(num_templates)
            for j, (name, prog) in enumerate(templates):
                correct, _ = kernel.execute_on_pairs(prog, [inp], [out])
                if correct == 1:
                    labels[j] = 1.0

            # Only add if at least one template solves it
            if labels.sum() > 0:
                X_list.append(features)
                Y_list.append(labels)

        except Exception:
            continue

    elapsed = time.time() - start_time
    print(f"Dataset generation complete: {len(X_list)} samples in {elapsed:.1f}s")

    if X_list:
        X = torch.stack(X_list)
        Y = torch.stack(Y_list)
        return X, Y
    else:
        return torch.empty(0, 32), torch.empty(0, num_templates)


def save_dataset(X: torch.Tensor, Y: torch.Tensor, path: str) -> None:
    """Save dataset to disk."""
    torch.save({"X": X, "Y": Y}, path)
    print(f"Saved dataset: X={X.shape}, Y={Y.shape} to {path}")


def load_dataset(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load dataset from disk."""
    data = torch.load(path)
    return data["X"], data["Y"]
