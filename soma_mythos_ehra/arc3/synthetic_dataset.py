"""Synthetic Dataset Pipeline — generates unlimited training data from recursive grammar.

The pipeline:
1. Generates random initial grids
2. Samples random programs from the recursive grammar
3. Executes programs via AST executor
4. Emits (input, output) pairs with multi-hot target labels
5. Saves to checkpoints/synthetic_dataset.pt
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from soma_mythos_ehra.arc3.ast_executor import ASTExecutor
from soma_mythos_ehra.arc3.recursive_grammar import (
    NUM_TOKENS,
    encode_multi_hot,
    sample_program,
)
from soma_mythos_ehra.arc3.synthetic_grids import generate_random_grid


def generate_synthetic_dataset(
    num_samples: int = 10000,
    min_size: int = 3,
    max_size: int = 10,
    max_depth: int = 3,
    retries_per_sample: int = 10,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic training dataset from recursive grammar.

    Args:
        num_samples: Number of valid (input, output, labels) triples to generate.
        min_size: Minimum grid dimension.
        max_size: Maximum grid dimension.
        max_depth: Maximum AST depth for sampled programs.
        retries_per_sample: Number of random programs to try before moving on.
        verbose: Print progress.

    Returns:
        X: (N, feature_dim) structural features
        Y: (N, NUM_TOKENS) multi-hot target labels
        grids: (N, 2, max_H, max_W) input/output grid pairs (padded)
    """
    executor = ASTExecutor(background=0)

    X_list = []
    Y_list = []
    grid_list = []

    start_time = time.time()
    generated = 0
    attempts = 0
    max_attempts = num_samples * retries_per_sample

    while generated < num_samples and attempts < max_attempts:
        attempts += 1

        if verbose and generated % 500 == 0 and generated > 0:
            elapsed = time.time() - start_time
            rate = generated / elapsed if elapsed > 0 else 0
            print(f"  [{generated}/{num_samples}] {rate:.1f} samples/sec, "
                  f"{attempts} attempts, {elapsed:.1f}s elapsed")

        # 1. Generate random grid
        grid = generate_random_grid(min_size, max_size)

        # 2. Sample random program
        program = sample_program(max_depth=max_depth)

        # 3. Execute program
        output = executor.execute(program, grid)
        if output is None:
            continue

        # 4. Verify the program actually changed the grid
        if torch.equal(grid, output):
            continue

        # 5. Extract structural features from the pair
        features = _extract_features(grid, output)

        # 6. Encode multi-hot labels
        labels = encode_multi_hot(program)

        X_list.append(features)
        Y_list.append(torch.tensor(labels, dtype=torch.float32))

        # Pad grids to max_size for batching
        pad_h = max_size - grid.shape[0]
        pad_w = max_size - grid.shape[1]
        padded_in = torch.nn.functional.pad(grid, (0, pad_w, 0, pad_h), value=0)
        padded_out = torch.nn.functional.pad(output, (0, pad_w, 0, pad_h), value=0)
        grid_pair = torch.stack([padded_in, padded_out])  # (2, max_H, max_W)
        grid_list.append(grid_pair)

        generated += 1

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nDataset generation complete: {generated} samples from {attempts} attempts in {elapsed:.1f}s")
        print(f"  Rate: {generated/elapsed:.1f} samples/sec")
        print(f"  Success rate: {generated/attempts:.2%}")

    if X_list:
        X = torch.stack(X_list)
        Y = torch.stack(Y_list)
        grids = torch.stack(grid_list)
        return X, Y, grids
    else:
        return (
            torch.empty(0, 32),
            torch.empty(0, NUM_TOKENS),
            torch.empty(0, 2, max_size, max_size),
        )


def _extract_features(inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Extract fixed-size structural features from input/output pair."""
    ih, iw = inp.shape
    oh, ow = out.shape

    height_ratio = oh / ih if ih > 0 else 1.0
    width_ratio = ow / iw if iw > 0 else 1.0
    area_ratio = (oh * ow) / (ih * iw) if (ih * iw) > 0 else 1.0
    shape_changed = float((ih != oh) or (iw != ow))
    is_scaled = float((oh % ih == 0) and (ow % iw == 0) and (height_ratio == width_ratio))

    inp_colors = set(inp.flatten().tolist())
    out_colors = set(out.flatten().tolist())
    num_inp_colors = len(inp_colors)
    num_out_colors = len(out_colors)
    new_colors = len(out_colors - inp_colors)
    lost_colors = len(inp_colors - out_colors)

    inp_density = (inp != 0).float().mean().item()
    out_density = (out != 0).float().mean().item()

    is_rotated = 0.0
    if not shape_changed:
        for k in [1, 2, 3]:
            rotated = torch.rot90(inp, k=k, dims=(0, 1))
            if rotated.shape == out.shape and torch.equal(rotated, out):
                is_rotated = 1.0
                break

    is_flipped = 0.0
    if not shape_changed:
        if torch.equal(inp.flip(1), out) or torch.equal(inp.flip(0), out):
            is_flipped = 1.0

    inp_objects = _count_objects(inp)
    out_objects = _count_objects(out)
    inp_holes = _count_holes(inp)

    return torch.tensor([
        height_ratio, width_ratio, area_ratio, shape_changed, is_scaled,
        float(num_inp_colors), float(num_out_colors), float(new_colors), float(lost_colors),
        1.0,  # consistent (synthetic pairs always are)
        float(inp_objects), float(out_objects),
        inp_density, out_density, is_rotated, is_flipped,
        float(inp_holes),
        0.0, 0.0,  # spread (skipped for speed)
        # Padding to 32
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=torch.float32)


def _count_objects(grid: torch.Tensor) -> int:
    """Count connected components."""
    H, W = grid.shape
    visited = torch.zeros(H, W, dtype=torch.bool)
    count = 0
    for r in range(H):
        for c in range(W):
            if int(grid[r, c].item()) != 0 and not visited[r, c]:
                count += 1
                color = int(grid[r, c].item())
                stack = [(r, c)]
                while stack:
                    y, x = stack.pop()
                    if 0 <= y < H and 0 <= x < W and not visited[y, x]:
                        if int(grid[y, x].item()) == color:
                            visited[y, x] = True
                            stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
    return count


def _count_holes(grid: torch.Tensor) -> int:
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


def save_synthetic_dataset(
    X: torch.Tensor,
    Y: torch.Tensor,
    grids: torch.Tensor,
    path: str,
) -> None:
    """Save synthetic dataset to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": X, "Y": Y, "grids": grids}, path)
    print(f"Saved synthetic dataset: X={X.shape}, Y={Y.shape}, grids={grids.shape} to {path}")


def load_synthetic_dataset(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load synthetic dataset from disk."""
    data = torch.load(path)
    return data["X"], data["Y"], data["grids"]
