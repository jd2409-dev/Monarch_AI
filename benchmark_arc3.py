"""Benchmark the ARC-AGI 3 solver on multiple training puzzles."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from soma_mythos_ehra.arc3.adapter import ARC3Task, load_tasks_from_dir
from soma_mythos_ehra.arc3.solver import ARC3Config, solve_task
from soma_mythos_ehra.arc3.transforms import apply_sequence


def benchmark(limit: int = 30) -> None:
    """Run solver on training puzzles and report accuracy."""
    tasks = load_tasks_from_dir("ARC-AGI/data/training", limit=limit)
    print(f"Loaded {len(tasks)} tasks")

    solved = 0
    total = 0
    errors = 0
    times = []

    for i, task in enumerate(tasks):
        try:
            t0 = time.time()
            result = solve_task(task, ARC3Config(max_depth=4, simulations=256, seed=42))
            elapsed = time.time() - t0
            times.append(elapsed)

            # Check if test output is correct
            if result["test_output"] is not None:
                expected = task.get_test_output()
                if expected is not None:
                    import torch
                    predicted = torch.tensor(result["test_output"])
                    if torch.equal(predicted, expected):
                        solved += 1
                        status = "OK"
                    else:
                        status = "WRONG"
                else:
                    # No expected output — count as solved if train accuracy is 100%
                    if result["train_accuracy"] == 1.0:
                        solved += 1
                        status = "OK (no test)"
                    else:
                        status = "PARTIAL"
            else:
                status = "NO SOLUTION"

            total += 1
            print(f"  [{i+1}/{len(tasks)}] {task.task_id}: {status} ({elapsed:.3f}s)")

        except Exception as e:
            errors += 1
            total += 1
            print(f"  [{i+1}/{len(tasks)}] {task.task_id}: ERROR - {e}")

    avg_time = sum(times) / len(times) if times else 0
    print(f"\n{'='*60}")
    print(f"Results: {solved}/{total} solved ({solved/total:.1%})")
    print(f"Errors: {errors}")
    print(f"Avg time: {avg_time:.3f}s")
    print(f"Throughput: {1/avg_time:.1f} tasks/sec")


if __name__ == "__main__":
    benchmark(limit=50)
