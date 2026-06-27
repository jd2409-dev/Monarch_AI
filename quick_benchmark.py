"""Quick test of hybrid solver on a few tasks."""
import sys, time
sys.path.insert(0, '.')
from soma_mythos_ehra.arc3.adapter import ARC3Task
from benchmark_hybrid import HybridSolver
from pathlib import Path

solver = HybridSolver()
task_files = sorted(Path('ARC-AGI/data/training').glob('*.json'))[:30]

solved = 0
for i, tf in enumerate(task_files):
    task = ARC3Task.from_file(tf)
    t0 = time.time()
    result = solver.solve(task)
    dt = time.time() - t0
    acc = result["train_accuracy"]
    method = result.get("method", "none")
    if acc == 1.0:
        solved += 1
    print(f"[{i+1:2d}] {task.task_id}: {acc:.0%} ({method}, {dt:.1f}s)")
    sys.stdout.flush()

print(f"\nSolved: {solved}/{len(task_files)} ({100*solved/len(task_files):.1f}%)")
