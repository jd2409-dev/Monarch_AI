"""Test the ARC-AGI 3 solver on a simple puzzle."""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from soma_mythos_ehra.arc3.adapter import ARC3Task
from soma_mythos_ehra.arc3.solver import ARC3Config, solve_task
from soma_mythos_ehra.arc3.transforms import apply_sequence


def test_0d3d703e():
    """Test on the color permutation puzzle."""
    task = ARC3Task.from_file("ARC-AGI/data/training/0d3d703e.json")
    print(f"Task: {task}")
    print(f"  Train pairs: {len(task.train_pairs)}")
    print(f"  Test pairs: {len(task.test_pairs)}")
    print(f"  Num colors: {task.num_colors()}")

    # Show train pairs
    for i, pair in enumerate(task.train_pairs):
        print(f"\n  Train {i}: input={pair['input']}")
        print(f"           output={pair['output']}")

    print(f"\n  Test input: {task.test_pairs[0]['input']}")
    print(f"  Expected:   {task.test_pairs[0].get('output', 'N/A')}")

    # Solve
    config = ARC3Config(max_depth=8, simulations=1024, seed=42)
    t0 = time.time()
    result = solve_task(task, config)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Solution found in {elapsed:.3f}s")
    print(f"  Sequence: {result['solution']}")
    print(f"  Train accuracy: {result['train_accuracy']:.1%}")

    if result["test_output"] is not None:
        print(f"  Test output: {result['test_output']}")

        # Verify against expected
        expected = task.get_test_output()
        if expected is not None:
            import torch
            predicted = torch.tensor(result["test_output"])
            if torch.equal(predicted, expected):
                print("\n  [CORRECT] Test output matches expected.")
            else:
                print("\n  [INCORRECT] Mismatch.")
                print(f"    Expected: {expected.tolist()}")
                print(f"    Got:      {result['test_output']}")
    else:
        print("  No solution found.")

    # Also verify by applying sequence directly
    if result["solution"] is not None:
        test_input = task.get_test_input()
        applied = apply_sequence(test_input, result["solution"])
        print(f"\n  Direct apply result: {applied.tolist()}")
        expected = task.get_test_output()
        if expected is not None and torch.equal(applied, expected):
            print("  [VERIFIED] Direct application matches expected output!")
        return result

    return result


def test_color_map_heuristic():
    """Test the color map learning heuristic directly."""
    task = ARC3Task.from_file("ARC-AGI/data/training/0d3d703e.json")
    solver_config = ARC3Config()
    from soma_mythos_ehra.arc3.solver import ARC3Solver
    solver = ARC3Solver(solver_config)

    inputs = task.get_train_inputs()
    outputs = task.get_train_outputs()

    color_map = solver._learn_color_map(inputs, outputs)
    print(f"\nLearned color map: {color_map}")

    # Apply to test
    if color_map:
        test_input = task.get_test_input()
        mapped = test_input.clone()
        for src, dst in color_map.items():
            mapped[test_input == src] = dst
        print(f"Mapped test input: {mapped.tolist()}")

        expected = task.get_test_output()
        if expected is not None:
            import torch
            if torch.equal(mapped, expected):
                print("[OK] Color map solves the puzzle!")
            else:
                print("[FAIL] Color map doesn't solve it.")
    return color_map


if __name__ == "__main__":
    print("="*60)
    print("ARC-AGI 3 Solver Test")
    print("="*60)

    # First test the simple heuristic
    print("\n--- Test 1: Color Map Heuristic ---")
    test_color_map_heuristic()

    # Then full MCTS solver
    print("\n--- Test 2: Full MCTS Solver ---")
    test_0d3d703e()
