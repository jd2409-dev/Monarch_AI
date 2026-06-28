"""Test specific puzzles."""
import torch
from soma_mythos_ehra.arc3.adapter import ARC3Task
from soma_mythos_ehra.arc3.solver import solve_task, ARC3Config

task = ARC3Task.from_file('ARC-AGI/data/training/08ed6ac7.json')
result = solve_task(task, ARC3Config(max_depth=4, simulations=256, seed=42))
print(f"Task: {task.task_id}")
print(f"Solution type: {type(result['solution'])}")
print(f"Train accuracy: {result['train_accuracy']:.1%}")
if result['test_output'] is not None:
    print(f"Test output: {result['test_output']}")
    expected = task.get_test_output()
    if expected is not None:
        import torch
        predicted = torch.tensor(result['test_output'])
        print(f"Expected: {expected.tolist()}")
        print(f"Match: {torch.equal(predicted, expected)}")
