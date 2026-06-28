"""Check 08ed6ac7 test pair."""
import torch
from soma_mythos_ehra.arc3.adapter import ARC3Task
from soma_mythos_ehra.arc3.objects import connected_component_labeling

task = ARC3Task.from_file('ARC-AGI/data/training/08ed6ac7.json')
test_input = task.get_test_input()
test_output = task.get_test_output()

print(f"Test input shape: {test_input.shape}")
print(f"Test output shape: {test_output.shape if test_output is not None else None}")

labels = connected_component_labeling(test_input)
print(f"Labels: {labels.tolist()}")
print(f"Expected: {test_output.tolist() if test_output is not None else None}")
print(f"Match: {torch.equal(labels, test_output) if test_output is not None else False}")
