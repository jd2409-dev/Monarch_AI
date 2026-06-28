"""Debug 08ed6ac7 train pairs."""
import torch
from soma_mythos_ehra.arc3.adapter import ARC3Task
from soma_mythos_ehra.arc3.objects import connected_component_labeling
from soma_mythos_ehra.arc3.transforms import apply_sequence

task = ARC3Task.from_file('ARC-AGI/data/training/08ed6ac7.json')
train_inputs = task.get_train_inputs()
train_outputs = task.get_train_outputs()

for i, (inp, out) in enumerate(zip(train_inputs, train_outputs)):
    labels = connected_component_labeling(inp)
    match = torch.equal(labels, out)
    print(f"Pair {i}: {inp.shape} match={match}")
    if not match:
        print(f"  Labels: {labels.tolist()}")
        print(f"  Expected: {out.tolist()}")
