"""Check if any puzzles match size/density/centroid sorting."""
import torch
from soma_mythos_ehra.arc3.adapter import load_tasks_from_dir
from soma_mythos_ehra.arc3.objects import extract_objects
from soma_mythos_ehra.arc3.transforms import apply_recolor_by_size, apply_sort_by_density, apply_sort_by_centroid

tasks = load_tasks_from_dir("ARC-AGI/data/training", limit=50)

for task in tasks:
    train_inputs = task.get_train_inputs()
    train_outputs = task.get_train_outputs()

    # Test recolor by size
    all_match = True
    for inp, tgt in zip(train_inputs, train_outputs):
        if inp.shape != tgt.shape:
            all_match = False
            break
        pred = apply_recolor_by_size(inp)
        if not torch.equal(pred, tgt):
            all_match = False
            break
    if all_match:
        print(f"RECOLOR_BY_SIZE: {task.task_id}")

    # Test sort by density
    all_match = True
    for inp, tgt in zip(train_inputs, train_outputs):
        if inp.shape != tgt.shape:
            all_match = False
            break
        pred = apply_sort_by_density(inp)
        if not torch.equal(pred, tgt):
            all_match = False
            break
    if all_match:
        print(f"SORT_BY_DENSITY: {task.task_id}")

    # Test sort by centroid
    for axis in [0, 1]:
        all_match = True
        for inp, tgt in zip(train_inputs, train_outputs):
            if inp.shape != tgt.shape:
                all_match = False
                break
            pred = apply_sort_by_centroid(inp, axis)
            if not torch.equal(pred, tgt):
                all_match = False
                break
        if all_match:
            print(f"SORT_BY_CENTROID (axis={axis}): {task.task_id}")
