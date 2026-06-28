"""Verify which templates solve which puzzles with test accuracy."""
import torch
from soma_mythos_ehra.arc3.adapter import load_tasks_from_dir
from soma_mythos_ehra.arc3.dsl_kernel import DSLKernel
from soma_mythos_ehra.arc3.template_library import build_template_library

templates = build_template_library()
kernel = DSLKernel(background=0)
tasks = load_tasks_from_dir("ARC-AGI/data/training", limit=50)

for task in tasks:
    train_inputs = task.get_train_inputs()
    train_outputs = task.get_train_outputs()
    test_input = task.get_test_input()
    test_output = task.get_test_output()
    for name, prog in templates:
        correct, _ = kernel.execute_on_pairs(prog, train_inputs, train_outputs)
        if correct == len(train_inputs):
            if test_input is not None and test_output is not None:
                pred = kernel.execute(prog, test_input)
                match = torch.equal(pred, test_output) if pred is not None else False
                status = "OK" if match else "WRONG"
                print(f"{task.task_id}: {name} -> test={status}")
            else:
                print(f"{task.task_id}: {name} -> no test")
