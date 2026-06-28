"""Check which templates solve which puzzles."""
import torch
from soma_mythos_ehra.arc3.adapter import load_tasks_from_dir
from soma_mythos_ehra.arc3.dsl_kernel import DSLKernel
from soma_mythos_ehra.arc3.template_library import build_template_library

templates = build_template_library()
kernel = DSLKernel(background=0)
tasks = load_tasks_from_dir("ARC-AGI/data/training", limit=50)

print(f"Testing {len(templates)} templates on {len(tasks)} puzzles")

solved_by_template = {}
for task in tasks:
    train_inputs = task.get_train_inputs()
    train_outputs = task.get_train_outputs()
    for name, prog in templates:
        correct, _ = kernel.execute_on_pairs(prog, train_inputs, train_outputs)
        if correct == len(train_inputs):
            if name not in solved_by_template:
                solved_by_template[name] = []
            solved_by_template[name].append(task.task_id)

print(f"\nTemplates that solve puzzles:")
for name, task_ids in sorted(solved_by_template.items()):
    print(f"  {name}: {len(task_ids)} puzzles")
