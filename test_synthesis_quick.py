"""Quick test of grammar synthesis on first 10 training puzzles."""
import torch
import time
from soma_mythos_ehra.arc3.adapter import load_tasks_from_dir
from soma_mythos_ehra.arc3.grammar_synthesis import GrammarSynthesizer, SynthesisConfig

tasks = load_tasks_from_dir("ARC-AGI/data/training", limit=10)
solved = 0
for i, task in enumerate(tasks):
    t0 = time.time()
    synth = GrammarSynthesizer(SynthesisConfig(beam_width=20, max_depth=4, timeout=15.0))
    program = synth.synthesize(task)
    elapsed = time.time() - t0

    inp = task.get_train_inputs()[0]
    tgt = task.get_train_outputs()[0]
    if program:
        out = synth.executor.execute(program, inp)
        eq = torch.equal(out, tgt) if out is not None else False
        if eq:
            solved += 1
        tag = "OK" if eq else "WRONG"
        prog_str = program.to_string()[:80]
        print(f"[{i+1}] {task.task_id}: {tag} ({elapsed:.2f}s) {prog_str}")
    else:
        print(f"[{i+1}] {task.task_id}: NO PROGRAM ({elapsed:.2f}s)")

print(f"\nSolved: {solved}/{len(tasks)}")
