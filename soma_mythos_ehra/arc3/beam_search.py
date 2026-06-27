"""Guided Beam Search Synthesizer — uses classifier priors and execution pruning.

Replaces MCTS with a prioritized beam search that:
1. Uses the 31-token grammar predictor to score templates via dot product
2. Executes candidates on train pairs to verify correctness
3. Prunes invalid branches immediately
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from soma_mythos_ehra.arc3.adapter import ARC3Task
from soma_mythos_ehra.arc3.dataset_generator import extract_structural_features
from soma_mythos_ehra.arc3.dsl_grammar import DSLNode, PRIMITIVES
from soma_mythos_ehra.arc3.dsl_kernel import DSLKernel
from soma_mythos_ehra.arc3.jepa_predictor import JEPA_template_router
from soma_mythos_ehra.arc3.spatial_classifier import SpatialDiffClassifier
from soma_mythos_ehra.arc3.template_library import build_template_library


@dataclass
class BeamConfig:
    beam_width: int = 12
    max_depth: int = 4
    timeout: float = 8.0
    top_k_primitives: int = 8
    seed: int = 42


@dataclass
class BeamCandidate:
    """A candidate program in the beam search."""
    program: DSLNode
    score: float = 0.0
    correct_pairs: int = 0
    verified: bool = False


class GuidedBeamSynthesizer:
    """Beam search synthesizer with 104-class template predictor guidance."""

    def __init__(self, config: BeamConfig | None = None) -> None:
        self.config = config or BeamConfig()
        self.classifier = SpatialDiffClassifier()
        self.kernel = DSLKernel(background=0)
        self.templates = build_template_library()

        # Load the 104-class template predictor
        self.template_predictor = JEPA_template_router(feature_dim=32, num_classes=len(self.templates))
        self.has_predictor = False
        try:
            self.template_predictor.load("checkpoints/jepa_template_predictor_104.pt")
            self.has_predictor = True
        except Exception:
            pass

    def synthesize(self, task: ARC3Task) -> DSLNode | None:
        """Search for a DSL program that solves the task."""
        train_inputs = task.get_train_inputs()
        train_outputs = task.get_train_outputs()
        if not train_inputs:
            return None

        start_time = time.time()

        # Step 1: Score templates using 104-class predictor (Method 2)
        if self.has_predictor and train_inputs:
            features = extract_structural_features(train_inputs[0], train_outputs[0])
            with torch.no_grad():
                top_indices, top_probs = self.template_predictor.predict_top_k(features.unsqueeze(0), k=len(self.templates))
            # Flatten sorted indices by probability
            sorted_indices = top_indices.squeeze(0)
        else:
            sorted_indices = torch.arange(len(self.templates))

        # Step 2: Try templates in predicted order
        for idx in sorted_indices:
            if time.time() - start_time > self.config.timeout:
                break
            i = idx.item()
            if i < len(self.templates):
                name, prog = self.templates[i]
                correct, _ = self.kernel.execute_on_pairs(prog, train_inputs, train_outputs)
                if correct == len(train_inputs):
                    return prog

        # Step 3: Fallback to beam search with primitives
        primitive_probs = self.classifier.predict_batch(train_inputs, train_outputs)
        top_k = min(self.config.top_k_primitives, len(PRIMITIVES))
        top_indices_prim = torch.argsort(primitive_probs, descending=True)[:top_k]
        top_names = [list(PRIMITIVES.keys())[i] for i in top_indices_prim if i < len(PRIMITIVES)]

        beam = []
        for name in top_names:
            prim = PRIMITIVES.get(name)
            if prim is None:
                continue
            prog = DSLNode(primitive=name, params=self._default_params(name))
            correct, _ = self.kernel.execute_on_pairs(prog, train_inputs, train_outputs)
            score = correct + float(primitive_probs[list(PRIMITIVES.keys()).index(name)]) if name in PRIMITIVES else 0
            candidate = BeamCandidate(program=prog, score=score, correct_pairs=correct)
            beam.append(candidate)
            if correct == len(train_inputs):
                return prog

        # Beam search
        for depth in range(1, self.config.max_depth):
            if time.time() - start_time > self.config.timeout:
                break
            candidates = []
            for cand in beam:
                if time.time() - start_time > self.config.timeout:
                    break
                for name in top_names:
                    if time.time() - start_time > self.config.timeout:
                        break
                    prim = PRIMITIVES.get(name)
                    if prim is None:
                        continue
                    new_prog = self._extend_program(cand.program, name)
                    if new_prog is None:
                        continue
                    correct, _ = self.kernel.execute_on_pairs(new_prog, train_inputs, train_outputs)
                    if correct <= cand.correct_pairs:
                        continue
                    prim_idx = list(PRIMITIVES.keys()).index(name) if name in PRIMITIVES else 0
                    score = correct + float(primitive_probs[prim_idx]) * 0.5
                    candidate = BeamCandidate(program=new_prog, score=score, correct_pairs=correct)
                    candidates.append(candidate)
                    if correct == len(train_inputs):
                        return new_prog
            candidates.sort(key=lambda c: c.score, reverse=True)
            beam = candidates[:self.config.beam_width]

        if beam:
            beam.sort(key=lambda c: c.score, reverse=True)
            return beam[0].program
        return None

    def _extend_program(self, program: DSLNode, prim_name: str) -> DSLNode | None:
        """Extend a program by adding a new primitive."""
        prim = PRIMITIVES.get(prim_name)
        if prim is None:
            return None

        new_node = DSLNode(primitive=prim_name, params=self._default_params(prim_name))

        if program.primitive == "compose" and not program.children:
            return new_node

        if program.primitive != "compose":
            return DSLNode(
                primitive="compose",
                children=[program, new_node],
            )

        if program.depth() >= self.config.max_depth:
            return None

        return DSLNode(
            primitive="compose",
            children=program.children + [new_node],
        )

    def _default_params(self, prim_name: str) -> dict:
        """Get default parameters for a primitive."""
        defaults = {
            "rotate": {"angle": 90},
            "flip": {"axis": "h"},
            "scale": {"factor": 2},
            "tile": {"reps_h": 2, "reps_w": 2},
            "shift": {"dy": 1, "dx": 0},
            "wrap": {"shift": 1, "axis": 0},
            "recolor": {"src": 0, "dst": 1},
            "recolor_objects": {"color": 1},
            "flood_fill": {"y": 0, "x": 0, "color": 1},
            "filter_by_area": {"mode": "max"},
            "filter_by_density": {"mode": "max"},
            "filter_by_color": {"color": 1},
            "filter_by_size": {"mode": "largest"},
            "filter_by_position": {"region": "center"},
            "take_n": {"n": 1, "order": "first"},
            "sort_by_position": {"axis": 0},
            "sort_by_area": {"reverse": True},
            "object_at": {"position": "center"},
        }
        return defaults.get(prim_name, {})


def beam_search_task(task: ARC3Task, config: BeamConfig | None = None) -> dict:
    """Solve an ARC task using guided beam search."""
    synth = GuidedBeamSynthesizer(config)
    train_inputs = task.get_train_inputs()
    train_outputs = task.get_train_outputs()

    program = synth.synthesize(task)

    result = {
        "task_id": task.task_id,
        "program": program,
        "program_str": program.to_string() if program else None,
        "test_output": None,
        "train_accuracy": 0.0,
    }

    if program is not None:
        correct, _ = synth.kernel.execute_on_pairs(program, train_inputs, train_outputs)
        result["train_accuracy"] = correct / len(train_inputs) if train_inputs else 0.0

        test_input = task.get_test_input()
        if test_input is not None:
            test_output = synth.kernel.execute(program, test_input)
            if test_output is not None:
                result["test_output"] = test_output.tolist()

    return result
