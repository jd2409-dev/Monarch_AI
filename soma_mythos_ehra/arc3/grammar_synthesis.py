"""Grammar Synthesis Beam Search — builds programs token-by-token.

Strategy: extract parameters from train pairs, generate parameterized programs,
score by execution, and iteratively extend the best candidates.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import torch

from soma_mythos_ehra.arc3.adapter import ARC3Task
from soma_mythos_ehra.arc3.ast_executor import ASTExecutor
from soma_mythos_ehra.arc3.dataset_generator import extract_structural_features
from soma_mythos_ehra.arc3.jepa_predictor import JEPAStructureRouter
from soma_mythos_ehra.arc3.recursive_grammar import (
    ALL_PRIMITIVES,
    NUM_TOKENS,
    TOKEN_VOCAB,
    PRIMITIVE_MAP,
    ASTNode,
    NodeType,
)


@dataclass
class SynthesisConfig:
    beam_width: int = 20
    max_depth: int = 6
    timeout: float = 10.0


@dataclass
class BeamState:
    ast: ASTNode | None = None
    score: float = 0.0
    correct_pairs: int = 0


class GrammarSynthesizer:
    """Token-level beam search with parameter extraction from train pairs."""

    def __init__(self, config: SynthesisConfig | None = None) -> None:
        self.config = config or SynthesisConfig()
        self.executor = ASTExecutor(background=0)

        self.predictor = JEPAStructureRouter(feature_dim=32, num_templates=NUM_TOKENS)
        self.has_predictor = False
        try:
            self.predictor.load("checkpoints/jepa_synthetic_predictor.pt")
            self.has_predictor = True
        except Exception:
            pass

    def synthesize(self, task: ARC3Task) -> ASTNode | None:
        train_inputs = task.get_train_inputs()
        train_outputs = task.get_train_outputs()
        if not train_inputs:
            return None

        start_time = time.time()

        # Get token priors
        if self.has_predictor:
            features = extract_structural_features(train_inputs[0], train_outputs[0])
            with torch.no_grad():
                token_probs = self.predictor.predict(features.unsqueeze(0)).squeeze(0)
        else:
            token_probs = torch.ones(NUM_TOKENS) / NUM_TOKENS

        prim_scores = {TOKEN_VOCAB[i]: token_probs[i].item() for i in range(NUM_TOKENS)}

        # Extract parameters from train pairs
        params = self._extract_params(train_inputs, train_outputs)

        # Build candidate programs using extracted parameters
        beam = self._generate_candidates(params, prim_scores)

        # Score all candidates
        for state in beam:
            self._score(state, train_inputs, train_outputs)

        # Sort and keep top
        beam.sort(key=lambda s: (s.correct_pairs, s.score), reverse=True)
        beam = beam[:self.config.beam_width]

        # Check for solution
        for state in beam:
            if state.correct_pairs == len(train_inputs):
                return state.ast

        # Iterative extension
        for depth in range(2, self.config.max_depth):
            if time.time() - start_time > self.config.timeout:
                break

            candidates = []
            for state in beam:
                if state.ast is None:
                    continue

                # Prepend primitive
                for prim in ALL_PRIMITIVES:
                    child = self._make_primitive(prim)
                    new_ast = ASTNode(NodeType.COMPOSE, name="compose",
                                      children=[child, state.ast])
                    new_state = BeamState(ast=new_ast, score=state.score + prim_scores.get(prim, 0))
                    self._score(new_state, train_inputs, train_outputs)
                    candidates.append(new_state)
                    if new_state.correct_pairs == len(train_inputs):
                        return new_ast

                # Append primitive
                for prim in ALL_PRIMITIVES:
                    child = self._make_primitive(prim)
                    new_ast = ASTNode(NodeType.COMPOSE, name="compose",
                                      children=[state.ast, child])
                    new_state = BeamState(ast=new_ast, score=state.score + prim_scores.get(prim, 0))
                    self._score(new_state, train_inputs, train_outputs)
                    candidates.append(new_state)
                    if new_state.correct_pairs == len(train_inputs):
                        return new_ast

            candidates.sort(key=lambda s: (s.correct_pairs, s.score), reverse=True)
            beam = candidates[:self.config.beam_width]

        if beam:
            return beam[0].ast
        return None

    def _extract_params(self, inputs: list[torch.Tensor], outputs: list[torch.Tensor]) -> dict:
        """Extract transformation parameters from train pairs."""
        params = {}
        inp, out = inputs[0], outputs[0]
        ih, iw = inp.shape
        oh, ow = out.shape

        # Shape ratio
        if oh % ih == 0 and ow % iw == 0:
            params["scale_y"] = oh // ih
            params["scale_x"] = ow // iw

        # Color mapping (check all pairs for consistency)
        mapping = {}
        consistent = True
        for inp_t, out_t in zip(inputs, outputs):
            for iv, ov in zip(inp_t.flatten().tolist(), out_t.flatten().tolist()):
                if iv in mapping:
                    if mapping[iv] != ov:
                        consistent = False
                        break
                else:
                    mapping[iv] = ov
        if consistent and mapping:
            params["color_map"] = mapping

        # Rotation detection
        if ih == oh and iw == ow:
            for k in [1, 2, 3]:
                rotated = torch.rot90(inp, k=k, dims=(0, 1))
                if torch.equal(rotated, out):
                    params["rotation"] = {1: 90, 2: 180, 3: 270}[k]
                    break

        # Flip detection
        if ih == oh and iw == ow:
            if torch.equal(inp.flip(1), out):
                params["flip"] = "h"
            elif torch.equal(inp.flip(0), out):
                params["flip"] = "v"

        # Interior fill detection (holes surrounded by non-background)
        if ih == oh and iw == ow:
            from soma_mythos_ehra.arc3.transforms import apply_fill_holes
            filled = apply_fill_holes(inp)
            if torch.equal(filled, out):
                params["fill_holes"] = True

        # Recolor detection (same shape, consistent color mapping)
        if ih == oh and iw == ow and "color_map" in params:
            cm = params["color_map"]
            # Check if it's a simple recolor (all non-identity mappings)
            non_identity = {s: d for s, d in cm.items() if s != d}
            if non_identity:
                params["recolor"] = non_identity

        return params

    def _generate_candidates(self, params: dict, prim_scores: dict) -> list[BeamState]:
        """Generate candidate programs using extracted parameters."""
        candidates = []

        # 1. Fill holes
        if params.get("fill_holes"):
            node = ASTNode(NodeType.PRIMITIVE, name="fill_holes", params={})
            candidates.append(BeamState(ast=node, score=1.0))

        # 2. Color map (multi-step recolor)
        if "recolor" in params:
            for src, dst in params["recolor"].items():
                node = ASTNode(NodeType.PRIMITIVE, name="recolor",
                               params={"src": src, "dst": dst})
                candidates.append(BeamState(ast=node, score=1.0))

        # 3. Rotation
        if "rotation" in params:
            angle = params["rotation"]
            prim = {90: "rotate_90", 180: "rotate_180", 270: "rotate_270"}[angle]
            node = self._make_primitive(prim)
            candidates.append(BeamState(ast=node, score=1.0))

        # 4. Flip
        if "flip" in params:
            prim = "flip_h" if params["flip"] == "h" else "flip_v"
            node = self._make_primitive(prim)
            candidates.append(BeamState(ast=node, score=1.0))

        # 5. Scale/tile with exact factors
        if "scale_y" in params:
            sy, sx = params["scale_y"], params["scale_x"]
            if sy > 1 or sx > 1:
                # Try scale
                if sy == sx:
                    node = ASTNode(NodeType.PRIMITIVE, name="scale",
                                   params={"factor": sy})
                    candidates.append(BeamState(ast=node, score=1.0))
                # Try tile
                node = ASTNode(NodeType.PRIMITIVE, name="tile",
                               params={"reps_h": sy, "reps_w": sx})
                candidates.append(BeamState(ast=node, score=1.0))

        # 6. Compose recolor + tile
        if "recolor" in params and "scale_y" in params:
            recolor_nodes = []
            for src, dst in params["recolor"].items():
                recolor_nodes.append(ASTNode(NodeType.PRIMITIVE, name="recolor",
                                             params={"src": src, "dst": dst}))
            if recolor_nodes:
                cm_chain = recolor_nodes[0] if len(recolor_nodes) == 1 else ASTNode(
                    NodeType.COMPOSE, name="compose", children=recolor_nodes)
                sy, sx = params["scale_y"], params["scale_x"]
                tile_node = ASTNode(NodeType.PRIMITIVE, name="tile",
                                    params={"reps_h": sy, "reps_w": sx})
                # Try tile then recolor
                node = ASTNode(NodeType.COMPOSE, name="compose",
                               children=[tile_node, cm_chain])
                candidates.append(BeamState(ast=node, score=2.0))
                # Try recolor then tile
                node = ASTNode(NodeType.COMPOSE, name="compose",
                               children=[cm_chain, tile_node])
                candidates.append(BeamState(ast=node, score=2.0))

        # 7. Compose fill_holes + recolor
        if params.get("fill_holes") and "recolor" in params:
            fill_node = ASTNode(NodeType.PRIMITIVE, name="fill_holes", params={})
            recolor_nodes = []
            for src, dst in params["recolor"].items():
                recolor_nodes.append(ASTNode(NodeType.PRIMITIVE, name="recolor",
                                             params={"src": src, "dst": dst}))
            if recolor_nodes:
                cm_chain = recolor_nodes[0] if len(recolor_nodes) == 1 else ASTNode(
                    NodeType.COMPOSE, name="compose", children=recolor_nodes)
                node = ASTNode(NodeType.COMPOSE, name="compose",
                               children=[fill_node, cm_chain])
                candidates.append(BeamState(ast=node, score=2.0))

        # 8. All single primitives
        for prim in ALL_PRIMITIVES:
            node = self._make_primitive(prim)
            candidates.append(BeamState(ast=node, score=prim_scores.get(prim, 0)))

        # 9. All compose pairs
        for p1 in ALL_PRIMITIVES:
            for p2 in ALL_PRIMITIVES:
                if p1 == p2:
                    continue
                left = self._make_primitive(p1)
                right = self._make_primitive(p2)
                node = ASTNode(NodeType.COMPOSE, name="compose", children=[left, right])
                score = prim_scores.get(p1, 0) + prim_scores.get(p2, 0)
                candidates.append(BeamState(ast=node, score=score))

        # 10. Random programs from grammar
        for _ in range(100):
            from soma_mythos_ehra.arc3.recursive_grammar import sample_program
            node = sample_program(max_depth=3)
            candidates.append(BeamState(ast=node, score=0))

        return candidates

    def _score(self, state: BeamState, inputs: list[torch.Tensor], targets: list[torch.Tensor]) -> None:
        if state.ast is None:
            return
        correct = 0
        for inp, tgt in zip(inputs, targets):
            pred = self.executor.execute(state.ast, inp)
            if pred is not None and torch.equal(pred, tgt):
                correct += 1
        state.correct_pairs = correct
        state.score += correct * 10.0

    def _make_primitive(self, token: str) -> ASTNode:
        prim_name, params = PRIMITIVE_MAP[token]
        if prim_name == "recolor":
            params = {"src": random.randint(1, 5), "dst": random.randint(1, 5)}
        elif prim_name == "flood_fill":
            params = {"y": random.randint(0, 7), "x": random.randint(0, 7), "color": random.randint(1, 5)}
        elif prim_name == "shift":
            params = {"dy": random.choice([-2, -1, 1, 2]), "dx": random.choice([-2, -1, 1, 2])}
        elif prim_name == "wrap":
            params = {"shift": random.randint(1, 3), "axis": random.choice([0, 1])}
        return ASTNode(NodeType.PRIMITIVE, name=prim_name, params=params)


def synthesis_search_task(task: ARC3Task, config: SynthesisConfig | None = None) -> dict:
    synth = GrammarSynthesizer(config)
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
        correct, _ = synth.executor.execute_on_pairs(program, train_inputs, train_outputs)
        result["train_accuracy"] = correct / len(train_inputs) if train_inputs else 0.0

        test_input = task.get_test_input()
        if test_input is not None:
            test_output = synth.executor.execute(program, test_input)
            if test_output is not None:
                result["test_output"] = test_output.tolist()

    return result
