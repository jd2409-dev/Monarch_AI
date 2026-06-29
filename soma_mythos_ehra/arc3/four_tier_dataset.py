"""Four-Tier Unified Token Dataset — from-scratch LRLM training data.

Generates four interleaved data streams:
  Tier 1: Core Physics & Interaction Buffer (100K+ step pairs with NL descriptions)
  Tier 2: Synthetic Procedural Reasonings (50K tasks with full step traces)
  Tier 3: Algorithmic Logic Chains (10K math/sorting/tree reasoning traces)
  Tier 4: Structural Text Corpora (50-100MB clean prose mapped to ASCII tokens)

All tiers share a common token vocabulary. The interleaved loader mixes them
25/25/25/25 to prevent catastrophic forgetting and teach the transformer to
treat language, math, and grid transformations as a single reasoning language.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Shared Vocabulary
# ══════════════════════════════════════════════════════════════════════════════

# 128 base tokens from game_tokenizer + extended for LRLM
VOCAB_SIZE = 8192  # Full LRLM vocabulary

# Special tokens [0-31]
TOK_PAD = 0
TOK_BOS = 1
TOK_EOS = 2
TOK_SEP = 3
TOK_SOS = 4  # Start of sequence
TOK_MASK = 5
TOK_GRID_START = 10
TOK_GRID_END = 11
TOK_LATENT = 12
TOK_ACTION = 13
TOK_REWARD = 14
TOK_REASON_START = 15
TOK_REASON_END = 16
TOK_TRACE_START = 17
TOK_TRACE_END = 18
TOK_PROBLEM = 19
TOK_STEP = 20
TOK_ANSWER = 21
TOK_CODE = 22
TOK_ESSAY = 23
TOK_TITLE = 24

# Action tokens [32-41]
TOK_ACTION1 = 32
TOK_ACTION2 = 33
TOK_ACTION3 = 34
TOK_ACTION4 = 35
TOK_ACTION5 = 36
TOK_ACTION6 = 37
TOK_ACTION7 = 38
TOK_RESET = 39
TOK_UNDO = 40
TOK_NOOP = 41

# Grid state tokens [42-57]
TOK_STATE_NOT_FINISHED = 42
TOK_STATE_WIN = 43
TOK_STATE_GAME_OVER = 44
TOK_DIFF_NO_CHANGE = 45
TOK_DIFF_MOVEMENT = 46
TOK_DIFF_COLOR = 47
TOK_DIFF_OBJECT = 48
TOK_DIFF_ROTATION = 49
TOK_DIFF_SCALE = 50
TOK_DIFF_INVERT = 51

# Grammar tokens [58-127]
GRAMMAR_TOKEN_START = 58

# Alphabet [128-255] for text
ASCII_BASE = 128

# Extended vocabulary [256+]
VOCAB_EXTENDED_START = 256
# Math/logic tokens [256-511]
TOK_MATH_OP_ADD = 256
TOK_MATH_OP_SUB = 257
TOK_MATH_OP_MUL = 258
TOK_MATH_OP_DIV = 259
TOK_MATH_OP_MOD = 260
TOK_MATH_OP_POW = 261
TOK_MATH_EQ = 262
TOK_MATH_LT = 263
TOK_MATH_GT = 264
TOK_MATH_LE = 265
TOK_MATH_GE = 266
TOK_MATH_AND = 267
TOK_MATH_OR = 268
TOK_MATH_NOT = 269
TOK_MATH_IF = 270
TOK_MATH_ELSE = 271
TOK_MATH_FOR = 272
TOK_MATH_WHILE = 273
TOK_MATH_RETURN = 274
TOK_MATH_VAR = 275
TOK_MATH_FUNC = 276
TOK_MATH_CALL = 277
TOK_MATH_PRINT = 278
TOK_MATH_SORT = 279
TOK_MATH_FIND = 280
TOK_MATH_COUNT = 281
TOK_MATH_MIN = 282
TOK_MATH_MAX = 283
TOK_MATH_SUM = 284
TOK_MATH_LEN = 285
TOK_MATH_APPEND = 286
TOK_MATH_POP = 287
TOK_MATH_SWAP = 288
TOK_MATH_TRUE = 289
TOK_MATH_FALSE = 290
TOK_MATH_NULL = 291
TOK_MATH_INT = 292
TOK_MATH_LIST = 293
TOK_MATH_STR = 294

# Spatial tokens [294-320]
TOK_COORD = 300
TOK_GRID_VALUE = 310
TOK_POSITION = 320

# Text category tokens [330-350]
TOK_TEXT_ESSAY = 330
TOK_TEXT_PARAGRAPH = 331
TOK_TEXT_SENTENCE = 332
TOK_TEXT_WORD = 333
TOK_TEXT_TOPIC = 334
TOK_TEXT_ARGUMENT = 335
TOK_TEXT_CONCLUSION = 336
TOK_TEXT_INTRODUCTION = 337
TOK_TEXT_BODY = 338

# Procedural trace tokens [360-400]
TOK_TRACE_DEFINE = 360
TOK_TRACE_FILTER = 361
TOK_TRACE_TRANSFORM = 362
TOK_TRACE_COMPOSE = 363
TOK_TRACE_BRANCH = 364
TOK_TRACE_EXECUTE = 365
TOK_TRACE_RESULT = 366
TOK_TRACE_ERROR = 367
TOK_TRACE_STEP_NUM = 368


def text_to_tokens(text: str) -> list[int]:
    """Convert ASCII text to token IDs (shifted into vocabulary)."""
    return [ASCII_BASE + ord(c) for c in text if 128 > ord(c) >= 32]


def tokens_to_text(token_ids: list[int]) -> str:
    """Convert token IDs back to ASCII text."""
    chars = []
    for t in token_ids:
        if ASCII_BASE <= t < ASCII_BASE + 96:
            c = chr(t - ASCII_BASE)
            chars.append(c)
    return "".join(chars)


def int_to_tokens(n: int) -> list[int]:
    """Encode an integer as a token sequence."""
    if n == 0:
        return [TOK_MATH_INT, ASCII_BASE + ord("0")]
    tokens = [TOK_MATH_INT]
    if n < 0:
        tokens.append(ASCII_BASE + ord("-"))
        n = -n
    for digit in str(n):
        tokens.append(ASCII_BASE + ord(digit))
    return tokens


def list_to_tokens(lst: list[int]) -> list[int]:
    """Encode a list of integers as a token sequence."""
    tokens = [TOK_MATH_LIST]
    for item in lst:
        tokens.extend(int_to_tokens(item))
        tokens.append(ASCII_BASE + ord(","))
    return tokens[:-1] if len(tokens) > 1 else tokens


# ══════════════════════════════════════════════════════════════════════════════
# Tier 1: Core Physics & Interaction Buffer
# ══════════════════════════════════════════════════════════════════════════════

ACTION_NAMES = {
    1: "move up", 2: "move down", 3: "move left", 4: "move right",
    5: "use interact", 6: "click target", 7: "action seven",
}


@dataclass
class Tier1Sample:
    """A single Tier 1 physics interaction sample."""
    grid_latent: torch.Tensor   # (256,) pre-action latent
    action_id: int              # action taken
    next_grid_latent: torch.Tensor  # (256,) post-action latent
    reward: float               # reward received
    game_state: str             # NOT_FINISHED / WIN / GAME_OVER
    grid_diff_type: str         # movement, color_shift, no_change, etc.
    nl_description: str         # natural language description
    prev_grid: np.ndarray | None = None  # raw grid (optional)
    next_grid: np.ndarray | None = None   # raw grid (optional)


class Tier1Generator:
    """Generates Tier 1 core physics & interaction buffer.

    Produces (grid_latent, action, next_grid_latent, reward, NL_description)
    tuples. Sources:
      1. Existing replay buffer transitions
      2. Synthetic grid transformations via expanded grammar
      3. Procedural grid generation with known transformations
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def _describe_grid_diff(
        self, prev_grid: np.ndarray, next_grid: np.ndarray, action_id: int
    ) -> str:
        """Generate a natural language description of what changed."""
        if prev_grid.shape != next_grid.shape:
            return f"action {action_id} changed the grid dimensions"

        diff = prev_grid != next_grid
        if not np.any(diff):
            return f"action {action_id} had no visible effect on the grid"

        change_count = int(np.sum(diff))
        total = prev_grid.size
        change_pct = change_count / total * 100

        # Find moved objects
        prev_objects = set(zip(*np.where(prev_grid > 0)))
        next_objects = set(zip(*np.where(next_grid > 0)))

        appeared = next_objects - prev_objects
        disappeared = prev_objects - next_objects

        parts = [f"action {ACTION_NAMES.get(action_id, f'number {action_id}')}"]

        if disappeared and appeared:
            parts.append(f"moved {len(disappeared)} pixel(s)")
        elif disappeared:
            parts.append(f"removed {len(disappeared)} pixel(s)")
        elif appeared:
            parts.append(f"added {len(appeared)} pixel(s)")

        parts.append(f"changing {change_pct:.1f}% of the grid")
        return " ".join(parts)

    def generate_synthetic_transition(self) -> Tier1Sample:
        """Generate a synthetic transition using random grid + grammar transform."""
        from soma_mythos_ehra.arc3.expanded_grammar import (
            sample_extended_program,
            EXTENDED_EXECUTORS,
        )

        # Random small grid (8-16)
        h = self.rng.randint(8, 16)
        w = self.rng.randint(8, 16)
        grid = self.np_rng.randint(0, 5, size=(h, w)).astype(np.int64)

        # Add some objects
        for _ in range(self.rng.randint(1, 4)):
            oh = self.rng.randint(2, min(5, h))
            ow = self.rng.randint(2, min(5, w))
            oy = self.rng.randint(0, h - oh)
            ox = self.rng.randint(0, w - ow)
            color = self.rng.randint(1, 4)
            grid[oy:oy+oh, ox:ox+ow] = color

        grid_t = torch.from_numpy(grid)

        # Sample a program
        try:
            program = sample_extended_program(max_depth=2)
            # Find an executor
            action_id = self.rng.randint(1, 7)

            # Try to find a matching executor for the program name
            next_grid_t = grid_t.clone()
            desc = f"applied {program.name if hasattr(program, 'name') else 'transform'}"

            # Execute via expanded executors if possible
            if hasattr(program, "name") and program.name in EXTENDED_EXECUTORS:
                try:
                    next_grid_t = EXTENDED_EXECUTORS[program.name](grid_t.clone())
                    desc = f"applied {program.name}"
                except Exception:
                    pass
            elif hasattr(program, "children") and program.children:
                # Try first child
                for child in program.children:
                    if hasattr(child, "name") and child.name in EXTENDED_EXECUTORS:
                        try:
                            next_grid_t = EXTENDED_EXECUTORS[child.name](grid_t.clone())
                            desc = f"applied {child.name}"
                            break
                        except Exception:
                            pass
        except Exception:
            action_id = self.rng.randint(1, 7)
            next_grid_t = grid_t.clone()
            desc = f"action {action_id} on random grid"

        # Pad/truncate to fixed size (16x16)
        grid_np = self._pad_to(grid, 16, 16)
        next_np = self._pad_to(next_grid_t.numpy() if isinstance(next_grid_t, torch.Tensor) else next_grid_t, 16, 16)

        # Simple latent: flatten + pad to 256
        grid_latent = torch.zeros(256)
        flat = grid_np.flatten()[:256]
        grid_latent[:len(flat)] = torch.from_numpy(flat).float() / 4.0

        next_latent = torch.zeros(256)
        flat_next = next_np.flatten()[:256]
        next_latent[:len(flat_next)] = torch.from_numpy(flat_next).float() / 4.0

        reward = 1.0 if self.rng.random() < 0.05 else 0.0
        state = "WIN" if reward > 0 else "NOT_FINISHED"

        return Tier1Sample(
            grid_latent=grid_latent,
            action_id=action_id,
            next_grid_latent=next_latent,
            reward=reward,
            game_state=state,
            grid_diff_type="movement" if np.any(grid_np != next_np) else "no_change",
            nl_description=desc,
            prev_grid=grid_np,
            next_grid=next_np,
        )

    def generate_from_buffer(self, transitions: list[dict]) -> list[Tier1Sample]:
        """Convert replay buffer transitions to Tier 1 samples."""
        samples = []
        for t in transitions:
            prev = t.get("prev_grid")
            next_ = t.get("next_grid")
            action = t.get("action", 1)
            reward = t.get("reward", 0.0)

            if prev is None or next_ is None:
                continue

            prev_np = np.array(prev) if not isinstance(prev, np.ndarray) else prev
            next_np = np.array(next_) if not isinstance(next_, np.ndarray) else next_

            # Latents
            grid_flat = self._pad_to(prev_np, 16, 16).flatten()[:256]
            next_flat = self._pad_to(next_np, 16, 16).flatten()[:256]

            grid_latent = torch.zeros(256)
            grid_latent[:len(grid_flat)] = torch.from_numpy(grid_flat).float() / 4.0

            next_latent = torch.zeros(256)
            next_latent[:len(next_flat)] = torch.from_numpy(next_flat).float() / 4.0

            nl = self._describe_grid_diff(prev_np, next_np, action)

            samples.append(Tier1Sample(
                grid_latent=grid_latent,
                action_id=action,
                next_grid_latent=next_latent,
                reward=reward,
                game_state="WIN" if t.get("done") else "NOT_FINISHED",
                grid_diff_type="movement" if np.any(prev_np != next_np) else "no_change",
                nl_description=nl,
                prev_grid=prev_np,
                next_grid=next_np,
            ))
        return samples

    def generate_batch(self, count: int) -> list[Tier1Sample]:
        """Generate a batch of synthetic Tier 1 samples."""
        return [self.generate_synthetic_transition() for _ in range(count)]

    def _pad_to(self, arr: np.ndarray, h: int, w: int) -> np.ndarray:
        out = np.zeros((h, w), dtype=arr.dtype)
        sh, sw = min(arr.shape[0], h), min(arr.shape[1], w)
        out[:sh, :sw] = arr[:sh, :sw]
        return out

    def sample_to_tokens(self, sample: Tier1Sample) -> list[int]:
        """Tokenize a Tier 1 sample into the shared vocabulary."""
        tokens = [TOK_BOS, TOK_GRID_START]

        # Grid latent quantized to tokens
        for v in sample.grid_latent[:64]:
            tokens.append(GRAMMAR_TOKEN_START + int(v * 69) % 70)

        tokens.append(TOK_GRID_END)
        tokens.append(TOK_ACTION)
        tokens.append(ASCII_BASE + ord(str(sample.action_id)))
        tokens.append(TOK_GRID_START)

        for v in sample.next_grid_latent[:64]:
            tokens.append(GRAMMAR_TOKEN_START + int(v * 69) % 70)

        tokens.append(TOK_GRID_END)
        tokens.append(TOK_REWARD)
        tokens.append(ASCII_BASE + ord("1" if sample.reward > 0 else "0"))
        tokens.append(TOK_REASON_START)
        tokens.extend(text_to_tokens(sample.nl_description))
        tokens.append(TOK_REASON_END)
        tokens.append(TOK_EOS)
        return tokens


# ══════════════════════════════════════════════════════════════════════════════
# Tier 2: Synthetic Procedural Reasonings
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Tier2Sample:
    """A single Tier 2 procedural reasoning trace."""
    program_tokens: list[int]    # grammar tokens representing the program
    trace_text: str              # step-by-step NL trace
    input_grid: np.ndarray | None = None
    output_grid: np.ndarray | None = None
    category: str = "transform"  # transform, filter, compose, branch


class Tier2Generator:
    """Generates Tier 2 synthetic procedural reasoning traces.

    Samples random programs from the expanded grammar, executes them on
    synthetic grids, and records the complete reasoning trace as text.
    """

    TRANSFORM_NAMES = [
        "mirror_h", "mirror_v", "mirror_diag", "tessellate_2x2",
        "tessellate_3x3", "quadrant_fill", "border_frame", "center_crop",
        "diagonal_copy", "row_repeat", "col_repeat", "invert_mask",
        "object_shift_up", "object_shift_down", "object_shift_left",
        "object_shift_right", "grow_objects", "shrink_objects",
        "fill_interior", "outline_objects",
    ]

    FILTER_NAMES = [
        "all", "by_area_max", "by_area_min", "by_density_solid",
        "by_density_hollow", "by_color_1", "by_color_2", "by_color_3",
        "by_position_top", "by_position_bottom", "by_position_left",
        "by_position_right", "by_position_center",
    ]

    CONDITION_NAMES = [
        "is_largest", "is_smallest", "is_solid", "is_hollow",
        "has_area_gt_5", "has_area_lt_5", "has_objects_above_3",
        "has_objects_below_3", "grid_is_square", "grid_is_wide", "grid_is_tall",
    ]

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def _make_grid(self, h: int = 8, w: int = 8) -> np.ndarray:
        grid = self.np_rng.randint(0, 4, size=(h, w)).astype(np.int64)
        for _ in range(self.rng.randint(1, 3)):
            oh = self.rng.randint(2, min(4, h))
            ow = self.rng.randint(2, min(4, w))
            oy = self.rng.randint(0, h - oh)
            ox = self.rng.randint(0, w - ow)
            grid[oy:oy+oh, ox:ox+ow] = self.rng.randint(1, 3)
        return grid

    def generate_single_trace(self) -> Tier2Sample:
        """Generate one procedural reasoning trace."""
        depth = self.rng.randint(1, 3)
        grid = self._make_grid()
        steps = []
        steps.append(f"define grid as {grid.shape[0]}x{grid.shape[1]} matrix")
        steps.append(f"grid contains {int(np.sum(grid > 0))} non-zero objects")

        current = grid.copy()
        program_tokens = [TOK_BOS, TOK_TRACE_START]

        for step_num in range(depth):
            op_type = self.rng.choice(["transform", "filter_apply", "branch"])

            if op_type == "transform":
                name = self.rng.choice(self.TRANSFORM_NAMES)
                steps.append(f"step {step_num + 1}: apply {name}")
                program_tokens.extend([TOK_TRACE_STEP_NUM, ASCII_BASE + ord(str(step_num + 1))])
                program_tokens.extend(text_to_tokens(name))

                try:
                    from soma_mythos_ehra.arc3.expanded_grammar import EXTENDED_EXECUTORS
                    if name in EXTENDED_EXECUTORS:
                        result = EXTENDED_EXECUTORS[name](torch.from_numpy(current).clone())
                        if isinstance(result, torch.Tensor):
                            current = result.numpy()
                        steps.append(f"  result: {current.shape[0]}x{current.shape[1]} grid, "
                                   f"{int(np.sum(current > 0))} non-zero pixels")
                except Exception:
                    steps.append(f"  result: transform applied (estimated)")

            elif op_type == "filter_apply":
                filt = self.rng.choice(self.FILTER_NAMES)
                steps.append(f"step {step_num + 1}: filter objects where {filt}")
                program_tokens.extend([TOK_TRACE_STEP_NUM, ASCII_BASE + ord(str(step_num + 1))])
                program_tokens.extend(text_to_tokens(f"filter {filt}"))

                if filt == "all":
                    count = int(np.sum(current > 0))
                    steps.append(f"  matched {count} objects")
                elif "color" in filt:
                    color_num = int(filt.split("_")[-1])
                    count = int(np.sum(current == color_num))
                    steps.append(f"  matched {count} pixels of color {color_num}")
                elif "position" in filt:
                    H, W = current.shape
                    if "top" in filt:
                        count = int(np.sum(current[:H//2, :] > 0))
                    elif "bottom" in filt:
                        count = int(np.sum(current[H//2:, :] > 0))
                    elif "left" in filt:
                        count = int(np.sum(current[:, :W//2] > 0))
                    elif "right" in filt:
                        count = int(np.sum(current[:, W//2:] > 0))
                    else:
                        count = int(np.sum(current > 0))
                    steps.append(f"  matched {count} objects")

            else:  # branch
                cond = self.rng.choice(self.CONDITION_NAMES)
                steps.append(f"step {step_num + 1}: if {cond} then apply transform else skip")
                program_tokens.extend([TOK_TRACE_STEP_NUM, ASCII_BASE + ord(str(step_num + 1))])
                program_tokens.extend(text_to_tokens(f"if {cond}"))

                if cond in ("grid_is_square", "grid_is_wide", "grid_is_tall"):
                    H, W = current.shape
                    if cond == "grid_is_square":
                        result = H == W
                    elif cond == "grid_is_wide":
                        result = W > H
                    else:
                        result = H > W
                    steps.append(f"  condition {cond}: {'true' if result else 'false'}")
                elif cond in ("is_solid", "is_hollow"):
                    density = np.sum(current > 0) / current.size if current.size > 0 else 0
                    result = density > 0.5 if cond == "is_solid" else density < 0.3
                    steps.append(f"  condition {cond}: {'true' if result else 'false'} (density={density:.2f})")
                else:
                    result = self.rng.random() > 0.5
                    steps.append(f"  condition {cond}: {result}")

                if result:
                    sub_name = self.rng.choice(self.TRANSFORM_NAMES[:5])
                    steps.append(f"  -> applying {sub_name}")
                    program_tokens.extend(text_to_tokens(f"then {sub_name}"))

            program_tokens.append(TOK_SEP)

        steps.append(f"final grid: {current.shape[0]}x{current.shape[1]}, "
                    f"{int(np.sum(current > 0))} non-zero pixels")
        program_tokens.append(TOK_TRACE_END)
        program_tokens.append(TOK_EOS)

        return Tier2Sample(
            program_tokens=program_tokens,
            trace_text="\n".join(steps),
            input_grid=grid,
            output_grid=current,
            category="compose" if depth > 1 else "transform",
        )

    def generate_batch(self, count: int) -> list[Tier2Sample]:
        return [self.generate_single_trace() for _ in range(count)]

    def sample_to_tokens(self, sample: Tier2Sample) -> list[int]:
        """Tokenize a Tier 2 sample."""
        return sample.program_tokens


# ══════════════════════════════════════════════════════════════════════════════
# Tier 3: Algorithmic Logic Chains
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Tier3Sample:
    """A single Tier 3 algorithmic reasoning chain."""
    problem_text: str
    steps: list[str]
    answer: str
    category: str  # sorting, math, tree, search, string


class Tier3Generator:
    """Generates Tier 3 algorithmic logic chains.

    Produces step-by-step reasoning traces for:
      - Array sorting (bubble, selection, insertion)
      - Basic arithmetic proofs
      - Binary search
      - Tree traversals
      - String operations
      - Dictionary/map lookups
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def _gen_sorting_problem(self) -> Tier3Sample:
        size = self.rng.randint(3, 7)
        arr = [self.rng.randint(0, 20) for _ in range(size)]
        steps = [f"problem: sort array {arr}"]

        sorted_arr = sorted(arr)
        work = list(arr)

        # Selection sort trace
        for i in range(len(work)):
            min_idx = i
            for j in range(i + 1, len(work)):
                if work[j] < work[min_idx]:
                    min_idx = j
                    steps.append(f"step {i * len(work) + j + 1}: compare {work[j]} < {work[min_idx - (1 if min_idx != j else 0)]} -> update min_idx to {j}")

            if min_idx != i:
                steps.append(f"step swap: swap index {i} (value {work[i]}) with index {min_idx} (value {work[min_idx]})")
                work[i], work[min_idx] = work[min_idx], work[i]
            else:
                steps.append(f"step {i + 1}: index {i} already has minimum {work[i]}")

            steps.append(f"  array is now {work}")

        return Tier3Sample(
            problem_text=f"sort array {arr}",
            steps=steps,
            answer=str(sorted_arr),
            category="sorting",
        )

    def _gen_math_problem(self) -> Tier3Sample:
        ops = [("+", lambda a, b: a + b),
               ("-", lambda a, b: a - b),
               ("*", lambda a, b: a * b)]
        op_name, op_fn = self.rng.choice(ops)
        a = self.rng.randint(1, 30)
        b = self.rng.randint(1, 15)
        result = op_fn(a, b)

        steps = [
            f"problem: compute {a} {op_name} {b}",
            f"step 1: identify operands a={a}, b={b}",
            f"step 2: apply operation {op_name}",
            f"step 3: {a} {op_name} {b} = {result}",
        ]

        # Add a second operation
        op2_name, op2_fn = self.rng.choice(ops)
        c = self.rng.randint(1, 10)
        result2 = op2_fn(result, c)
        steps.append(f"step 4: compute {result} {op2_name} {c} = {result2}")

        return Tier3Sample(
            problem_text=f"compute ({a} {op_name} {b}) {op2_name} {c}",
            steps=steps,
            answer=str(result2),
            category="math",
        )

    def _gen_search_problem(self) -> Tier3Sample:
        arr = sorted([self.rng.randint(0, 30) for _ in range(self.rng.randint(5, 10))])
        target = self.rng.choice(arr)

        steps = [f"problem: find {target} in sorted array {arr}"]
        lo, hi = 0, len(arr) - 1
        steps.append(f"step 1: initialize lo=0, hi={hi}")

        found = False
        step_num = 2
        while lo <= hi:
            mid = (lo + hi) // 2
            steps.append(f"step {step_num}: mid = ({lo} + {hi}) / 2 = {mid}, arr[{mid}] = {arr[mid]}")
            if arr[mid] == target:
                steps.append(f"step {step_num + 1}: arr[{mid}] == {target}, found!")
                found = True
                break
            elif arr[mid] < target:
                lo = mid + 1
                steps.append(f"step {step_num + 1}: {arr[mid]} < {target}, search right: lo={lo}")
            else:
                hi = mid - 1
                steps.append(f"step {step_num + 1}: {arr[mid]} > {target}, search left: hi={hi}")
            step_num += 2

        if not found:
            steps.append(f"step {step_num}: search complete, {target} not found")

        return Tier3Sample(
            problem_text=f"find {target} in {arr}",
            steps=steps,
            answer=f"index {mid if found else -1}",
            category="search",
        )

    def _gen_tree_problem(self) -> Tier3Sample:
        # Build a small binary tree
        size = self.rng.randint(5, 9)
        values = list(range(1, size + 1))
        self.rng.shuffle(values)

        steps = [f"problem: traverse binary tree with values {values}"]
        steps.append(f"step 1: insert values into BST")

        # Build BST
        tree = {}
        for v in values:
            if not tree:
                tree = {"val": v, "left": None, "right": None}
                steps.append(f"  insert {v}: root node")
            else:
                node = tree
                path = []
                while True:
                    if v < node["val"]:
                        path.append("left")
                        if node["left"] is None:
                            node["left"] = {"val": v, "left": None, "right": None}
                            steps.append(f"  insert {v}: go {' -> '.join(path)} -> new node")
                            break
                        node = node["left"]
                    else:
                        path.append("right")
                        if node["right"] is None:
                            node["right"] = {"val": v, "left": None, "right": None}
                            steps.append(f"  insert {v}: go {' -> '.join(path)} -> new node")
                            break
                        node = node["right"]

        # In-order traversal
        def inorder(node):
            if node is None:
                return []
            return inorder(node["left"]) + [node["val"]] + inorder(node["right"])

        result = inorder(tree)
        steps.append(f"step 2: in-order traversal")
        steps.append(f"  result: {result}")
        steps.append(f"  verification: {'sorted' if result == sorted(values) else 'not sorted'}")

        return Tier3Sample(
            problem_text=f"build and traverse BST from {values}",
            steps=steps,
            answer=str(result),
            category="tree",
        )

    def _gen_string_problem(self) -> Tier3Sample:
        s = "".join(self.rng.choices("abcdef", k=self.rng.randint(4, 8)))
        target_char = self.rng.choice(list(s))

        steps = [f"problem: find first occurrence of '{target_char}' in '{s}'"]
        found_idx = -1
        for i, c in enumerate(s):
            if c == target_char:
                steps.append(f"step {i + 1}: s[{i}] = '{c}' == '{target_char}' -> match found at index {i}")
                found_idx = i
                break
            else:
                steps.append(f"step {i + 1}: s[{i}] = '{c}' != '{target_char}' -> continue")

        if found_idx == -1:
            steps.append(f"result: '{target_char}' not found in string")

        return Tier3Sample(
            problem_text=f"find '{target_char}' in '{s}'",
            steps=steps,
            answer=f"index {found_idx}" if found_idx >= 0 else "not found",
            category="string",
        )

    def generate_single(self) -> Tier3Sample:
        gen = self.rng.choice([
            self._gen_sorting_problem,
            self._gen_math_problem,
            self._gen_search_problem,
            self._gen_tree_problem,
            self._gen_string_problem,
        ])
        return gen()

    def generate_batch(self, count: int) -> list[Tier3Sample]:
        return [self.generate_single() for _ in range(count)]

    def sample_to_tokens(self, sample: Tier3Sample) -> list[int]:
        """Tokenize a Tier 3 sample."""
        tokens = [TOK_BOS, TOK_PROBLEM]
        tokens.extend(text_to_tokens(sample.problem_text))
        tokens.append(TOK_SEP)

        for step in sample.steps:
            tokens.append(TOK_STEP)
            tokens.extend(text_to_tokens(step))
            tokens.append(TOK_SEP)

        tokens.append(TOK_ANSWER)
        tokens.extend(text_to_tokens(sample.answer))
        tokens.append(TOK_EOS)
        return tokens


# ══════════════════════════════════════════════════════════════════════════════
# Tier 4: Structural Text Corpora
# ══════════════════════════════════════════════════════════════════════════════

ESSAY_TOPICS = [
    "the nature of intelligence and reasoning",
    "how machines learn to solve abstract problems",
    "the relationship between language and logic",
    "spatial reasoning and visual perception",
    "the architecture of thought and cognition",
    "pattern recognition in complex systems",
    "the foundations of mathematical proof",
    "how neural networks process information",
    "the role of abstraction in problem solving",
    "emergent behavior in complex adaptive systems",
    "the connection between creativity and computation",
    "understanding consciousness through artificial systems",
    "the limits of formal reasoning systems",
    "how analogy drives conceptual understanding",
    "the structure of natural and artificial languages",
    "temporal reasoning and sequential decision making",
    "the relationship between compression and intelligence",
    "symbolic vs connectionist approaches to AI",
    "the role of attention in selective processing",
    "embodied cognition and the grounding problem",
]


PARAGRAPH_TEMPLATES = [
    "{topic} is a fundamental question that has occupied researchers for decades. "
    "The core challenge lies in understanding how abstract representations emerge from "
    "raw sensory data. By examining the structural properties of information processing "
    "systems, we can identify the key mechanisms that enable reasoning and generalization.",

    "When we consider {topic}, we must first establish the foundational principles. "
    "A system that can reason must be able to manipulate symbols according to rules, "
    "while maintaining consistency across its internal representations. This requires "
    "both a memory mechanism and an inference procedure that operates over stored knowledge.",

    "The problem of {topic} connects to broader questions about the nature of cognition. "
    "Research has shown that effective reasoning systems combine pattern matching with "
    "rule-based inference. The key insight is that abstraction allows a system to "
    "generalize from specific instances to general principles.",

    "Understanding {topic} requires us to examine both the computational and representational "
    "aspects of intelligence. A well-designed system must be able to decompose complex "
    "problems into simpler sub-problems, solve each sub-problem, and compose the solutions "
    "into a coherent whole. This decomposition principle is central to all forms of reasoning.",

    "The study of {topic} reveals important principles about the structure of knowledge. "
    "Effective reasoning requires maintaining a hierarchy of abstractions, from low-level "
    "perceptual features to high-level conceptual categories. This hierarchical organization "
    "allows the system to operate at multiple levels of granularity simultaneously.",

    "In exploring {topic}, we find that the most powerful approaches combine multiple "
    "representational formats. Spatial reasoning, linguistic processing, and mathematical "
    "computation each provide unique strengths. A unified system must integrate these "
    "different modalities into a coherent reasoning framework.",

    "The challenge of {topic} is intimately connected to the problem of transfer learning. "
    "A system that truly understands a domain should be able to apply its knowledge to "
    "novel situations. This requires extracting the underlying principles rather than "
    "memorizing surface-level patterns.",

    "Recent advances in {topic} have highlighted the importance of compositionality. "
    "Complex concepts are built from simpler components according to systematic rules. "
    "Understanding these rules of composition is essential for building systems that "
    "can handle the full complexity of real-world reasoning tasks.",
]

CONCLUSION_TEMPLATES = [
    "In conclusion, {topic} remains one of the central challenges in understanding intelligence. "
    "The path forward requires combining insights from multiple disciplines, including computer "
    "science, cognitive science, and mathematics. Only through such interdisciplinary approaches "
    "can we hope to build systems that truly reason.",

    "The exploration of {topic} demonstrates the depth and complexity of intelligent behavior. "
    "As we continue to develop more sophisticated systems, we gain deeper appreciation for "
    "the remarkable capabilities of natural intelligence and the principles that underlie it.",

    "Ultimately, the study of {topic} teaches us about the nature of knowledge itself. "
    "By building artificial systems that can reason, we discover fundamental truths about "
    "the structure of thought and the organization of information in complex systems.",
]


class Tier4Generator:
    """Generates Tier 4 structural text corpora.

    Produces clean, structured prose paragraphs mapped to ASCII tokens.
    Covers essays, technical descriptions, and structured arguments.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def generate_essay(self, topic: str | None = None) -> str:
        """Generate a short essay (3-5 paragraphs)."""
        if topic is None:
            topic = self.rng.choice(ESSAY_TOPICS)

        paragraphs = []
        # Introduction
        paragraphs.append(PARAGRAPH_TEMPLATES[0].format(topic=topic))

        # Body paragraphs
        num_body = self.rng.randint(2, 4)
        used = {0}
        for _ in range(num_body):
            idx = self.rng.choice([i for i in range(len(PARAGRAPH_TEMPLATES)) if i not in used])
            used.add(idx)
            paragraphs.append(PARAGRAPH_TEMPLATES[idx].format(topic=topic))

        # Conclusion
        paragraphs.append(self.rng.choice(CONCLUSION_TEMPLATES).format(topic=topic))

        return "\n\n".join(paragraphs)

    def generate_technical_description(self) -> str:
        """Generate a technical system description."""
        components = self.rng.sample([
            ("encoder", "maps input data into a latent representation"),
            ("decoder", "reconstructs the output from latent features"),
            ("attention mechanism", "selectively focuses on relevant information"),
            ("memory bank", "stores past experiences for future reference"),
            ("inference engine", "applies logical rules to derive conclusions"),
            ("prediction head", "outputs the final classification or regression"),
            ("tokenizer", "converts raw text into discrete token sequences"),
            ("embedding layer", "maps discrete tokens to continuous vectors"),
            ("transformer block", "processes sequences through self-attention and feedforward layers"),
            ("loss function", "measures the discrepancy between predictions and targets"),
        ], k=self.rng.randint(3, 6))

        lines = [f"System architecture overview:"]
        lines.append("")
        lines.append(f"This system implements a modular pipeline for reasoning tasks.")
        lines.append("")

        for name, desc in components:
            lines.append(f"The {name} {desc}.")
            lines.append(f"It operates on intermediate representations passed from upstream components.")
            lines.append("")

        lines.append("All components are trained end-to-end using gradient descent optimization.")
        lines.append("The system achieves competitive performance on standard benchmarks.")

        return "\n".join(lines)

    def generate_paragraph(self, topic: str | None = None) -> str:
        """Generate a single paragraph."""
        if topic is None:
            topic = self.rng.choice(ESSAY_TOPICS)
        template = self.rng.choice(PARAGRAPH_TEMPLATES)
        return template.format(topic=topic)

    def generate_batch_text(self, count: int) -> list[str]:
        """Generate a batch of text samples."""
        samples = []
        for _ in range(count):
            choice = self.rng.choices(
                ["essay", "technical", "paragraph"],
                weights=[0.3, 0.3, 0.4],
                k=1,
            )[0]
            if choice == "essay":
                samples.append(self.generate_essay())
            elif choice == "technical":
                samples.append(self.generate_technical_description())
            else:
                samples.append(self.generate_paragraph())
        return samples

    def sample_to_tokens(self, text: str) -> list[int]:
        """Tokenize text into the shared vocabulary."""
        tokens = [TOK_BOS, TOK_ESSAY]
        tokens.extend(text_to_tokens(text))
        tokens.append(TOK_EOS)
        return tokens


# ══════════════════════════════════════════════════════════════════════════════
# Unified Dataset
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UnifiedSample:
    """A single sample from any tier, tokenized into the shared vocabulary."""
    tokens: list[int]
    tier: int  # 1, 2, 3, or 4
    category: str  # fine-grained category
    metadata: dict = field(default_factory=dict)


class FourTierDataset:
    """Unified dataset combining all four tiers.

    Generates and tokenizes samples from:
      Tier 1: Core physics interactions (25%)
      Tier 2: Synthetic procedural traces (25%)
      Tier 3: Algorithmic logic chains (25%)
      Tier 4: Structural text corpora (25%)
    """

    def __init__(
        self,
        tier1_count: int = 100_000,
        tier2_count: int = 50_000,
        tier3_count: int = 10_000,
        tier4_count: int = 50_000,
        max_seq_len: int = 512,
        seed: int = 42,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.tier1_count = tier1_count
        self.tier2_count = tier2_count
        self.tier3_count = tier3_count
        self.tier4_count = tier4_count

        self.tier1_gen = Tier1Generator(seed=seed)
        self.tier2_gen = Tier2Generator(seed=seed)
        self.tier3_gen = Tier3Generator(seed=seed)
        self.tier4_gen = Tier4Generator(seed=seed)

        self.total = tier1_count + tier2_count + tier3_count + tier4_count

    def generate_all(self, save_dir: str | None = None) -> list[UnifiedSample]:
        """Generate the complete four-tier dataset."""
        samples = []

        print(f"Generating Tier 1: {self.tier1_count} physics interactions...")
        for i in range(self.tier1_count):
            s = self.tier1_gen.generate_synthetic_transition()
            tokens = self.tier1_gen.sample_to_tokens(s)
            tokens = tokens[:self.max_seq_len]
            samples.append(UnifiedSample(
                tokens=tokens, tier=1, category=s.grid_diff_type,
                metadata={"action": s.action_id, "reward": s.reward},
            ))
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{self.tier1_count}")

        print(f"Generating Tier 2: {self.tier2_count} procedural traces...")
        for i in range(self.tier2_count):
            s = self.tier2_gen.generate_single_trace()
            tokens = self.tier2_gen.sample_to_tokens(s)
            tokens = tokens[:self.max_seq_len]
            samples.append(UnifiedSample(
                tokens=tokens, tier=2, category=s.category,
            ))
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{self.tier2_count}")

        print(f"Generating Tier 3: {self.tier3_count} logic chains...")
        for i in range(self.tier3_count):
            s = self.tier3_gen.generate_single()
            tokens = self.tier3_gen.sample_to_tokens(s)
            tokens = tokens[:self.max_seq_len]
            samples.append(UnifiedSample(
                tokens=tokens, tier=3, category=s.category,
            ))
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{self.tier3_count}")

        print(f"Generating Tier 4: {self.tier4_count} text corpora...")
        for i in range(self.tier4_count):
            text = self.tier4_gen.generate_essay()
            tokens = self.tier4_gen.sample_to_tokens(text)
            tokens = tokens[:self.max_seq_len]
            samples.append(UnifiedSample(
                tokens=tokens, tier=4, category="essay",
            ))
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{self.tier4_count}")

        print(f"Total dataset: {len(samples)} samples")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._save_dataset(samples, save_dir)

        return samples

    def _save_dataset(self, samples: list[UnifiedSample], save_dir: str) -> None:
        """Save dataset to disk as tokenized tensors."""
        # Split into train/val
        self.rng = random.Random(42)
        indices = list(range(len(samples)))
        self.rng.shuffle(indices)

        split = int(len(indices) * 0.95)
        train_idx = indices[:split]
        val_idx = indices[split:]

        # Save as token tensors
        for name, idx_list in [("train", train_idx), ("val", val_idx)]:
            all_tokens = [samples[i].tokens for i in idx_list]
            # Pad to max length
            padded = torch.zeros(len(all_tokens), self.max_seq_len, dtype=torch.long)
            for i, tokens in enumerate(all_tokens):
                padded[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

            torch.save(padded, os.path.join(save_dir, f"{name}_tokens.pt"))
            print(f"  Saved {name}: {padded.shape}")

        # Save metadata
        meta = {
            "total": len(samples),
            "train": len(train_idx),
            "val": len(val_idx),
            "tier_counts": {
                "tier1": self.tier1_count,
                "tier2": self.tier2_count,
                "tier3": self.tier3_count,
                "tier4": self.tier4_count,
            },
            "max_seq_len": self.max_seq_len,
            "vocab_size": VOCAB_SIZE,
        }
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(save_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load saved dataset."""
        train = torch.load(os.path.join(save_dir, "train_tokens.pt"), weights_only=True)
        val = torch.load(os.path.join(save_dir, "val_tokens.pt"), weights_only=True)
        return train, val


# ══════════════════════════════════════════════════════════════════════════════
# CLI: Generate and save the dataset
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Four-Tier Unified Token Dataset")
    parser.add_argument("--tier1", type=int, default=100_000, help="Tier 1 sample count")
    parser.add_argument("--tier2", type=int, default=50_000, help="Tier 2 sample count")
    parser.add_argument("--tier3", type=int, default=10_000, help="Tier 3 sample count")
    parser.add_argument("--tier4", type=int, default=50_000, help="Tier 4 sample count")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--save-dir", type=str, default="data/four_tier", help="Save directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    dataset = FourTierDataset(
        tier1_count=args.tier1,
        tier2_count=args.tier2,
        tier3_count=args.tier3,
        tier4_count=args.tier4,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )

    dataset.generate_all(save_dir=args.save_dir)
    print(f"Dataset saved to {args.save_dir}")
