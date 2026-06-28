"""Game Trajectory Tokenizer — converts environment transitions to integer tokens.

Encodes (grid_diff, action, reward, game_state) into structural integer sequences
for training the local action transformer. Vocabulary:
  0-9:     Action tokens (ACTION1-ACTION7, RESET, UNDO)
  10-19:   Reward magnitude buckets (quantized 0.0-1.0)
  20-29:   Game state tokens (NOT_FINISHED, WIN, GAME_OVER)
  30-39:   Grid diff structural tokens (movement detected, color change, object moved)
  40-49:   Coordinate region tokens (quantized x,y click positions)
  50-59:   Temporal tokens (step count buckets)
  60-69:   Topology tokens (object count, symmetry, density)
  70-79:   Sequence context tokens (SOS, EOS, SEP, PAD, MASK)
"""
from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass


# ── Token ID Constants ──

ACTION_BASE = 0
ACTION_IDS = {
    "RESET": 0, "ACTION1": 1, "ACTION2": 2, "ACTION3": 3,
    "ACTION4": 4, "ACTION5": 5, "ACTION6": 6, "ACTION7": 7,
    "UNDO": 8, "NOOP": 9,
}

REWARD_BASE = 10
NUM_REWARD_BUCKETS = 10

STATE_BASE = 20
STATE_IDS = {
    "NOT_FINISHED": 20, "WIN": 21, "GAME_OVER": 22,
}

DIFF_BASE = 30
DIFF_IDS = {
    "NO_CHANGE": 30, "MOVEMENT": 31, "COLOR_SHIFT": 32,
    "OBJECT_MOVED": 33, "FILL_CHANGED": 34, "ROTATION": 35,
    "SCALING": 36, "INVERSION": 37, "BOUNDARY": 38, "RANDOM": 39,
}

COORD_BASE = 40
NUM_COORD_BUCKETS = 10

TEMPORAL_BASE = 50
NUM_TEMPORAL_BUCKETS = 10

TOPOLOGY_BASE = 60
TOPOLOGY_IDS = {
    "OBJECTS_0": 60, "OBJECTS_1": 61, "OBJECTS_2_3": 62,
    "OBJECTS_4_7": 63, "OBJECTS_8_PLUS": 64,
    "ASYMMETRIC": 65, "SYMMETRIC_H": 66, "SYMMETRIC_V": 67,
    "DENSE": 68, "SPARSE": 69,
}

CONTEXT_BASE = 70
CONTEXT_IDS = {
    "SOS": 70, "EOS": 71, "SEP": 72, "PAD": 73, "MASK": 74,
}

VOCAB_SIZE = 128  # Reserved space for future expansion


def quantize_reward(reward: float) -> int:
    """Quantize reward [0.0, 1.0] into bucket index [0, 9]."""
    return min(int(reward * NUM_REWARD_BUCKETS), NUM_REWARD_BUCKETS - 1)


def quantize_coord(x: int, y: int, grid_shape: tuple[int, int] = (64, 64)) -> tuple[int, int]:
    """Quantize x,y coordinates into bucket indices [0, 9]."""
    H, W = grid_shape
    qx = min(int(x / W * NUM_COORD_BUCKETS), NUM_COORD_BUCKETS - 1)
    qy = min(int(y / H * NUM_COORD_BUCKETS), NUM_COORD_BUCKETS - 1)
    return qx, qy


def quantize_step(step: int, max_steps: int = 500) -> int:
    """Quantize step count into bucket index [0, 9]."""
    return min(int(step / max_steps * NUM_TEMPORAL_BUCKETS), NUM_TEMPORAL_BUCKETS - 1)


def classify_diff(prev_grid: np.ndarray, next_grid: np.ndarray) -> int:
    """Classify the structural nature of a grid transition."""
    if prev_grid.shape != next_grid.shape:
        return DIFF_IDS["SCALING"]

    diff = prev_grid != next_grid
    if not np.any(diff):
        return DIFF_IDS["NO_CHANGE"]

    change_ratio = np.mean(diff)

    # Check for rotation
    for k in [1, 2, 3]:
        if np.array_equal(np.rot90(prev_grid, k), next_grid):
            return DIFF_IDS["ROTATION"]

    # Check for translation/shift
    for axis in [0, 1]:
        for shift in [-3, -2, -1, 1, 2, 3]:
            shifted = np.roll(prev_grid, shift, axis=axis)
            if np.array_equal(shifted, next_grid):
                return DIFF_IDS["MOVEMENT"]

    # Check for color shift (same pattern, different values)
    if np.sum(prev_grid > 0) == np.sum(next_grid > 0) and change_ratio < 0.3:
        return DIFF_IDS["COLOR_SHIFT"]

    # Check for fill changes
    if np.sum(prev_grid == 0) != np.sum(next_grid == 0):
        return DIFF_IDS["FILL_CHANGED"]

    if change_ratio > 0.5:
        return DIFF_IDS["INVERSION"]

    return DIFF_IDS["OBJECT_MOVED"]


def count_topology(grid: np.ndarray) -> dict[str, int]:
    """Extract topology features from a grid."""
    nonzero = np.count_nonzero(grid)
    total = grid.size
    density = nonzero / total if total > 0 else 0

    # Object count (connected components)
    from scipy import ndimage
    labeled, num_objects = ndimage.label(grid > 0)

    result = {}
    if num_objects == 0:
        result["objects"] = TOPOLOGY_IDS["OBJECTS_0"]
    elif num_objects == 1:
        result["objects"] = TOPOLOGY_IDS["OBJECTS_1"]
    elif num_objects <= 3:
        result["objects"] = TOPOLOGY_IDS["OBJECTS_2_3"]
    elif num_objects <= 7:
        result["objects"] = TOPOLOGY_IDS["OBJECTS_4_7"]
    else:
        result["objects"] = TOPOLOGY_IDS["OBJECTS_8_PLUS"]

    # Symmetry
    if np.array_equal(grid, grid[:, ::-1]):
        result["symmetry"] = TOPOLOGY_IDS["SYMMETRIC_H"]
    elif np.array_equal(grid, grid[::-1, :]):
        result["symmetry"] = TOPOLOGY_IDS["SYMMETRIC_V"]
    else:
        result["symmetry"] = TOPOLOGY_IDS["ASYMMETRIC"]

    # Density
    result["density"] = TOPOLOGY_IDS["DENSE"] if density > 0.3 else TOPOLOGY_IDS["SPARSE"]

    return result


class GameTrajectoryTokenizer:
    """Tokenizes environment transitions into integer sequences for training.

    Produces fixed-length token sequences from (state, action, reward) triples
    suitable for causal transformer training.
    """

    def __init__(self, max_seq_len: int = 64) -> None:
        self.max_seq_len = max_seq_len
        self.vocab_size = VOCAB_SIZE

    def encode_step(
        self,
        action: int,
        reward: float,
        game_state: str = "NOT_FINISHED",
        prev_grid: np.ndarray | None = None,
        next_grid: np.ndarray | None = None,
        x: int | None = None,
        y: int | None = None,
        step: int = 0,
    ) -> list[int]:
        """Encode a single transition step into tokens."""
        tokens = []

        # Action token
        action_name = f"ACTION{action}" if 1 <= action <= 7 else (
            "RESET" if action == 0 else "UNDO" if action == 7 else "NOOP"
        )
        tokens.append(ACTION_IDS.get(action_name, ACTION_IDS["NOOP"]))

        # Reward bucket
        tokens.append(REWARD_BASE + quantize_reward(reward))

        # Game state
        tokens.append(STATE_IDS.get(game_state, STATE_BASE))

        # Grid diff classification
        if prev_grid is not None and next_grid is not None:
            diff_token = classify_diff(
                np.array(prev_grid) if not isinstance(prev_grid, np.ndarray) else prev_grid,
                np.array(next_grid) if not isinstance(next_grid, np.ndarray) else next_grid,
            )
            tokens.append(diff_token)
        else:
            tokens.append(DIFF_IDS["NO_CHANGE"])

        # Coordinate tokens (for ACTION6 click)
        if x is not None and y is not None:
            qx, qy = quantize_coord(x, y)
            tokens.append(COORD_BASE + qx)
            tokens.append(COORD_BASE + qy)
        else:
            tokens.append(CONTEXT_IDS["PAD"])
            tokens.append(CONTEXT_IDS["PAD"])

        # Temporal token
        tokens.append(TEMPORAL_BASE + quantize_step(step))

        return tokens

    def encode_trajectory(
        self,
        actions: list[int],
        rewards: list[float],
        game_states: list[str] | None = None,
        prev_grids: list[np.ndarray] | None = None,
        next_grids: list[np.ndarray] | None = None,
        coords: list[tuple[int, int] | None] | None = None,
    ) -> torch.Tensor:
        """Encode a full trajectory into a token tensor.

        Returns:
            (max_seq_len,) tensor of token IDs, padded or truncated.
        """
        if game_states is None:
            game_states = ["NOT_FINISHED"] * len(actions)
        if prev_grids is None:
            prev_grids = [None] * len(actions)
        if next_grids is None:
            next_grids = [None] * len(actions)
        if coords is None:
            coords = [None] * len(actions)

        all_tokens = [CONTEXT_IDS["SOS"]]

        for i in range(len(actions)):
            step_tokens = self.encode_step(
                action=actions[i],
                reward=rewards[i],
                game_state=game_states[i],
                prev_grid=prev_grids[i],
                next_grid=next_grids[i],
                x=coords[i][0] if coords[i] else None,
                y=coords[i][1] if coords[i] else None,
                step=i,
            )
            all_tokens.extend(step_tokens)
            all_tokens.append(CONTEXT_IDS["SEP"])

        all_tokens.append(CONTEXT_IDS["EOS"])

        # Pad or truncate
        if len(all_tokens) < self.max_seq_len:
            all_tokens.extend([CONTEXT_IDS["PAD"]] * (self.max_seq_len - len(all_tokens)))
        else:
            all_tokens = all_tokens[:self.max_seq_len]

        return torch.tensor(all_tokens, dtype=torch.long)

    def encode_buffer_batch(
        self,
        transitions: list[dict],
    ) -> torch.Tensor:
        """Encode a batch of replay buffer transitions.

        Each transition dict should have: action, reward, prev_grid, next_grid, done, level
        Returns:
            (batch_size, max_seq_len) token tensor
        """
        batch_tokens = []
        for t in transitions:
            actions = [t["action"]]
            rewards = [t["reward"]]
            states = ["WIN" if t.get("done", False) else "NOT_FINISHED"]
            prev_grids = [t.get("prev_grid")]
            next_grids = [t.get("next_grid")]

            tokens = self.encode_trajectory(
                actions=actions,
                rewards=rewards,
                game_states=states,
                prev_grids=prev_grids,
                next_grids=next_grids,
            )
            batch_tokens.append(tokens)

        return torch.stack(batch_tokens)

    def decode_tokens(self, token_ids: torch.Tensor) -> list[str]:
        """Decode token IDs back to human-readable labels."""
        id_to_name = {}
        id_to_name.update({v: k for k, v in ACTION_IDS.items()})
        id_to_name.update({v: f"REWARD_{i}" for i, v in enumerate(range(REWARD_BASE, REWARD_BASE + NUM_REWARD_BUCKETS))})
        id_to_name.update({v: k for k, v in STATE_IDS.items()})
        id_to_name.update({v: k for k, v in DIFF_IDS.items()})
        id_to_name.update({v: f"COORD_{i}" for i, v in enumerate(range(COORD_BASE, COORD_BASE + NUM_COORD_BUCKETS))})
        id_to_name.update({v: f"STEP_{i}" for i, v in enumerate(range(TEMPORAL_BASE, TEMPORAL_BASE + NUM_TEMPORAL_BUCKETS))})
        id_to_name.update({v: k for k, v in TOPOLOGY_IDS.items()})
        id_to_name.update({v: k for k, v in CONTEXT_IDS.items()})

        labels = []
        for tid in token_ids.tolist():
            if isinstance(tid, list):
                tid = tid[0]
            label = id_to_name.get(tid, f"UNK_{tid}")
            if label != "PAD":
                labels.append(label)
        return labels
