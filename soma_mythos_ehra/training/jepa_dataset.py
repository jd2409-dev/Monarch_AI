from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

# Grid symbol mapping from harness
SYMBOL_MAP: dict[int, int] = {v: k for k, v in {
    "empty": 0,
    "agent": 1,
    "goal": 2,
    "open_door": 3,
    "closed_door": 4,
    "switch_off": 5,
    "switch_on": 6,
    "lamp_off": 7,
    "lamp_on": 8,
    "box": 9,
    "blocked": 10,
    "key": 11,
    "teleporter": 12,
    "cooler": 13,
    "heater": 14,
    "trigger_zone": 15,
    "counter": 16,
}.items()}

# Reverse map for display
SYMBOL_NAMES: dict[int, str] = {v: k for k, v in {
    "empty": 0,
    "agent": 1,
    "goal": 2,
    "open_door": 3,
    "closed_door": 4,
    "switch_off": 5,
    "switch_on": 6,
    "lamp_off": 7,
    "lamp_on": 8,
    "box": 9,
    "blocked": 10,
    "key": 11,
    "teleporter": 12,
    "cooler": 13,
    "heater": 14,
    "trigger_zone": 15,
    "counter": 16,
}.items()}

# Default grid dimensions
DEFAULT_GRID_H = 12
DEFAULT_GRID_W = 12

# Action mapping from ARC-AGI-3
ACTION_MAP: dict[str, int] = {
    "noop": 0,
    "toggle_up": 1,
    "toggle_down": 2,
    "toggle_left": 3,
    "toggle_right": 4,
    "toggle_open": 5,
}


def _find_agent_pos(frame: list[list[int]]) -> tuple[int, int] | None:
    """Find agent position (value 1) in a 2D frame."""
    for r, row in enumerate(frame):
        for c, val in enumerate(row):
            if val == 1:
                return (r, c)
    return None


def _find_diff_action(
    prev_frame: list[list[int]],
    curr_frame: list[list[int]],
) -> int | None:
    """Detect action by comparing two consecutive frames.

    Returns action index or None if no change detected.
    """
    prev_agent = _find_agent_pos(prev_frame)
    curr_agent = _find_agent_pos(curr_frame)

    if prev_agent is None or curr_agent is None:
        return None

    pr, pc = prev_agent
    cr, cc = curr_agent

    if cr < pr:
        return 1  # toggle_up
    elif cr > pr:
        return 2  # toggle_down
    elif cc < pc:
        return 3  # toggle_left
    elif cc > pc:
        return 4  # toggle_right

    # Check for toggle (cell value changes without position change)
    for r in range(len(prev_frame)):
        for c in range(len(prev_frame[0])):
            if prev_frame[r][c] != curr_frame[r][c]:
                return 5  # toggle_open

    return 0  # noop


def _pad_grid(grid: list[list[int]], target_h: int, target_w: int) -> list[list[int]]:
    """Pad grid to target dimensions with zeros (empty)."""
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    padded = [[0] * target_w for _ in range(target_h)]
    for r in range(min(h, target_h)):
        for c in range(min(w, target_w)):
            padded[r][c] = grid[r][c]
    return padded


def parse_recording(
    recording_path: Path,
    target_h: int = DEFAULT_GRID_H,
    target_w: int = DEFAULT_GRID_W,
) -> list[tuple[torch.Tensor, int, torch.Tensor]]:
    """Parse an ARC recording JSONL file into (state, action, next_state) triples.

    Each line in the recording is a JSON object with:
    - timestamp: ISO timestamp
    - data.game_id: game identifier
    - data.frame: 3D array (the grid state)
    """
    transitions: list[tuple[torch.Tensor, int, torch.Tensor]] = []
    frames: list[list[list[int]]] = []

    with recording_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "data" not in entry or "frame" not in entry.get("data", {}):
                continue

            frame = entry["data"]["frame"]
            if not isinstance(frame, list) or len(frame) == 0:
                continue

            # Ensure 2D (take first channel if 3D)
            if isinstance(frame[0], list) and isinstance(frame[0][0], list):
                frame = frame[0]

            frames.append(frame)

    # Build transitions from consecutive frames
    for i in range(len(frames) - 1):
        state = frames[i]
        next_state = frames[i + 1]

        action = _find_diff_action(state, next_state)
        if action is None:
            action = 0  # noop

        # Pad and convert to tensors
        state_padded = _pad_grid(state, target_h, target_w)
        next_state_padded = _pad_grid(next_state, target_h, target_w)

        state_tensor = torch.tensor(state_padded, dtype=torch.long)
        next_state_tensor = torch.tensor(next_state_padded, dtype=torch.long)

        transitions.append((state_tensor, action, next_state_tensor))

    return transitions


class ARCRecordingDataset(Dataset):
    """Dataset of (state, action, next_state) transitions from ARC recordings."""

    def __init__(
        self,
        recordings_dir: Path,
        target_h: int = DEFAULT_GRID_H,
        target_w: int = DEFAULT_GRID_W,
    ) -> None:
        self.transitions: list[tuple[torch.Tensor, int, torch.Tensor]] = []
        self.target_h = target_h
        self.target_w = target_w

        recording_files = sorted(recordings_dir.glob("*.jsonl"))
        for rec_file in recording_files:
            transitions = parse_recording(rec_file, target_h, target_w)
            self.transitions.extend(transitions)

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, action, next_state = self.transitions[idx]
        return state, torch.tensor(action, dtype=torch.long), next_state


def create_contrastive_pairs(
    batch_states: torch.Tensor,
    batch_actions: torch.Tensor,
    batch_next_states: torch.Tensor,
    num_negatives: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create contrastive training pairs.

    Returns:
        states: (B,) current states
        actions: (B,) actions taken
        pos_states: (B,) true next states
        neg_states: (B, num_negatives) negative next states
        labels: (B,) ones for positives
    """
    B = batch_states.shape[0]
    device = batch_states.device

    # Sample random negatives from the batch
    neg_indices = torch.randint(0, B, (B, num_negatives), device=device)
    neg_states = batch_next_states[neg_indices]

    return batch_states, batch_actions, batch_next_states, neg_states, torch.ones(B, device=device)
