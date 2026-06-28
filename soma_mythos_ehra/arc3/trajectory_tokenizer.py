"""Trajectory Tokenizer — maps environment trajectories to integer token sequences.

Token vocabulary:
  0:     [PAD]
  1:     [SOS] (start of sequence)
  2:     [EOS] (end of sequence)
  3:     [SEP] (separator between turns)
  4-10:  Actions (ACTION1-ACTION7 mapped to 4-10)
  11:    Reward=0 (no reward)
  12:    Reward=1 (goal reached)
  13-28: Grid state summary (16 cell values → 4-bit feature hash)
  29:    State=NOT_FINISHED
  30:    State=WIN
  31:    State=GAME_OVER
  32-97: Grammar primitives (from EXTENDED_PRIMITIVES, 66 tokens)
  98-107: Filters (from EXTENDED_FILTERS)
  108-119: Conditions (from EXTENDED_CONDITIONS)
  120-124: Compositions (from EXTENDED_COMPOSITIONS)
  125-188: Positional/numerical tokens (grid coordinates, sizes, counts)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


# Fixed token IDs
PAD = 0
SOS = 1
EOS = 2
SEP = 3
ACTION_OFFSET = 4      # Actions 1-7 → tokens 4-10
REWARD_BASE = 11       # Reward tokens
STATE_BASE = 29        # State tokens
GRAMMAR_BASE = 32      # Grammar tokens start here


@dataclass
class TokenMap:
    """Maps between semantic tokens and integer IDs."""
    action_to_id: dict[int, int] = None
    id_to_action: dict[int, int] = None
    grammar_to_id: dict[str, int] = None
    id_to_grammar: dict[int, str] = None
    vocab_size: int = 189

    def __post_init__(self):
        if self.action_to_id is None:
            self.action_to_id = {i: ACTION_OFFSET + i for i in range(1, 8)}
            self.id_to_action = {v: k for k, v in self.action_to_id.items()}
        if self.grammar_to_id is None:
            self._build_grammar_map()

    def _build_grammar_map(self):
        try:
            from soma_mythos_ehra.arc3.expanded_grammar import EXTENDED_TOKEN_VOCAB
            self.grammar_to_id = {
                tok: GRAMMAR_BASE + i
                for i, tok in enumerate(EXTENDED_TOKEN_VOCAB)
            }
            self.id_to_grammar = {v: k for k, v in self.grammar_to_id.items()}
            self.vocab_size = GRAMMAR_BASE + len(EXTENDED_TOKEN_VOCAB) + 64
        except ImportError:
            self.grammar_to_id = {}
            self.id_to_grammar = {}
            self.vocab_size = 189


# Global token map
TOKEN_MAP = TokenMap()


def tokenize_state(grid: np.ndarray) -> list[int]:
    """Summarize grid state into a short token sequence.

    Uses 4-bit feature hashing to compress grid information.
    Returns a fixed-length token sequence (8 tokens).
    """
    features = []

    # Feature 1: shape
    h, w = grid.shape
    features.append(13 + (h % 4) * 4 + (w % 4))

    # Feature 2: unique value count
    n_unique = len(np.unique(grid))
    features.append(13 + n_unique)

    # Feature 3: nonzero count ratio
    nz_ratio = np.count_nonzero(grid) / grid.size
    features.append(13 + int(nz_ratio * 15))

    # Feature 4: mean value
    features.append(13 + int(grid.mean()) % 16)

    # Feature 5: std value
    features.append(13 + int(grid.std()) % 16)

    # Feature 6: row symmetry
    if h > 1:
        row_sym = np.mean(grid == np.flip(grid, axis=0))
        features.append(13 + int(row_sym * 15))
    else:
        features.append(13)

    # Feature 7: col symmetry
    if w > 1:
        col_sym = np.mean(grid == np.flip(grid, axis=1))
        features.append(13 + int(col_sym * 15))
    else:
        features.append(13)

    # Feature 8: boundary complexity
    if h > 2 and w > 2:
        interior = grid[1:-1, 1:-1]
        boundary = grid[1:-1, 0:-2] != grid[1:-1, 1:-1]
        features.append(13 + min(int(boundary.sum()), 15))
    else:
        features.append(13)

    return [min(max(f, 13), 28) for f in features]


def tokenize_trajectory(
    prev_grids: list[np.ndarray],
    actions: list[int],
    rewards: list[float],
    states: list[str],
    grammar_tokens: list[int] | None = None,
) -> list[int]:
    """Convert a full trajectory into a token sequence.

    Format:
    [SOS] [state_features] [SEP] [action] [reward] [SEP] ... [EOS]

    If grammar_tokens are provided, they are appended at the end:
    ... [SEP] [grammar_token_1] [grammar_token_2] ... [EOS]
    """
    tokens = [SOS]

    for i in range(len(actions)):
        # State summary
        if i < len(prev_grids):
            state_tokens = tokenize_state(prev_grids[i])
            tokens.extend(state_tokens)

        tokens.append(SEP)

        # Action
        action_id = TOKEN_MAP.action_to_id.get(actions[i], ACTION_OFFSET)
        tokens.append(action_id)

        # Reward
        if i < len(rewards) and rewards[i] > 0:
            tokens.append(REWARD_BASE + 1)  # Reward=1
        else:
            tokens.append(REWARD_BASE)  # Reward=0

        # State label
        if i < len(states):
            state_label = {
                "NOT_FINISHED": STATE_BASE,
                "WIN": STATE_BASE + 1,
                "GAME_OVER": STATE_BASE + 2,
            }.get(states[i], STATE_BASE)
            tokens.append(state_label)

        tokens.append(SEP)

    # Append grammar tokens if provided (target for supervised training)
    if grammar_tokens:
        tokens.append(SEP)
        for gt in grammar_tokens:
            if gt in TOKEN_MAP.id_to_grammar:
                tokens.append(gt)
            elif gt < TOKEN_MAP.vocab_size:
                tokens.append(gt)

    tokens.append(EOS)
    return tokens


def detokenize_actions(token_ids: list[int]) -> list[int]:
    """Extract action IDs from a token sequence."""
    actions = []
    for tid in token_ids:
        if tid in TOKEN_MAP.id_to_action:
            actions.append(TOKEN_MAP.id_to_action[tid])
    return actions


def detokenize_grammar(token_ids: list[int]) -> list[str]:
    """Extract grammar token names from a token sequence."""
    grammar_tokens = []
    for tid in token_ids:
        if tid in TOKEN_MAP.id_to_grammar:
            grammar_tokens.append(TOKEN_MAP.id_to_grammar[tid])
    return grammar_tokens


def pad_sequence(tokens: list[int], max_len: int, pad_value: int = PAD) -> list[int]:
    """Pad a token sequence to max_len."""
    if len(tokens) >= max_len:
        return tokens[:max_len]
    return tokens + [pad_value] * (max_len - len(tokens))


def batch_tokenize(
    trajectories: list[dict],
    max_len: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-tokenize multiple trajectories.

    Args:
        trajectories: list of dicts with keys:
            prev_grids, actions, rewards, states, grammar_tokens (optional)
        max_len: maximum sequence length
    Returns:
        input_ids: (batch, max_len) tensor
        target_ids: (batch, max_len) tensor (shifted by 1)
    """
    input_seqs = []
    target_seqs = []

    for traj in trajectories:
        tokens = tokenize_trajectory(
            prev_grids=traj.get("prev_grids", []),
            actions=traj.get("actions", []),
            rewards=traj.get("rewards", []),
            states=traj.get("states", []),
            grammar_tokens=traj.get("grammar_tokens"),
        )

        # Input is everything except last token
        input_seq = pad_sequence(tokens[:-1], max_len)
        # Target is everything except first token
        target_seq = pad_sequence(tokens[1:], max_len)

        input_seqs.append(input_seq)
        target_seqs.append(target_seq)

    return torch.tensor(input_seqs, dtype=torch.long), torch.tensor(target_seqs, dtype=torch.long)
