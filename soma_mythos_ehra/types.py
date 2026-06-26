from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import torch


class Action(IntEnum):
    RESET = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    USE = 5
    CLICK = 6
    UNDO = 7


DIRECTION_DELTAS: dict[Action, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


@dataclass(frozen=True)
class GridState:
    grid: torch.Tensor
    step: int = 0
    score: float = 0.0
    done: bool = False

    def clone(self) -> "GridState":
        return GridState(
            grid=self.grid.clone(),
            step=self.step,
            score=self.score,
            done=self.done,
        )


@dataclass(frozen=True)
class SearchDecision:
    action: int
    energy: float
    visits: int
    sequence: tuple[int, ...]
    reason: str


def normalize_actions(actions: Iterable[int] | None, max_action: int = 7) -> tuple[int, ...]:
    if actions is None:
        return tuple(range(1, max_action + 1))
    clean = sorted({int(a) for a in actions if 0 <= int(a) <= max_action})
    return tuple(clean) if clean else tuple(range(1, max_action + 1))
