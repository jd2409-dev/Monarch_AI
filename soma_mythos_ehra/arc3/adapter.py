"""ARC-AGI 3 Puzzle Adapter — loads JSON task files into tensors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class ARC3Task:
    """Represents a single ARC-AGI puzzle task."""

    def __init__(self, task_id: str, task_data: dict[str, Any]) -> None:
        self.task_id = task_id
        self.train_pairs = task_data.get("train", [])
        self.test_pairs = task_data.get("test", [])

    @classmethod
    def from_file(cls, path: str | Path) -> ARC3Task:
        path = Path(path)
        task_id = path.stem
        with open(path, "r") as f:
            data = json.load(f)
        return cls(task_id, data)

    def get_train_inputs(self) -> list[torch.Tensor]:
        return [torch.tensor(pair["input"], dtype=torch.long) for pair in self.train_pairs]

    def get_train_outputs(self) -> list[torch.Tensor]:
        return [torch.tensor(pair["output"], dtype=torch.long) for pair in self.train_pairs]

    def get_test_input(self) -> torch.Tensor | None:
        if self.test_pairs:
            return torch.tensor(self.test_pairs[0]["input"], dtype=torch.long)
        return None

    def get_test_output(self) -> torch.Tensor | None:
        if self.test_pairs and "output" in self.test_pairs[0]:
            return torch.tensor(self.test_pairs[0]["output"], dtype=torch.long)
        return None

    def num_colors(self) -> int:
        all_vals = []
        for pair in self.train_pairs:
            all_vals.extend(v for row in pair["input"] for v in row)
            all_vals.extend(v for row in pair["output"] for v in row)
        return max(all_vals) + 1 if all_vals else 10

    def __repr__(self) -> str:
        return f"ARC3Task({self.task_id}, train={len(self.train_pairs)}, test={len(self.test_pairs)})"


def load_tasks_from_dir(directory: str | Path, limit: int | None = None) -> list[ARC3Task]:
    """Load all .json task files from a directory."""
    directory = Path(directory)
    tasks = []
    for json_file in sorted(directory.glob("*.json")):
        tasks.append(ARC3Task.from_file(json_file))
        if limit and len(tasks) >= limit:
            break
    return tasks
