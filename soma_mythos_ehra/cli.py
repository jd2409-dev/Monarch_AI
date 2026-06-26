from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from soma_mythos_ehra.agent import MonarchAI, MonarchConfig


def load_grid(path: Path) -> torch.Tensor:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("grid") or data.get("frame")
    return torch.tensor(data, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Monarch_AI SOMA-Mythos-EHRA on a grid JSON file.")
    parser.add_argument("grid", type=Path, help="JSON file containing a 2D grid or {'grid': ...}.")
    parser.add_argument("--max-actions", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    agent = MonarchAI(MonarchConfig(max_actions=args.max_actions, simulations=args.simulations, horizon=args.horizon))
    result = agent.solve(load_grid(args.grid))
    print(json.dumps({"agent": agent.config.agent_name, "actions": result.actions, "telemetry": str(result.telemetry_path)}))


if __name__ == "__main__":
    main()
