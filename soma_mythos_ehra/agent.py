from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from soma_mythos_ehra.ehra.harness import EHRARuntime, RuntimeResult
from soma_mythos_ehra.mythos.search import MythosConfig, MythosSearch
from soma_mythos_ehra.soma.gpu_simulator import TensorGridSimulator
from soma_mythos_ehra.soma.jepa import JEPAWorldModel
from soma_mythos_ehra.types import GridState


@dataclass(frozen=True)
class MonarchConfig:
    agent_name: str = "Monarch_AI"
    max_actions: int = 100
    horizon: int = 24
    simulations: int = 256
    num_symbols: int = 17
    num_actions: int = 8
    latent_dim: int = 64
    exploration: float = 3.5
    cycle_penalty: float = 15.0
    tabu_window: int = 8
    tabu_revisit_penalty: float = 25.0
    model_path: str | None = "checkpoints/best_jepa.pt"


class MonarchAI:
    """Composable SOMA-Mythos-EHRA runner."""

    def __init__(self, config: MonarchConfig | None = None, device: str | None = None) -> None:
        self.config = config or MonarchConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def build_runtime(self, initial_grid: torch.Tensor) -> EHRARuntime:
        initial_grid = self._normalize_observation(initial_grid)
        simulator = TensorGridSimulator(initial_grid, device=self.device)
        world_model = JEPAWorldModel(
            num_symbols=self.config.num_symbols,
            num_actions=self.config.num_actions,
            latent_dim=self.config.latent_dim,
        )
        if self.config.model_path is not None:
            path = Path(self.config.model_path)
            if path.exists():
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                world_model.load_state_dict(ckpt["model_state_dict"])
                world_model.to(self.device)
                world_model.eval()
        search = MythosSearch(
            simulator=simulator,
            world_model=world_model,
            config=MythosConfig(
                horizon=self.config.horizon,
                simulations=self.config.simulations,
                exploration=self.config.exploration,
                cycle_penalty=self.config.cycle_penalty,
                tabu_window=self.config.tabu_window,
                tabu_revisit_penalty=self.config.tabu_revisit_penalty,
            ),
        )
        return EHRARuntime(search=search, max_actions=self.config.max_actions)

    def solve(self, initial_grid: torch.Tensor, available_actions: tuple[int, ...] | None = None) -> RuntimeResult:
        initial_grid = self._normalize_observation(initial_grid)
        runtime = self.build_runtime(initial_grid)
        return runtime.run(GridState(initial_grid.detach().cpu()), available_actions=available_actions)

    def _normalize_observation(self, observation: torch.Tensor) -> torch.Tensor:
        grid = observation.detach().cpu()
        if grid.ndim == 2:
            return grid.long()
        if grid.ndim == 3 and grid.shape[0] == 1:
            return grid[0].long()
        if grid.ndim == 3 and grid.shape[0] == grid.shape[1]:
            return grid.float().argmax(dim=-1).long()
        if grid.ndim == 3 and grid.shape[1] == grid.shape[2]:
            return grid.float().argmax(dim=0).long()
        raise ValueError(f"unsupported observation shape: {tuple(grid.shape)}")
