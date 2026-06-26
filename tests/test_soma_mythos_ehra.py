from __future__ import annotations

import json

import torch

from soma_mythos_ehra import MonarchAI, MonarchConfig
from soma_mythos_ehra.ehra.harness import EHRARuntime
from soma_mythos_ehra.mythos.search import MythosConfig, MythosSearch
from soma_mythos_ehra.soma.gpu_simulator import TensorGridSimulator
from soma_mythos_ehra.soma.jepa import JEPAWorldModel
from soma_mythos_ehra.types import Action, GridState


def sample_grid() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 2, 0, 3],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.long,
    )


def test_tensor_simulator_rejects_wall_collision() -> None:
    simulator = TensorGridSimulator(sample_grid(), device="cpu")
    next_states, energy = simulator.step_batch(sample_grid(), torch.tensor([Action.DOWN]))

    assert int(next_states[0, 1, 1]) == 2
    assert energy.item() >= 5.0


def test_tensor_simulator_reaches_goal_with_low_energy() -> None:
    simulator = TensorGridSimulator(sample_grid(), device="cpu")
    state, energy = simulator.step_batch(sample_grid(), torch.tensor([Action.RIGHT]))
    state, energy = simulator.step_batch(state, torch.tensor([Action.RIGHT]))

    assert energy.item() == 0.0
    assert int(state[0, 1, 3]) == 2


def test_mythos_search_returns_valid_action() -> None:
    grid = sample_grid()
    simulator = TensorGridSimulator(grid, device="cpu")
    world_model = JEPAWorldModel(latent_dim=16)
    search = MythosSearch(simulator, world_model, MythosConfig(horizon=4, simulations=16))

    decision = search.choose(grid, (1, 2, 3, 4, 5))

    assert decision.action in (1, 2, 3, 4, 5)
    assert decision.visits > 0
    assert decision.sequence


def test_ehra_runtime_writes_jsonl_telemetry(tmp_path) -> None:
    grid = sample_grid()
    simulator = TensorGridSimulator(grid, device="cpu")
    world_model = JEPAWorldModel(latent_dim=16)
    search = MythosSearch(simulator, world_model, MythosConfig(horizon=4, simulations=8))
    runtime = EHRARuntime(search, telemetry_dir=tmp_path, max_actions=3)

    result = runtime.run(GridState(grid), available_actions=(1, 2, 3, 4, 5))

    assert result.telemetry_path.exists()
    events = [json.loads(line) for line in result.telemetry_path.read_text().splitlines()]
    assert events
    assert events[0]["worker"] == "Thread-main"
    assert all(event["action"] in (1, 2, 3, 4, 5) for event in events)


def test_monarch_ai_solves_with_clean_api(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    agent = MonarchAI(MonarchConfig(max_actions=4, simulations=8, horizon=4, latent_dim=16), device="cpu")

    result = agent.solve(sample_grid(), available_actions=(1, 2, 3, 4, 5))

    assert result.actions
    assert result.telemetry_path.exists()
