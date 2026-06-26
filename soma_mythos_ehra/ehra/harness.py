from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from soma_mythos_ehra.mythos.search import MythosSearch
from soma_mythos_ehra.types import GridState, SearchDecision, normalize_actions


@dataclass(frozen=True)
class TelemetryEvent:
    run_id: str
    worker: str
    step: int
    action: int
    energy: float
    visits: int
    elapsed_ms: float
    reason: str


@dataclass(frozen=True)
class RuntimeResult:
    run_id: str
    actions: tuple[int, ...]
    final_state: GridState
    telemetry_path: Path


TransitionFn = Callable[[GridState, int], GridState]


class EHRARuntime:
    """Execution harness with action filtering and JSONL telemetry."""

    def __init__(
        self,
        search: MythosSearch,
        transition: TransitionFn | None = None,
        telemetry_dir: str | Path = "recordings",
        max_actions: int = 100,
        workers: int = 4,
    ) -> None:
        self.search = search
        self.transition = transition or self._simulator_transition
        self.telemetry_dir = Path(telemetry_dir)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.max_actions = max_actions
        self.workers = workers

    def run(self, initial_state: GridState, available_actions: Iterable[int] | None = None) -> RuntimeResult:
        run_id = f"monarch_ai.{uuid.uuid4()}"
        telemetry_path = self.telemetry_dir / f"{run_id}.jsonl"
        actions = normalize_actions(available_actions)
        state = initial_state.clone()
        chosen: list[int] = []

        for step in range(self.max_actions):
            started = time.perf_counter()
            decision = self.search.choose(state.grid, actions)
            action = self.filter_action(decision.action, actions)
            elapsed = (time.perf_counter() - started) * 1000.0
            self._write_event(
                telemetry_path,
                TelemetryEvent(
                    run_id=run_id,
                    worker="Thread-main",
                    step=step,
                    action=action,
                    energy=decision.energy,
                    visits=decision.visits,
                    elapsed_ms=elapsed,
                    reason=decision.reason,
                ),
            )
            chosen.append(action)
            state = self.transition(state, action)
            if state.done:
                break
        return RuntimeResult(run_id, tuple(chosen), state, telemetry_path)

    def evaluate_parallel(
        self,
        states: Iterable[GridState],
        available_actions: Iterable[int] | None = None,
    ) -> list[SearchDecision]:
        actions = normalize_actions(available_actions)
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="Thread") as pool:
            futures = [pool.submit(self.search.choose, state.grid, actions) for state in states]
            return [future.result() for future in as_completed(futures)]

    def filter_action(self, action: int, available_actions: tuple[int, ...]) -> int:
        return int(action if action in available_actions else available_actions[0])

    def _simulator_transition(self, state: GridState, action: int) -> GridState:
        grid, energy = self.search.simulator.step_batch(state.grid, torch.tensor([action]))
        done = bool(energy[0].item() == 0.0)
        return GridState(grid=grid[0].detach().cpu(), step=state.step + 1, score=state.score - float(energy[0]), done=done)

    def _write_event(self, path: Path, event: TelemetryEvent) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
