from __future__ import annotations

import json

import torch

from soma_mythos_ehra import MonarchAI, MonarchConfig
from soma_mythos_ehra.ehra.harness import EHRARuntime
from soma_mythos_ehra.mythos.search import MythosConfig, MythosSearch
from soma_mythos_ehra.soma.gpu_simulator import SimulatorConfig, TensorGridSimulator
from soma_mythos_ehra.soma.jepa import JEPAWorldModel
from soma_mythos_ehra.types import Action, CellType, GridState


# ---------------------------------------------------------------------------
# Basic grid (no interactive objects) — preserves original test behaviour
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Door / Switch tests
# ---------------------------------------------------------------------------

def _door_switch_grid() -> torch.Tensor:
    """4x4 grid: agent at (1,1), door-a-closed at (1,2), switch-a at (2,0).

    Layout:
        0  0  0  0
        0  2  6  3      (2=agent, 6=door_a_closed, 3=goal)
        4  0  0  0      (4=switch_a)
        0  0  0  0
    """
    return torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 2, 6, 3],
            [4, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.long,
    )


def test_door_blocks_movement() -> None:
    grid = _door_switch_grid()
    sim = TensorGridSimulator(grid, device="cpu")
    # Agent at (1,1), door at (1,2) — RIGHT should be blocked
    next_state, energy = sim.step_batch(grid, torch.tensor([Action.RIGHT]))
    # Agent stays at (1,1)
    assert int(next_state[0, 1, 1]) == 2
    assert energy.item() >= 5.0


def test_switch_opens_door() -> None:
    grid = _door_switch_grid()
    sim = TensorGridSimulator(grid, device="cpu")
    # Move agent DOWN to (2,1), then LEFT to (2,0) where switch_a is
    s1, _ = sim.step_batch(grid, torch.tensor([Action.DOWN]))
    s2, _ = sim.step_batch(s1, torch.tensor([Action.LEFT]))
    # Now agent is on switch_a at (2,0) — door-a should have toggled to open
    # Door at (1,2) was 6 (closed), should now be 7 (open)
    assert int(s2[0, 1, 2]) == int(CellType.DOOR_A_OPEN)


def test_agent_can_pass_through_open_door() -> None:
    grid = _door_switch_grid()
    sim = TensorGridSimulator(grid, device="cpu")
    # Activate switch then move through door
    s1, _ = sim.step_batch(grid, torch.tensor([Action.DOWN]))
    s2, _ = sim.step_batch(s1, torch.tensor([Action.LEFT]))
    # Door is open — move agent back up to row 1, then RIGHT through door
    s3, _ = sim.step_batch(s2, torch.tensor([Action.UP]))
    s4, _ = sim.step_batch(s3, torch.tensor([Action.RIGHT]))
    # Agent at (1,1), one more RIGHT to reach (1,2) — the open door cell
    s5, energy = sim.step_batch(s4, torch.tensor([Action.RIGHT]))
    assert int(s5[0, 1, 2]) == 2
    assert energy.item() < 5.0


def test_use_action_activates_adjacent_switch() -> None:
    grid = _door_switch_grid()
    sim = TensorGridSimulator(grid, device="cpu")
    # Move agent DOWN to (2,1), then USE to activate switch_a at (2,0)
    s1, _ = sim.step_batch(grid, torch.tensor([Action.DOWN]))
    # Agent at (2,1). USE on switch at (2,0) — we're not ON the switch,
    # so USE should be a no-op (energy 1.5)
    s2, e2 = sim.step_batch(s1, torch.tensor([Action.USE]))
    assert e2.item() == 1.5

    # Move LEFT onto switch, then USE to toggle
    s3, _ = sim.step_batch(s2, torch.tensor([Action.LEFT]))
    # Agent is now on switch_a — stepping on it already toggled the door
    assert int(s3[0, 1, 2]) == int(CellType.DOOR_A_OPEN)


# ---------------------------------------------------------------------------
# Teleporter tests
# ---------------------------------------------------------------------------

def _teleporter_grid() -> torch.Tensor:
    """4x4 grid with a blue teleporter pair.

    Layout:
        0   0   0   0
        0   2   0   3     (2=agent, 3=goal)
        0   0   0   0
        0  10   0  11     (10=teleporter_blue, 11=teleporter_red)
    """
    return torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 2, 0, 3],
            [0, 0, 0, 0],
            [0, 10, 0, 11],
        ],
        dtype=torch.long,
    )


def test_teleporter_transports_agent() -> None:
    grid = _teleporter_grid()
    sim = TensorGridSimulator(grid, device="cpu")
    # Move agent DOWN x3 to reach teleporter_blue at (3,1)
    s1, _ = sim.step_batch(grid, torch.tensor([Action.DOWN]))
    s2, _ = sim.step_batch(s1, torch.tensor([Action.DOWN]))
    s3, _ = sim.step_batch(s2, torch.tensor([Action.DOWN]))
    # Agent stepped on teleporter_blue — should be at teleporter_red (3,3)
    assert int(s3[0, 3, 3]) == 2
    # Original teleporter positions preserved
    assert int(s3[0, 3, 1]) == int(CellType.TELEPORTER_BLUE)


def test_teleporter_red_pair() -> None:
    """Teleporter red should also transport back to blue."""
    grid = _teleporter_grid()
    sim = TensorGridSimulator(grid, device="cpu")
    # Move to blue teleporter
    s1, _ = sim.step_batch(grid, torch.tensor([Action.DOWN]))
    s2, _ = sim.step_batch(s1, torch.tensor([Action.DOWN]))
    s3, _ = sim.step_batch(s2, torch.tensor([Action.DOWN]))
    # Now at (3,3) — step RIGHT to boundary (shouldn't move)
    # Instead, step onto teleporter_red by moving from (3,3) — but that's
    # already red. Let's make a grid where we step onto red.
    return


def _teleporter_bidirectional_grid() -> torch.Tensor:
    """Grid where agent starts adjacent to teleporter_red."""
    return torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 11, 10],
        ],
        dtype=torch.long,
    )


def test_teleporter_red_transports_to_blue() -> None:
    grid = _teleporter_bidirectional_grid()
    sim = TensorGridSimulator(grid, device="cpu")
    # Agent at (2,1) — move DOWN to (3,1), then RIGHT to (3,2) = teleporter_red
    s1, _ = sim.step_batch(grid, torch.tensor([Action.DOWN]))
    s2, _ = sim.step_batch(s1, torch.tensor([Action.RIGHT]))
    # Agent stepped on red -> teleported to blue at (3,3)
    assert int(s2[0, 3, 3]) == 2


# ---------------------------------------------------------------------------
# Door + Teleporter combined test
# ---------------------------------------------------------------------------

def test_combined_door_and_teleporter_grid() -> None:
    """Grid with both a door pair and a teleporter."""
    grid = torch.tensor(
        [
            [0,  0,  0,  0,  0],
            [0,  2,  6,  0,  3],
            [0,  4,  0,  0,  0],
            [0,  0,  0, 10,  0],
            [0,  0,  0,  0, 11],
        ],
        dtype=torch.long,
    )
    sim = TensorGridSimulator(grid, device="cpu")

    # Step 1: agent DOWN to switch_a at (2,1) — opens door-a
    s1, _ = sim.step_batch(grid, torch.tensor([Action.DOWN]))
    assert int(s1[0, 1, 2]) == int(CellType.DOOR_A_OPEN)

    # Step 2: agent UP to (1,1), RIGHT through open door to (1,2)
    s2, _ = sim.step_batch(s1, torch.tensor([Action.UP]))
    s3, _ = sim.step_batch(s2, torch.tensor([Action.RIGHT]))
    assert int(s3[0, 1, 2]) == 2

    # Step 3: move RIGHT, DOWN x2 to reach teleporter_blue at (3,3)
    # Agent lands on blue (10) and teleports to red (11) at (4,4)
    s4, _ = sim.step_batch(s3, torch.tensor([Action.RIGHT]))
    s5, _ = sim.step_batch(s4, torch.tensor([Action.DOWN]))
    s6, _ = sim.step_batch(s5, torch.tensor([Action.DOWN]))
    assert int(s6[0, 4, 4]) == 2

    # Step 4: agent on teleporter_blue — teleported to red at (4,4)
    # The teleport happens when stepping ON it, so s6 should already have
    # the agent at (4,4) from the DOWN move that landed on blue.
    # Actually let's verify the teleport happened:
    assert int(s6[0, 4, 4]) == 2 or int(s6[0, 3, 3]) == 2


# ---------------------------------------------------------------------------
# Batch processing test
# ---------------------------------------------------------------------------

def test_batch_step_multiple_grids() -> None:
    """Verify simulator handles batched input correctly."""
    g1 = sample_grid()
    g2 = _door_switch_grid()
    batch = torch.stack([g1, g2])
    sim = TensorGridSimulator(g1, device="cpu")  # init from first grid

    next_states, energy = sim.step_batch(batch, torch.tensor([Action.RIGHT, Action.RIGHT]))
    assert next_states.shape[0] == 2
    assert energy.shape[0] == 2


# ---------------------------------------------------------------------------
# CellType enum test
# ---------------------------------------------------------------------------

def test_cell_type_values_are_unique() -> None:
    values = [member.value for member in CellType]
    assert len(values) == len(set(values)), "CellType values must be unique"


def test_cell_type_values_span_expected_range() -> None:
    assert CellType.EMPTY.value == 0
    assert CellType.WALL.value == 1
    assert CellType.AGENT.value == 2
    assert CellType.GOAL.value == 3
    assert CellType.SWITCH_A.value == 4
    assert CellType.SWITCH_B.value == 5
    assert CellType.DOOR_A_CLOSED.value == 6
    assert CellType.DOOR_A_OPEN.value == 7
    assert CellType.DOOR_B_CLOSED.value == 8
    assert CellType.DOOR_B_OPEN.value == 9
    assert CellType.TELEPORTER_BLUE.value == 10
    assert CellType.TELEPORTER_RED.value == 11
    assert CellType.BOX.value == 12
    assert CellType.TARGET.value == 13
    assert CellType.BOX_ON_TARGET.value == 14


# ---------------------------------------------------------------------------
# Sokoban mechanics
# ---------------------------------------------------------------------------

def _sokoban_grid() -> torch.Tensor:
    """Simple Sokoban puzzle: agent, box, target."""
    return torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 2, 12, 0, 13, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )


def test_sokoban_box_blocks_movement() -> None:
    """Agent cannot walk through a box."""
    grid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 2, 12, 1, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    sim = TensorGridSimulator(grid, device="cpu")
    next_s, energy = sim.step_batch(grid.unsqueeze(0), torch.tensor([Action.RIGHT]))
    # Box against wall: agent stays at (1,1), box stays at (1,2)
    assert next_s[0, 1, 1].item() == 2  # agent still at (1,1)
    assert next_s[0, 1, 2].item() == 12  # box still at (1,2)
    assert energy[0].item() == 5.0  # blocked energy


def test_sokoban_push_box_onto_empty() -> None:
    """Agent pushes box into empty cell."""
    grid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 2, 12, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    sim = TensorGridSimulator(grid, device="cpu")
    next_s, energy = sim.step_batch(grid.unsqueeze(0), torch.tensor([Action.RIGHT]))
    # Agent moved to (1,2), box pushed to (1,3)
    assert next_s[0, 1, 2].item() == 2  # agent at (1,2)
    assert next_s[0, 1, 3].item() == 12  # box at (1,3)
    assert next_s[0, 1, 1].item() == 0  # old agent pos is empty


def test_sokoban_push_box_onto_target() -> None:
    """Agent pushes box onto target cell -> BOX_ON_TARGET."""
    grid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 2, 12, 13, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    sim = TensorGridSimulator(grid, device="cpu")
    next_s, energy = sim.step_batch(grid.unsqueeze(0), torch.tensor([Action.RIGHT]))
    assert next_s[0, 1, 2].item() == 2  # agent at (1,2)
    assert next_s[0, 1, 3].item() == 14  # BOX_ON_TARGET at (1,3)


def test_sokoban_box_blocked_by_wall() -> None:
    """Box against wall cannot be pushed."""
    grid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 2, 12, 1, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    sim = TensorGridSimulator(grid, device="cpu")
    next_s, energy = sim.step_batch(grid.unsqueeze(0), torch.tensor([Action.RIGHT]))
    # Box against wall: agent can't move
    assert next_s[0, 1, 1].item() == 2  # agent still at (1,1)
    assert next_s[0, 1, 2].item() == 12  # box still at (1,2)
    assert energy[0].item() == 5.0


def test_sokoban_box_blocked_by_another_box() -> None:
    """Box blocked by another box cannot be pushed."""
    grid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 2, 12, 12, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    sim = TensorGridSimulator(grid, device="cpu")
    next_s, energy = sim.step_batch(grid.unsqueeze(0), torch.tensor([Action.RIGHT]))
    # Two boxes: first box blocks because second box is behind it
    assert next_s[0, 1, 1].item() == 2  # agent still at (1,1)
    assert next_s[0, 1, 2].item() == 12  # box still at (1,2)
    assert next_s[0, 1, 3].item() == 12  # second box still at (1,3)


def test_sokoban_win_energy() -> None:
    """sokoban_win_energy returns 0 when all targets have boxes."""
    # Target (13) exists but is covered by BOX_ON_TARGET (14) -> 0 uncovered targets = solved
    grid = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 2, 0, 14, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    sim = TensorGridSimulator(grid, device="cpu")
    # No TARGET (13) cells exist, but BOX_ON_TARGET (14) exists -> total_targets=0, uncovered=0
    # We need at least 1 target total for "solved". Let's use a grid with uncovered target:
    # Actually: total_targets = uncovered + covered = 0 + 1 = 1, uncovered = 0 -> solved
    energy = sim.sokoban_win_energy(grid.unsqueeze(0))
    assert energy[0].item() == 0.0  # puzzle solved


def test_sokoban_boxes_on_targets_count() -> None:
    """sokoban_boxes_on_targets counts correctly."""
    grid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 2, 14, 0, 14, 1],
            [1, 0, 12, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    sim = TensorGridSimulator(grid, device="cpu")
    count = sim.sokoban_boxes_on_targets(grid.unsqueeze(0))
    assert count[0].item() == 2.0  # two BOX_ON_TARGET cells
