# SOMA-Mythos-EHRA / Monarch AI

A hybrid artificial intelligence architecture for non-linguistic, open-ended,
fluid inductive reasoning. Monarch AI is the reference implementation of the
**SOMA-Mythos-EHRA** (State-Observation-Model Architecture /
Metacognitive-Omnithos-Harness) framework.

## Architecture Overview

```text
+---------------------------------------------------------------+
|             EHRA (Execution & Tracking)                       |
|  - Manages Async Multi-Threaded Engine Worker Pools           |
|  - Tracks Global State / Unified Telemetry Parsing            |
+----------------------------+----------------------------------+
                             |
                 State       |  Lookahead
                 Sync        |  Batches
                             v
+---------------------------------------------------------------+
|             MYTHOS / OMNITHOS (Search)                        |
|  - Metacognitive Controller & Paced-Search Execution          |
|  - Multi-Threaded SMNS/MCTS Search Tree Topology              |
+----------------------------+----------------------------------+
                             |
                  Heuristic  |  Parallel Latent
                  Gradients   |  Grid Projections
                             v
+---------------------------------------------------------------+
|             SOMA / JEPA (Core Brain)                          |
|  - Joint Embedding Predictive Representation                  |
|  - Massively Parallel VRAM Physics Engine / Wall Masks        |
+---------------------------------------------------------------+
```

## Components

### SOMA (State-Observation-Model Architecture)

SOMA acts as the sub-symbolic "visual cortex" and physics simulator. It maps
2D spatial layouts directly into structural latent variables without
pixel-reconstruction.

**JEPA Vector Spaces (Joint Embedding Predictive Architecture):** An Encoder
maps a 4-channel grid into a compact latent space, while a Predictor network
forecasts the downstream latent space (z_{t+1}) given an abstract action token
(a_t). This completely bypasses generative pixel-reconstruction, saving
capacity on irrelevant details.

**Energy-Based Evaluation (EBM):** The JEPA network outputs a scalar Energy
value (E). Correct, mathematically sound, or closer-to-goal states yield
minimal energy (E -> 0), whereas invalid configurations or spatial collisions
yield high energy.

**VRAM Tensor Simulator (`gpu_simulator.py`):** Physical interaction rules
are vectorized into PyTorch. Features like `wall_mask` and coordinate maps
are checked as concurrent tensor arrays on GPU, allowing thousands of forward
simulations to run concurrently.

**Multi-Channel Grid Representation:** The simulator uses a 4-channel tensor
layout for rich spatial encoding:

| Channel | Name        | Contents                                              |
|---------|-------------|-------------------------------------------------------|
| 0       | Wall        | Static walls, borders (value 1 in remapped grid)      |
| 1       | Interactive | Targets (9), doors, switches, teleporters             |
| 2       | Dynamic     | Agent position (value 2 in remapped grid)             |
| 3       | Floor       | Walkable cells: empty, floor, boxes, platforms        |

**Sokoban Box-Pushing Mechanics:** The simulator implements Sokoban-style
puzzle mechanics:

- **Box-pushing**: Agent pushes box into empty/target cells
- **Chain pushes**: Only one box at a time (standard Sokoban rules)
- **BOX_ON_TARGET**: Box becomes value 14 when pushed onto a target (13)
- **Win detection**: All targets covered by boxes = puzzle solved
- **Collision detection**: Boxes block movement unless pushable (empty/target
  behind, not wall/another box)

**CellType Enum (15 values):**

| Value | Name             | Description                    |
|-------|------------------|--------------------------------|
| 0     | EMPTY            | Empty cell                     |
| 1     | WALL             | Impassable wall                |
| 2     | AGENT            | Player position                |
| 3     | GOAL             | Goal/target position           |
| 4     | SWITCH_A         | Toggle switch (pair A)         |
| 5     | SWITCH_B         | Toggle switch (pair B)         |
| 6     | DOOR_A_CLOSED    | Closed door (pair A)           |
| 7     | DOOR_A_OPEN      | Open door (pair A)             |
| 8     | DOOR_B_CLOSED    | Closed door (pair B)           |
| 9     | DOOR_B_OPEN      | Open door (pair B)             |
| 10    | TELEPORTER_BLUE  | Blue teleporter pair           |
| 11    | TELEPORTER_RED   | Red teleporter pair            |
| 12    | BOX              | Pushable box                   |
| 13    | TARGET           | Sokoban target position        |
| 14    | BOX_ON_TARGET    | Box placed on target           |

### Mythos / Omnithos (Metacognitive Search Layer)

Mythos provides strategic logical reasoning and orchestrates non-linear
lookahead tree generation.

**Paced MCTS Solver:** An advanced variant of Monte Carlo Tree Search
constructs macro-action lookahead paths up to the configured horizon. The
search queries the SOMA simulator to evaluate branches in parallel batches.

**Energy Weighting:** The search combines three energy signals:
```
combined = physics_energy + 0.1 * latent_energy + 10.0 * goal_energy
```
- `physics_energy`: Simulator collision/distance signals (1.0 for free moves,
  5.0 for blocked moves)
- `latent_energy`: JEPA prediction error (weighted low at 0.1x due to
  synthetic training noise)
- `goal_energy`: Distance to goal/target positions (weighted high at 10.0x
  to drive exploration toward objectives)

**Metacognitive Controller:** Continuously evaluates search progress via an
explicit feedback loop. If a path-finding heuristic hits a local minimum or
dead end, the controller intercepts the state exception and executes a broad
exploratory MCTS fallback routine.

**Self-Reflection Layer:** Reviews recent action/energy deltas to identify
cycle oscillation (e.g., an agent pacing back and forth between two tiles).
If a structural trap is detected, the controller actively adjusts the policy
filter to drop the node's choice weight.

### EHRA (Environment-Harness Runtime Agent)

EHRA is the deterministic orchestration infrastructure that wraps the
cognitive pipeline, protecting it from runtime failure and linking it to the
outside world.

**Asynchronous Worker Pools:** Isolates distinct game tasks into individual,
decoupled worker processes. If a configuration hits an abstract exception or
constraint failure, the main loop catches the event, updates telemetry, and
isolates the thread to preserve the global runner state.

**Telemetry Logging:** Pipelines fine-grained operational states directly to
a unified JSONL tracking structure (`recordings/*.jsonl`). Logs raw frame
states, exact execution timings, and algorithmic reasoning paths.

**Unified Action Filtering:** Translates raw abstract search vectors into
concrete integer action tokens validated by the host evaluation environment.
Filters unauthorized tokens and tracks strict internal level constraints.

## ARC-AGI-3 Integration

### Grid Mapping

The ARC-AGI-3 game uses different cell values than the internal simulator.
The `remap_arc_grid` function translates between them:

| ARC Value | Meaning       | Simulator Value | Simulator Name  |
|-----------|---------------|-----------------|-----------------|
| 0         | Empty         | 0               | EMPTY           |
| 1         | Agent         | 2               | AGENT           |
| 3         | Floor         | 0               | EMPTY           |
| 4         | Wall          | 1               | WALL            |
| 5         | Border        | 1               | WALL            |
| 8         | Box           | 12              | BOX             |
| 9         | Target        | 13              | TARGET          |
| 11        | Corridor      | 0               | EMPTY           |
| 12        | Platform      | 0               | EMPTY           |

**Key Finding:** The grid received from the game server is **static** --
the "agent" value 1 at (32,20)/(33,21) never moves across frames. The actual
player position is tracked server-side and is invisible in the grid data.
Only decorative elements (platforms at rows 45-46) oscillate between frames.

### Heuristic Exploration Strategy

Since the player position is invisible, the agent uses a heuristic approach:

1. **Estimated Position Tracking**: Starts at (32,20) based on grid analysis,
   tracks estimated position changes based on actions sent
2. **Walkable Cell Validation**: Uses the remapped grid to validate moves
   (cells with values 0, 2, 3, 12, 13 are walkable)
3. **Exploration Priority**: Right > Down > Left > Up (explore right first,
   then descend to lower levels)
4. **Visited Set**: Avoids revisiting estimated positions
5. **Periodic USE Action**: Every 20 steps, sends USE (ACTION5) to interact
   with the environment
6. **Oscillation Detection**: If the agent detects a 2-action repeating
   pattern over 6+ steps, switches to breakout mode

### Game Structure (ls20)

The ls20 game is a 64x64 grid with 7 levels stacked vertically:

- **Rows 0-10**: Level 1 area (walled rooms with boxes and targets)
- **Rows 11-30**: Main play area with multiple rooms
- **Rows 31-44**: Agent starting position at (32,20)/(33,21)
- **Rows 45-49**: Oscillating platform/target zone (decorative)
- **Rows 50-55**: Transition zone
- **Rows 56-60**: Level endpoint targets (rows 55-60, cols 3-9)
- **Rows 61-62**: Corridor connecting levels (value 11)
- **Row 63**: Border wall

**Cell Value Distribution:**
- 894 floor cells (value 3)
- 2609 wall cells (value 4)
- 439 border cells (value 5)
- 12 box cells (value 8)
- 45 target cells (value 9)
- 82 corridor cells (value 11)
- 10 platform cells (value 12)

## Core Algorithmic Coordination

1. **State Ingestion:** EHRA initializes a raw observation layout, spinning
   up dedicated worker threads for parallel puzzles.
2. **Multi-Channel Encoding:** SOMA converts the raw 64x64 grid into a
   4-channel tensor (wall, interactive, dynamic, floor) for the JEPA encoder.
3. **GPU Boundary Mapping:** SOMA reads the task rules, isolates structural
   features, and uploads an immutable `wall_mask` array directly to GPU
   memory.
4. **Tree Search Generation:** The Mythos layer initiates a paced lookahead
   tree. It groups hundreds of macro-moves into a parallel batch and throws
   them across the PCIe bus to SOMA's VRAM engine.
5. **Energy Feedback Routing:** The JEPA evaluator calculates energy profiles
   across the generated latent projection branches. These gradients feed
   back into the Mythos MCTS tracker to refine path weights.
6. **Action Dispatch:** The meta-controller selects the absolute
   lowest-energy sequence root, passes the clean scalar token back across
   the hardware boundary to EHRA, and updates the telemetry harness.

## Install

Requires Python >= 3.11 and PyTorch >= 2.0.

```bash
python -m pip install -e ".[dev]"
```

## Usage

### CLI

Create a grid JSON with `0` empty cells, `1` walls, `2` the agent, and `3`
the goal:

```json
{"grid": [[0, 0, 0], [2, 0, 3], [0, 0, 0]]}
```

Run:

```bash
monarch-ai sample_grid.json --max-actions 8 --simulations 32
```

### Python API

```python
import torch
from soma_mythos_ehra import MonarchAI, MonarchConfig

grid = torch.tensor([[0, 0, 0], [2, 0, 3], [0, 0, 0]], dtype=torch.long)
agent = MonarchAI(MonarchConfig(max_actions=50, simulations=128))
result = agent.solve(grid)

print(result.actions)        # sequence of action tokens
print(result.telemetry_path) # path to JSONL telemetry log
```

### Sokoban Puzzle Example

```python
import torch
from soma_mythos_ehra.soma.gpu_simulator import TensorGridSimulator

# Simple Sokoban: agent, box, target
grid = torch.tensor([
    [1, 1, 1, 1, 1, 1],
    [1, 2, 12, 0, 13, 1],  # agent(2), box(12), empty(0), target(13)
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
], dtype=torch.long)

sim = TensorGridSimulator(grid, device="cpu")

# Push box right (ACTION4)
next_state, energy = sim.step_batch(
    grid.unsqueeze(0), torch.tensor([4])
)

# Check if box landed on target
win = sim.sokoban_win_energy(next_state)
print(f"Solved: {win[0].item() == 0.0}")
```

## Tests

```bash
python -m pytest
```

23 tests covering:
- Basic grid movement and wall collision
- Goal reaching with energy minimization
- Door/switch mechanics (toggle, open, pass-through)
- Teleporter mechanics (blue/red pairs, USE action)
- Sokoban box-pushing (push, block, chain, win detection)
- Multi-channel encoding
- Batch simulation
- CellType enum validation

## ARC-AGI-3 Benchmark

The ARC-AGI-3 integration requires the official
[ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) harness
and an API key from the [ARC Prize site](https://three.arcprize.org/).

### Setup

1. Clone the ARC-AGI-3-Agents repo:

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents
```

2. Configure your API key:

```bash
cp .env.example .env
# Edit .env and set ARC_API_KEY
```

3. Install dependencies (using the EHRA venv):

```bash
C:\Users\Jaydan\EHRA\venv\Scripts\pip.exe install -e /path/to/ARC-AGI-3-Agents
```

### Run

```bash
python main.py --agent monarch_ai --game=ls20 --tags Monarch_AI
```

The `Monarch_AI` agent class is registered automatically via
`soma_mythos_ehra.integrations.arc_agi_3`.

### Benchmark Results

| Run    | Score | Levels | FPS  | Tags        |
|--------|-------|--------|------|-------------|
| v1     | 0/7   | 0      | 0.69 | baseline    |
| v2     | 0/7   | 0      | 0.78 | fixed remap |
| v3-ich | 0/7   | 0      | 0.90 | in_channels |
| v3-gw  | 0/7   | 0      | 0.85 | goal weight |
| v4     | 0/7   | 0      | 0.82 | goal fix    |
| v5     | 0/7   | 0      | 1.06 | sokoban-v1  |

Scorecard: https://three.arcprize.org/scorecards/49f7e795-9a44-4b47-a90b-7d63eb35736d

## License

MIT License. See [LICENSE](LICENSE) for details.
