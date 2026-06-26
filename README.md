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

The architecture converts raw sensory grids into high-dimensional geometric
latents, explores potential branches using massive sub-surface simulation,
and utilizes reflective meta-controllers to adapt execution paths dynamically.

## Components

### SOMA (State-Observation-Model Architecture)

SOMA acts as the sub-symbolic "visual cortex" and physics simulator. It maps
2D spatial layouts directly into structural latent variables without
pixel-reconstruction.

**JEPA Vector Spaces (Joint Embedding Predictive Architecture):** An Encoder
maps a grid into a compact latent space, while a Predictor network forecasts
the downstream latent space (z_{t+1}) given an abstract action token (a_t).
This completely bypasses generative pixel-reconstruction, saving capacity on
irrelevant details.

**Energy-Based Evaluation (EBM):** The JEPA network outputs a scalar Energy
value (E). Correct, mathematically sound, or closer-to-goal states yield
minimal energy (E -> 0), whereas invalid configurations or spatial collisions
yield high energy.

**VRAM Tensor Simulator (`gpu_simulator.py`):** Physical interaction rules
are vectorized into PyTorch. Features like `wall_mask` and coordinate maps
are checked as concurrent tensor arrays on GPU, allowing thousands of forward
simulations to run concurrently.

### Mythos / Omnithos (Metacognitive Search Layer)

Mythos provides strategic logical reasoning and orchestrates non-linear
lookahead tree generation.

**Paced MCTS Solver:** An advanced variant of Monte Carlo Tree Search
constructs macro-action lookahead paths up to the configured horizon. The
search queries the SOMA simulator to evaluate branches in parallel batches.

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

## Core Algorithmic Coordination

1. **State Ingestion:** EHRA initializes a raw observation layout, spinning
   up dedicated worker threads for parallel puzzles.
2. **GPU Boundary Mapping:** SOMA reads the task rules, isolates structural
   features, and uploads an immutable `wall_mask` array directly to GPU
   memory.
3. **Tree Search Generation:** The Mythos layer initiates a paced lookahead
   tree. It groups hundreds of macro-moves into a parallel batch and throws
   them across the PCIe bus to SOMA's VRAM engine.
4. **Energy Feedback Routing:** The JEPA evaluator calculates energy profiles
   across the generated latent projection branches. These gradients feed
   back into the Mythos MCTS tracker to refine path weights.
5. **Action Dispatch:** The meta-controller selects the absolute
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

## Tests

```bash
python -m pytest
```

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

## License

MIT License. See [LICENSE](LICENSE) for details.
