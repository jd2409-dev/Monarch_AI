# SOMA-Mythos-EHRA — Active-Inference LRLM for ARC-AGI-3

A self-contained, dual-engine Large Reasoning and Language Model (LRLM) built
entirely on local GPU. Combines active-inference world modeling, MCTS-guided
reasoning, and behavioral-cloned action prediction to solve interactive
ARC-AGI-3 game environments — with zero external API dependencies.

## Architecture

```
                         THE UNIFIED DUAL-ENGINE LRLM

  [User Query] ──► [IntentClassifier] ──► Router ──┬──► System 1 (Creative)
                                                    │    Local Qwen3/Gemma via Ollama
                                                    │    Essays, brainstorming, logic
                                                    │
                                                    └──► System 2 (Symbolic)
                                                         Active Inference Core
                                                         MCTS-verified execution
                                                         Zero-hallucination
```

### Core Brain: SOMA (State-Observation-Model Architecture)

| Component | File | Purpose |
|-----------|------|---------|
| Grid Encoder | `active_world_model.py` | CNN encodes 64x64 grids (16 values) → 256-dim latent |
| Diff Encoder | `active_world_model.py` | Captures what changed between states |
| Grid Decoder | `active_world_model.py` | Predicts actual next grid pixels |
| Transition Predictor | `active_world_model.py` | Predicts next latent from (state, action, diff) |
| Reward Predictor | `active_world_model.py` | Predicts win probability per action |

### Search Layer: Mythos (Monte Carlo Tree Search)

| Component | File | Purpose |
|-----------|------|---------|
| Hypothesis Ensemble | `active_world_model.py` | 5 diverse world models for uncertainty estimation |
| InfoMax Explorer | `info_max_explorer.py` | Curiosity-driven action selection via ensemble disagreement |
| MCTS Reasoning | `lrlm_reasoning_engine.py` | Beam search with world model verification |
| Code Evolution | `code_evolver.py` | 27 executable heuristic hypotheses, evolved via local LLM |

### Execution Layer: EHRA (Environment-Harness Runtime Agent)

| Component | File | Purpose |
|-----------|------|---------|
| ARC-AGI-3 Connector | `agi3_connector.py` | Official SDK wrapper for live game environments |
| Interactive Agent | `interactive_agent.py` | Full active-inference loop (explore → train → evolve) |
| Experience Replay | `replay_buffer.py` | 100K-capacity prioritized buffer |
| Curriculum Manager | `curriculum_manager.py` | Multi-level progression with knowledge transfer |
| Efficiency Optimizer | `efficiency_optimizer.py` | Trajectory replay + RHAE action efficiency tracking |

### LRLM Layer (Large Reasoning and Language Model)

| Component | File | Params | Purpose |
|-----------|------|--------|---------|
| Action Model | `local_action_model.py` | 3.2M | Causal transformer for action prediction |
| LRLM Core | `lrlm_core.py` | 5.4M | Multi-modal fusion: grid + action + text |
| Game Tokenizer | `game_tokenizer.py` | — | 128-vocab tokenizer for trajectories |
| Unified Router | `unified_lrlm.py` | — | Dual-engine orchestration (System 1 + 2) |
| Architecture Chat | `architecture_chat.py` | — | Zero-hallucination conversational controller |

## Install

Requires Python >= 3.11, PyTorch >= 2.0, and CUDA-capable GPU.

```bash
git clone https://github.com/Jaydan/EHRA.git
cd EHRA
pip install -e ".[dev]"
```

## Quick Start

### 1. Run the Interactive Agent

```bash
# Play 3 ARC-AGI-3 games, 2 episodes each
python benchmark_interactive.py --max-games 3 --episodes 2
```

### 2. Train the Action Model

```bash
# Collect transitions from 3 games, train action model + LRLM
python train_action_model.py --games 3 --episodes 2 --epochs 10
```

This produces two checkpoints:
- `checkpoints/action_model.pt` (3.2M params)
- `checkpoints/lrlm.pt` (5.4M params)

### 3. Launch the LRLM Terminal

```bash
# Auto-routing between System 1 and System 2
python lrlm_terminal.py

# Force all queries to symbolic core (System 2)
python lrlm_terminal.py --system2

# Force all queries to creative engine (System 1)
python lrlm_terminal.py --system1
```

Example session:

```
==============================================================
  SOMA-MYTHOS-EHRA DUAL-ENGINE LRLM COGNITIVE CORE
==============================================================
  Device: cuda
  LRLM: 5,446,666 params on cuda

  System 1 (Foundation): Offline (qwen3-coder:30b)
  System 2 (Symbolic): Active (MCTS verified)
  Routing: 0 queries

  Commands: 'help', 'status', 'stats', 'exit'
--------------------------------------------------------------

[Operator] ──> What actions are available in the current game?

[LRLM Brain] ──> Buffer: 1318 transitions. Recent rewards: [0.0, 0.0, 0.0, 0.0, 0.0].

[Operator] ──> Write a short paragraph about how active inference works.

[Operator] ──> help

  Available commands:
    status    - System status
    stats     - Routing statistics
    help      - This message
    exit/quit - End session

  Queries are auto-routed:
    System 1: Creative writing, brainstorming, broad reasoning
    System 2: Grid operations, game mechanics, architecture queries
```

### 4. Deep Exploration of a Single Game

```bash
# Run 10 episodes on one game to discover winning patterns
python benchmark_focused.py --game ls20-9607627b --episodes 10
```

### 5. Full Benchmark

```bash
# Run 5 games, 3 episodes each
python benchmark_interactive.py --max-games 5 --episodes 3
```

## Training Results

| Model | Params | Loss Start | Loss Final | Epochs | Data |
|-------|--------|------------|------------|--------|------|
| Action Model | 3.2M | 3.01 | 1.76 | 10 | 1,318 transitions |
| LRLM | 5.4M | 0.64 | 0.00 | 5 | 1,318 transitions |
| World Model | — | 200+ | 8-14 | per-episode | online |

## Key Design Principles

### Zero-Hallucination via Latent Grounding

All System 2 outputs are anchored in real VRAM tensors. The LRLM cannot invent
game states because the first tokens injected into its transformer are always
the actual `grid_latent` and `action_logits` from the active world model.

### Test-Time Compute via MCTS

Instead of greedy token generation, the reasoning engine runs a beam search
over token candidates. Action tokens are cross-verified against the world
model's ensemble predictions. Paths that contradict physical transition rules
are pruned.

### Self-Contained Local Inference

Every component runs on local CUDA:
- **0 external API calls** required
- Sub-millisecond action model inference
- World model trains online from replay buffer
- Code evolution uses local LLM (841K params) + heuristic library

## Files

```
soma_mythos_ehra/arc3/
├── active_world_model.py      # World model v2 (stop-gradient, diff encoder)
├── info_max_explorer.py        # Curiosity-driven exploration
├── hypothesis_manager.py       # Bayesian belief tracking
├── replay_buffer.py            # 100K prioritized experience replay
├── world_model_trainer.py      # Online ensemble training
├── code_evolver.py             # 27 heuristic hypotheses + LLM evolution
├── curriculum_manager.py       # Multi-level progression
├── efficiency_optimizer.py     # Trajectory replay + RHAE tracking
├── agi3_connector.py           # ARC-AGI-3 SDK wrapper
├── interactive_agent.py        # Full active-inference agent v4
├── game_tokenizer.py           # 128-vocab trajectory tokenizer
├── local_action_model.py       # 3.2M causal transformer
├── lrlm_core.py                # 5.4M multi-modal LRLM
├── lrlm_reasoning_engine.py    # MCTS beam search verification
├── unified_lrlm.py             # Dual-engine router
├── architecture_chat.py        # Zero-hallucination chat controller
├── local_coder.py              # 841K local domain LLM
├── trajectory_tokenizer.py     # 162-vocab trajectory tokenizer
└── [existing DSL/solver files]

benchmarks/
├── benchmark_interactive.py    # Multi-game ARC-AGI-3 benchmark
├── benchmark_focused.py        # Deep single-game exploration
├── benchmark_dsl.py            # Static grid benchmark (1/30)
└── train_action_model.py       # Full training pipeline

checkpoints/
├── action_model.pt             # Trained action model (3.2M params)
├── lrlm.pt                     # Trained LRLM (5.4M params)
├── local_arc_llm.pt            # Local domain LLM (841K params)
└── [other training checkpoints]
```

## Tests

```bash
python -m pytest
```

23 tests covering:
- Grid movement, wall collision, Sokoban box-pushing
- Door/switch mechanics, teleporter mechanics
- Multi-channel encoding, batch simulation
- CellType enum validation

## Benchmark Results

### ARC-AGI-3 Interactive Track

| Run | Games | Won | Buffer | Action Loss | Notes |
|-----|-------|-----|--------|-------------|-------|
| v4 | 5 | 0/5 | 3,076 | 8-14 | Stop-gradient, heuristic code |
| v5-lrlm | 3 | 0/3 | 1,318 | 1.76 | Action model trained |

### Static Grid (ARC-AGI-1)

| Task | Result | Method |
|------|--------|--------|
| 08ed6ac7 | Solved | Template search |

## License

MIT License. See [LICENSE](LICENSE) for details.
