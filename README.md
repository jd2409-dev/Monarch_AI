# SOMA-Mythos-EHRA — Neuro-Symbolic Intelligence Platform

A self-contained, dual-engine Large Reasoning and Language Model (LRLM) built
entirely on local GPU from scratch. Combines active-inference world modeling,
MCTS-guided reasoning, Lean 4 formal verification, and meta-cognitive routing
to solve interactive ARC-AGI-3 game environments and mathematical theorem
proving — with zero external API dependencies and no local LLMs (no Gemma,
no Qwen3).

## Architecture

```
                         THE OMNISCIENT INTELLIGENCE CORE

                         ┌─────────────────────────┐
                         │ Raw User Prompt Input   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │     MetaCognitiveRouter             │
                    │     (Neuro-Symbolic Gatekeeper)     │
                    │     Entropy Analysis + Hidden State │
                    └───────────┬─────────────┬──────────┘
                                │             │
                     Score < 0.5│             │Score >= 0.5
                ┌───────────────┘             └───────────────┐
                ▼                                             ▼
   ┌────────────────────────┐               ┌────────────────────────┐
   │  SYSTEM 1: CREATIVE    │               │  SYSTEM 2: SYMBOLIC    │
   │  SOMA Creative Engine  │               │  SOMA Math Encoder     │
   │  Concept Induction     │               │  Mythos Math World Mdl │
   │  Essay Generation      │               │  EHRA Lean 4 Executor  │
   └────────────────────────┘               └────────────────────────┘
```

### Core Brain: SOMA (State-Observation-Model Architecture)

| Component | File | Purpose |
|-----------|------|---------|
| Grid Encoder | `active_world_model.py` | CNN encodes 64x64 grids (16 values) → 256-dim latent |
| Diff Encoder | `active_world_model.py` | Captures what changed between states |
| Grid Decoder | `active_world_model.py` | Predicts actual next grid pixels |
| Transition Predictor | `active_world_model.py` | Predicts next latent from (state, action, diff) |
| Reward Predictor | `active_world_model.py` | Predicts win probability per action |
| SOMA Math Encoder | `soma_math.py` | Lean 4 goal states → 512-dim latent vectors |
| SOMA Creative Engine | `soma_creative.py` | System 1 essay generation, concept maps |

### Search Layer: Mythos (Monte Carlo Tree Search)

| Component | File | Purpose |
|-----------|------|---------|
| Hypothesis Ensemble | `active_world_model.py` | 5 diverse world models for uncertainty estimation |
| InfoMax Explorer | `info_max_explorer.py` | Curiosity-driven action selection via ensemble disagreement |
| MCTS Reasoning | `lrlm_reasoning_engine.py` | Beam search with world model verification |
| Code Evolution | `code_evolver.py` | 27 executable heuristic hypotheses, evolved locally |
| Mythos Math World Model | `mythos_math.py` | Tactic success prediction + ensemble uncertainty |
| Concept Induction | `concept_induction.py` | Dynamic concept invention for reasoning breakthroughs |

### Execution Layer: EHRA (Environment-Harness Runtime Agent)

| Component | File | Purpose |
|-----------|------|---------|
| ARC-AGI-3 Connector | `agi3_connector.py` | Official SDK wrapper for live game environments |
| Interactive Agent | `interactive_agent.py` | Full active-inference loop (explore → train → evolve) |
| Experience Replay | `replay_buffer.py` | 100K-capacity prioritized buffer |
| EHRA Math Executor | `ehra_math.py` | Lean 4 subprocess + MCTS theorem proving |
| World Model Trainer | `world_model_trainer.py` | Online ensemble training with stop-gradient |

### Meta-Cognition Layer

| Component | File | Purpose |
|-----------|------|---------|
| MetaCognitiveRouter | `meta_router.py` | Entropy-based intent routing (no keywords) |
| Omniscient Shell | `lrlm_ultimate_shell.py` | Unified terminal with autonomous routing |

### LRLM Layer (Large Reasoning and Language Model)

| Component | File | Params | Purpose |
|-----------|------|--------|---------|
| ARCDomainLLM | `local_coder.py` | 5.4M | Scratch-built causal transformer |
| Full LRLM Trainer | `train_full_lrlm.py` | 5.4M | Four-tier interleaved training pipeline |
| Interleaved Loader | `interleaved_data_loader.py` | — | Balanced 25/25/25/25 batch mixing |
| Four-Tier Dataset | `four_tier_dataset.py` | — | 210K samples, 8192 shared vocab |

## Install

Requires Python >= 3.11, PyTorch >= 2.0, and CUDA-capable GPU.

```bash
git clone https://github.com/Jaydan/EHRA.git
cd EHRA
pip install -e ".[dev]"
```

### Lean 4 (Optional, for Theorem Proving)

```bash
# Install elan (Lean version manager)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
# Or on Windows:
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/leanprover/elan/master/elan-init.ps1' -OutFile 'elan-init.ps1'
powershell -ExecutionPolicy Bypass -File elan-init.ps1
```

## Quick Start

### 1. Train the LRLM from Scratch

```bash
# Train on 5K subset (quick validation)
python -c "from soma_mythos_ehra.arc3.train_full_lrlm import LRLMTrainer; LRLMTrainer().train()"

# Train on full 210K dataset
python -m soma_mythos_ehra.arc3.train_full_lrlm --epochs 50
```

### 2. Launch the Omniscient Intelligence Shell

```bash
# Meta-cognitive routing (no keywords, pure entropy analysis)
python -m soma_mythos_ehra.arc3.lrlm_ultimate_shell
```

Example session:

```
======================================================================
  SOMA-MYTHOS-EHRA OMNISCIENT INTELLIGENCE CORE
  Meta-Cognitive Routing | Zero Keywords | Active Entropy Analysis
======================================================================
  Device: cuda
  Parameters: 5,446,666
  Meta-Cognitive Router: initialized
----------------------------------------------------------------------

[Operator] ──> the emergence of conscious loops inside neural networks

  [Meta-Cognition] Entropy: 0.612 | Hidden: 0.489 | Score: 0.491 | Confidence: 0.995
  [Routing] System 1: Fluid Conceptual Generation Engine

  [System 1 Output]
  The emergence of conscious loops inside isolated neural networks...

[Operator] ──> theorem add_zero (n : Nat) : n + 0 = n

  [Meta-Cognition] Entropy: 0.531 | Hidden: 0.502 | Score: 0.503 | Confidence: 0.997
  [Routing] System 2: Formal Logic & Verification Engine

  [System 2] Attempting to prove: theorem add_zero (n : Nat) : n + 0 = n...
```

### 3. Run ARC-AGI-3 Benchmark

```bash
# Play 5 games, 3 episodes each
python benchmark_lrlm.py --max-games 5 --episodes 3
```

### 4. Calibrate World Model on Live Transitions

```bash
# Collect transitions, train ensemble, re-run benchmark
python calibrate_world_model.py --collect-games 5 --cal-epochs 50 --verification 0.7
```

### 5. Test the Mathematical Stack

```bash
# Test SOMA, Mythos, EHRA math components
python run_math_proof_loop.py
```

## Training Results

| Model | Params | Loss Start | Loss Final | PPL Final | Data |
|-------|--------|------------|------------|-----------|------|
| LRLM (5K subset) | 5.4M | 5.53 | 0.40 | 1.49 | 5K samples |
| LRLM (full) | 5.4M | — | — | — | 210K samples |
| World Model | — | 3.62 | 0.85 | — | 527 live transitions |

## Key Design Principles

### Zero External Dependencies

Every component runs on local CUDA:
- **0 external API calls** required
- **0 local LLMs** (no Gemma, no Qwen3) — only the scratch-built LRLM
- Sub-millisecond action model inference
- World model trains online from replay buffer

### Meta-Cognitive Routing (No Keywords)

The `MetaCognitiveRouter` analyzes prompts through the LRLM's own transformer
layers, measuring token distribution entropy and hidden state patterns:
- **Low entropy** (structured, logical) → System 2 (Formal Verification)
- **High entropy** (fluid, creative) → System 1 (Creative Synthesis)
- No string matching, no keyword filters

### Four-Tier Interleaved Training

The LRLM trains on interleaved data from four domains:
- **Tier 1** (25%): Core physics interactions
- **Tier 2** (25%): Synthetic procedural traces
- **Tier 3** (25%): Algorithmic logic chains
- **Tier 4** (25%): Structural text corpora

Shared 8192-token vocabulary across all tiers prevents catastrophic forgetting.

### Lean 4 Formal Verification

The EHRA Math Executor interfaces with Lean 4 via subprocess communication:
- Parse goal states and error messages from compiler output
- Execute tactics and verify proof steps
- Run in simulation mode when Lean 4 is not installed

## Files

```
soma_mythos_ehra/arc3/
├── active_world_model.py      # World model v2 (stop-gradient, diff encoder)
├── info_max_explorer.py       # Curiosity-driven exploration
├── hypothesis_manager.py      # Bayesian belief tracking
├── replay_buffer.py           # 100K prioritized experience replay
├── world_model_trainer.py     # Online ensemble training
├── code_evolver.py            # 27 heuristic hypotheses + LLM evolution
├── curriculum_manager.py      # Multi-level progression
├── efficiency_optimizer.py    # Trajectory replay + RHAE tracking
├── agi3_connector.py          # ARC-AGI-3 SDK wrapper
├── interactive_agent.py       # Full active-inference agent v4
├── lrlm_agent.py              # LRLM agent v2 (hypothesis-driven)
├── game_tokenizer.py          # 128-vocab trajectory tokenizer
├── local_coder.py             # 5.4M scratch-built ARCDomainLLM
├── train_full_lrlm.py         # Full training pipeline
├── four_tier_dataset.py       # 210K sample generator, 8192 vocab
├── interleaved_data_loader.py # Balanced batch mixing
├── lrlm_interactive_shell.py  # Live chat with LRLM
├── hypothesis_engine.py       # Scientific method loop
├── meta_router.py             # Neuro-symbolic gatekeeper
├── soma_math.py               # SOMA encoder for Lean 4
├── mythos_math.py             # Mythos world model for tactics
├── ehra_math.py               # EHRA executor for Lean 4
├── soma_creative.py           # Creative essay generation
├── concept_induction.py       # Dynamic concept invention
├── lrlm_ultimate_shell.py     # Omniscient intelligence terminal
└── [existing DSL/solver files]

checkpoints/
├── lrlm_full/                 # Trained LRLM checkpoints
│   ├── lrlm_best.pt
│   ├── lrlm_final.pt
│   └── metrics.json
├── calibrated_world_model.pt  # Calibrated ensemble
└── scratchpad.json            # Hypothesis scratchpad

data/
├── four_tier/                 # Training data
│   ├── train_tokens.pt        # 199.5K samples
│   ├── val_tokens.pt          # 10.5K samples
│   └── meta.json
└── training/                  # ARC-AGI static grids

formal_math_core/              # Lean 4 project (auto-generated)
├── lakefile.lean
├── Main.lean
└── TacticRunner.lean
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

## License

MIT License. See [LICENSE](LICENSE) for details.
