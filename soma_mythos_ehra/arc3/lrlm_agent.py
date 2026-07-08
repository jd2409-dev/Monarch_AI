"""LRLM Agent v2 — inductive reasoning drives action selection in ARC-AGI-3.

Replaces the 3-tier heuristic fallback with the Scientific Method Loop:
  1. LRLM generates a hypothesis (reasoning tokens)
  2. Hypothesis Engine verifies against world model ensemble
  3. If rejected → info-max exploration fallback
  4. If verified → execute action
  5. Scratchpad updated with validated/invalidated rules
  6. Scratchpad context fed back into next reasoning cycle

This is the System 2 scratchpad reasoning that separates
human-level ARC solvers from brute-force policy networks.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import numpy as np

from soma_mythos_ehra.arc3.agi3_connector import ARC3Connector, FrameObservation, EpisodeRecord
from soma_mythos_ehra.arc3.local_coder import ARCDomainLLM, ARCCoderConfig
from soma_mythos_ehra.arc3.active_world_model import HypothesisEnsemble
from soma_mythos_ehra.arc3.hypothesis_engine import (
    HypothesisEngine, Hypothesis, VerificationResult,
    encode_grid_for_lrlm, ACTION_TOKEN_MAP, ACTION_ID_FROM_TOKEN,
)
from soma_mythos_ehra.arc3.four_tier_dataset import (
    VOCAB_SIZE, ASCII_BASE, TOK_BOS, TOK_EOS, TOK_PAD,
    TOK_GRID_START, TOK_GRID_END, TOK_ACTION,
    TOK_PROBLEM, TOK_STEP, TOK_ANSWER, TOK_SEP,
    text_to_tokens, tokens_to_text,
)
from soma_mythos_ehra.arc3.replay_buffer import ExperienceReplayBuffer


@dataclass
class LRLMAgentConfig:
    max_steps: int = 200
    max_episodes: int = 3
    checkpoint_path: str = "checkpoints/lrlm_full/lrlm_best.pt"
    temperature: float = 0.5
    top_k: int = 10
    buffer_capacity: int = 100000
    ensemble_size: int = 5
    latent_dim: int = 256
    num_actions: int = 8
    verbose: bool = True
    device: str | None = None
    confidence_threshold: float = 0.15
    verification_threshold: float = 0.3
    exploration_rate: float = 0.3
    history_len: int = 10
    # World model training
    train_every_n_episodes: int = 1
    train_steps: int = 50
    scratchpad_save: str = "checkpoints/scratchpad.json"


@dataclass
class LRLMStats:
    game_id: str = ""
    total_steps: int = 0
    levels_completed: int = 0
    won: bool = False
    final_state: str = "NOT_FINISHED"
    episode_time: float = 0.0
    actions_taken: list[int] = field(default_factory=list)
    lrlm_actions: int = 0
    fallback_actions: int = 0
    hypotheses_generated: int = 0
    hypotheses_verified: int = 0
    hypotheses_rejected: int = 0
    train_loss: float = 0.0
    reasoning_traces: list[str] = field(default_factory=list)


class LRLMAgent:
    """Agent driven by inductive reasoning via the LRLM + Hypothesis Engine.

    Flow per step:
      1. Encode 64x64 grid -> compact tokens
      2. LRLM generates hypothesis + proposed action
      3. Hypothesis Engine verifies with world model
      4. If verified: execute
      5. If rejected: info-max exploration fallback
      6. Update scratchpad from actual outcome
      7. Feed scratchpad context into next reasoning cycle
    """

    def __init__(self, config: LRLMAgentConfig | None = None) -> None:
        self.config = config or LRLMAgentConfig()
        self.device = torch.device(
            self.config.device if self.config.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load LRLM
        model_config = ARCCoderConfig(
            vocab_size=VOCAB_SIZE,
            d_model=256,
            n_layer=4,
            n_head=8,
            max_seq_len=512,
            dropout=0.0,
        )
        self.lrlm = ARCDomainLLM(model_config).to(self.device)
        if os.path.exists(self.config.checkpoint_path):
            ckpt = torch.load(self.config.checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                self.lrlm.load_state_dict(ckpt["state_dict"])
            else:
                self.lrlm.load_state_dict(ckpt)
        self.lrlm.eval()
        self.param_count = sum(p.numel() for p in self.lrlm.parameters())

        # World model ensemble
        self.world_model = HypothesisEnsemble(
            num_models=self.config.ensemble_size,
            latent_dim=self.config.latent_dim,
            num_actions=self.config.num_actions,
        ).to(self.device)

        # Hypothesis engine
        self.hypothesis_engine = HypothesisEngine(
            lrlm=self.lrlm,
            world_model=self.world_model,
            device=self.device,
            confidence_threshold=self.config.confidence_threshold,
            verification_threshold=self.config.verification_threshold,
        )

        # Load scratchpad if exists
        if os.path.exists(self.config.scratchpad_save):
            self.hypothesis_engine.scratchpad.load(self.config.scratchpad_save)

        # Environment
        self.connector = ARC3Connector()
        self.buffer = ExperienceReplayBuffer(capacity=self.config.buffer_capacity)

        # Stats
        self.stats = LRLMStats()
        self.episode_count = 0
        self.all_stats: list[LRLMStats] = []

    def _world_model_fallback(
        self,
        grid: np.ndarray,
        available_actions: list[int],
    ) -> int:
        """World model ensemble selects action with highest predicted reward."""
        grid_tensor = torch.tensor(grid, dtype=torch.long, device=self.device).unsqueeze(0)
        latent = self.world_model.encode(grid_tensor)

        best_action = np.random.choice(available_actions)
        best_reward = -1.0

        with torch.no_grad():
            for a in available_actions:
                action_t = torch.tensor([a], dtype=torch.long, device=self.device)
                rew = self.world_model.predict_reward_ensemble(latent, action_t)
                mean_rew = rew.mean().item()
                # Bonus for high-uncertainty actions (exploration)
                var = rew.var().item()
                score = mean_rew + self.config.exploration_rate * var
                if score > best_reward:
                    best_reward = score
                    best_action = a

        return best_action

    def _random_exploration(self, available_actions: list[int]) -> int:
        """Random action for cold-start or high-uncertainty states."""
        return np.random.choice(available_actions)

    def play_episode(self, game_id: str) -> LRLMStats:
        """Play one episode using inductive reasoning."""
        self.stats = LRLMStats(game_id=game_id)
        start_time = time.time()

        obs = self.connector.make(game_id)
        prev_grid = self.connector.get_grid_tensor(obs).numpy()
        history: list[int] = []
        total_reward = 0.0

        if self.config.verbose:
            print(f"\n--- Playing {game_id} ---")
            print(f"  LRLM: {self.param_count / 1e6:.1f}M params | "
                  f"Buffer: {len(self.buffer)} | "
                  f"Scratchpad: {len(self.hypothesis_engine.scratchpad.hypotheses)} hypotheses")

        for step in range(self.config.max_steps):
            if obs.state in ("WIN", "GAME_OVER"):
                break

            available = obs.available_actions or list(range(1, 8))
            grid_tokens = encode_grid_for_lrlm(prev_grid)
            scratchpad_tokens = self.hypothesis_engine.scratchpad.get_validated_tokens()

            # --- Scientific Method Loop ---
            verification = None
            use_fallback = False

            # Only use LRLM hypothesis engine if scratchpad has some context
            # or if we're past the cold-start phase
            if step > 3 or len(self.hypothesis_engine.scratchpad.validated_rules) > 0:
                # 1. LRLM generates hypothesis
                hypothesis_text, proposed_action, lrlm_confidence = \
                    self.hypothesis_engine.generate_hypothesis(
                        grid_tokens, available, step, history, scratchpad_tokens,
                    )

                # 2. Verify with world model
                verification = self.hypothesis_engine.verify_hypothesis(
                    prev_grid, proposed_action, hypothesis_text, step,
                )

                self.stats.hypotheses_generated += 1

                if verification.is_verified:
                    action = verification.action
                    self.stats.hypotheses_verified += 1
                    trace = (f"HYPOTHESIS VERIFIED: {hypothesis_text[:80]}... "
                            f"-> ACTION{action} ({verification.reasoning_trace})")
                else:
                    # Hypothesis rejected → fallback
                    use_fallback = True
                    self.stats.hypotheses_rejected += 1
                    trace = (f"HYPOTHESIS REJECTED: {hypothesis_text[:60]}... "
                            f"({verification.reasoning_trace}) → fallback")
            else:
                use_fallback = True
                trace = "Cold-start: using world model exploration"

            # 3. Fallback: world model ensemble or random
            if use_fallback:
                if len(self.buffer) > 32:
                    action = self._world_model_fallback(prev_grid, available)
                    trace += f" [WM: ACTION{action}]"
                else:
                    action = self._random_exploration(available)
                    trace += f" [RANDOM: ACTION{action}]"

            # 4. Execute action
            x, y = None, None
            if action == 6:
                nonzero = np.argwhere(prev_grid > 0)
                if len(nonzero) > 0:
                    idx = np.random.randint(len(nonzero))
                    y, x = int(nonzero[idx][0]), int(nonzero[idx][1])
                else:
                    x, y = 32, 32

            next_obs = self.connector.step(action, x=x, y=y)
            next_grid = self.connector.get_grid_tensor(next_obs).numpy()
            reward = 1.0 if next_obs.state == "WIN" else 0.0
            total_reward += reward

            # 5. Update scratchpad from outcome
            if verification is not None:
                self.hypothesis_engine.update_from_outcome(
                    verification.hypothesis_text,
                    action,
                    reward,
                    next_grid,
                    prev_grid,
                    step,
                )

            # 6. Store transition
            self.buffer.add(
                prev_grid=torch.from_numpy(prev_grid),
                action=action,
                next_grid=torch.from_numpy(next_grid),
                reward=reward,
                done=next_obs.state in ("WIN", "GAME_OVER"),
                available_actions=available,
            )

            history.append(action)
            if len(history) > self.config.history_len:
                history = history[-self.config.history_len:]

            self.stats.total_steps += 1
            self.stats.actions_taken.append(action)
            self.stats.reasoning_traces.append(trace)

            if not use_fallback:
                self.stats.lrlm_actions += 1
            else:
                self.stats.fallback_actions += 1

            if self.config.verbose and step % 20 == 0:
                print(f"  Step {step:3d}: {next_obs.state:12s} act={action} "
                      f"buf={len(self.buffer)}")
                safe_trace = trace.encode("ascii", errors="replace").decode("ascii")
                print(f"         {safe_trace}")

            if next_obs.state == "WIN":
                if self.config.verbose:
                    print(f"  *** WIN at step {step}! ***")
                    print(self.hypothesis_engine.report())
                break

            prev_grid = next_grid
            obs = next_obs

        # Post-episode
        won = obs.state == "WIN"
        self.stats.won = won
        self.stats.final_state = obs.state
        self.stats.levels_completed = obs.levels_completed
        self.stats.episode_time = time.time() - start_time

        self.episode_count += 1

        # Save scratchpad
        self.hypothesis_engine.scratchpad.save(self.config.scratchpad_save)

        if self.config.verbose:
            lrlm_pct = self.stats.lrlm_actions / max(self.stats.total_steps, 1) * 100
            print(f"  Final: {self.stats.final_state} | steps={self.stats.total_steps} "
                  f"| time={self.stats.episode_time:.1f}s")
            print(f"  LRLM: {self.stats.lrlm_actions}/{self.stats.total_steps} "
                  f"({lrlm_pct:.0f}%) | Fallback: {self.stats.fallback_actions}")
            print(f"  Hypotheses: {self.stats.hypotheses_generated} generated, "
                  f"{self.stats.hypotheses_verified} verified, "
                  f"{self.stats.hypotheses_rejected} rejected")
            print(self.hypothesis_engine.report())

        return self.stats

    def play_game(self, game_id: str, episodes: int | None = None) -> list[LRLMStats]:
        max_eps = episodes or self.config.max_episodes
        stats_list = []
        for ep in range(max_eps):
            if self.config.verbose:
                print(f"\n=== Episode {ep+1}/{max_eps} for {game_id} ===")
            stats = self.play_episode(game_id)
            stats_list.append(stats)
        self.all_stats.extend(stats_list)
        return stats_list

    def play_all(self, max_games: int | None = None, episodes: int = 3) -> None:
        games = self.connector.available_games
        if max_games:
            games = games[:max_games]

        print("=" * 60)
        print("LRLM Agent v2 — Inductive Reasoning Benchmark")
        print(f"Model: {self.param_count / 1e6:.1f}M params | Device: {self.device}")
        print(f"Ensemble: {self.config.ensemble_size} world models")
        print(f"Verification threshold: {self.config.verification_threshold}")
        print(f"Games: {len(games)} | Episodes: {episodes}")
        print("=" * 60)

        total_won = 0
        total_lrlm = 0
        total_actions = 0
        total_hyp_gen = 0
        total_hyp_ver = 0
        total_hyp_rej = 0

        for i, game in enumerate(games):
            gid = game["game_id"]
            baselines = game["baseline_actions"]

            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(games)}] {gid}")
            print(f"  Tags: {game['tags']}, Baselines: {baselines}")

            stats_list = self.play_game(gid, episodes)
            final = stats_list[-1] if stats_list else None

            if final and final.won:
                total_won += 1

            if final:
                total_lrlm += final.lrlm_actions
                total_actions += final.total_steps
                total_hyp_gen += final.hypotheses_generated
                total_hyp_ver += final.hypotheses_verified
                total_hyp_rej += final.hypotheses_rejected

                status = "WON" if final.won else "LOST"
                rhae = ""
                if baselines and final.total_steps > 0:
                    level = min(final.levels_completed, len(baselines) - 1)
                    if level >= 0:
                        rhae = f"RHAE={((baselines[level] / final.total_steps) ** 2):.2f}"
                print(f"\n  {status} | steps={final.total_steps} | "
                      f"levels={final.levels_completed} | {rhae}")

        print(f"\n{'='*60}")
        print(f"RESULT: {total_won}/{len(games)} won ({100*total_won/max(len(games),1):.1f}%)")
        lrlm_pct = total_lrlm / max(total_actions, 1) * 100
        print(f"LRLM-driven: {total_lrlm}/{total_actions} ({lrlm_pct:.0f}%)")
        print(f"Hypotheses: {total_hyp_gen} gen, {total_hyp_ver} verified, {total_hyp_rej} rejected")
        verif_rate = total_hyp_ver / max(total_hyp_gen, 1) * 100
        print(f"Verification rate: {verif_rate:.1f}%")
        print(f"Buffer: {len(self.buffer)}")
        print(f"Scratchpad: {len(self.hypothesis_engine.scratchpad.validated_rules)} validated rules")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Stress Test
# ══════════════════════════════════════════════════════════════════════════════

def stress_test(agent: LRLMAgent) -> None:
    """Push the LRLM through complex multi-domain reasoning tasks."""
    print("\n" + "=" * 60)
    print("LRLM GENERATIVE STRESS TEST")
    print("=" * 60)

    tasks = [
        ("Tier 4 - Essay", "write an essay on the nature of intelligence and reasoning"),
        ("Tier 3 - Sorting", "problem : sort the array [ 7 , 3 , 9 , 1 , 5 ] step by step"),
        ("Tier 3 - Search", "problem : find 5 in sorted array [ 1 , 3 , 5 , 7 , 9 ]"),
        ("Tier 3 - Math", "problem : compute ( 12 + 8 ) * 3 step by step"),
        ("Tier 2 - Mirror", "apply mirror_h then grow_objects to a grid"),
        ("Tier 2 - Branch", "if grid_is_square then tessellate_2x2 else border_frame"),
        ("Tier 1 - Physics", "describe what action 3 does to a grid with objects"),
        ("Cross-domain", "given a grid with 5 objects and density 0.3 , select the best transformation"),
    ]

    for name, prompt in tasks:
        print(f"\n--- {name} ---")
        print(f"[Input] {prompt}")

        input_tokens = [TOK_BOS] + text_to_tokens(prompt)
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=agent.device)

        generated = []
        with torch.no_grad():
            for _ in range(200):
                crop = input_ids[:, -508:]
                logits, _ = agent.lrlm(crop)
                next_logits = logits[0, -1, :] / 0.7

                values, _ = torch.topk(next_logits, 50)
                if values[-1] > float("-inf"):
                    next_logits[next_logits < values[-1]] = float("-inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if next_token.item() in (TOK_EOS, TOK_PAD):
                    break

                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        output = tokens_to_text(generated)
        print(f"[Output] {output}")
        print(f"  ({len(generated)} tokens)")

    print(f"\n{'='*60}")
    print("Stress test complete.")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LRLM Agent v2 — Inductive Reasoning")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lrlm_full/lrlm_best.pt")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--verification", type=float, default=0.3)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stress-test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = LRLMAgentConfig(
        checkpoint_path=args.checkpoint,
        max_steps=args.max_steps,
        verification_threshold=args.verification,
        confidence_threshold=args.confidence,
        device=args.device,
        verbose=not args.quiet,
    )

    agent = LRLMAgent(config)

    if args.stress_test:
        stress_test(agent)
    else:
        agent.play_all(max_games=args.max_games, episodes=args.episodes)
