"""Hypothesis Verification Engine — inductive reasoning scratchpad for ARC-AGI-3.

Forces the LRLM to think before acting:
  1. Generate a text hypothesis about what will happen
  2. Execute a safe exploratory move
  3. Compare actual outcome vs predicted outcome
  4. Score hypothesis validity (Bayesian update)
  5. Add validated rules to the scratchpad
  6. Use scratchpad context for future decisions

This is the System 2 reasoning layer that separates human-level
ARC solvers from brute-force policy networks.
"""
from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.nn.functional as F
import numpy as np

from soma_mythos_ehra.arc3.local_coder import ARCDomainLLM, ARCCoderConfig
from soma_mythos_ehra.arc3.active_world_model import HypothesisEnsemble
from soma_mythos_ehra.arc3.four_tier_dataset import (
    VOCAB_SIZE, ASCII_BASE, TOK_BOS, TOK_EOS, TOK_PAD,
    TOK_GRID_START, TOK_GRID_END, TOK_ACTION, TOK_REWARD,
    TOK_REASON_START, TOK_REASON_END, TOK_PROBLEM, TOK_STEP,
    TOK_ANSWER, TOK_SEP, TOK_MATH_INT, TOK_MATH_IF, TOK_MATH_ELSE,
    text_to_tokens, tokens_to_text,
)


# ══════════════════════════════════════════════════════════════════════════════
# Token mappings
# ══════════════════════════════════════════════════════════════════════════════

GRID_VAL_BASE = 2000
ACTION_TOKEN_MAP = {i: TOK_ACTION + i for i in range(1, 8)}
ACTION_ID_FROM_TOKEN = {v: k for k, v in ACTION_TOKEN_MAP.items()}

# Reverse map: ACTION ID -> token for generation
ACTION_ID_TO_TOKEN = {i: TOK_ACTION + i for i in range(1, 8)}


# ══════════════════════════════════════════════════════════════════════════════
# Grid encoding for LRLM input
# ══════════════════════════════════════════════════════════════════════════════

def encode_grid_for_lrlm(grid: np.ndarray, max_tokens: int = 80) -> list[int]:
    """Encode a 64x64 grid into a compact token sequence.

    Samples an 8x8 subgrid (every 8th cell) plus structural stats.
    """
    H, W = grid.shape
    tokens = [TOK_GRID_START]

    # Sample 8x8 grid
    step_h = max(1, H // 8)
    step_w = max(1, W // 8)
    for r in range(0, min(H, 8 * step_h), step_h):
        for c in range(0, min(W, 8 * step_w), step_w):
            val = int(grid[r, c]) % 16
            tokens.append(GRID_VAL_BASE + val)

    tokens.append(TOK_GRID_END)

    # Structural stats
    nonzero = int(np.count_nonzero(grid))
    density = nonzero / grid.size if grid.size > 0 else 0

    tokens.append(TOK_MATH_INT)
    for ch in str(min(nonzero, 999)):
        tokens.append(ASCII_BASE + ord(ch))

    density_bucket = min(int(density * 10), 9)
    tokens.append(TOK_MATH_INT)
    tokens.append(ASCII_BASE + ord(str(density_bucket)))

    # Color histogram
    present_vals = sorted(set(int(v) for v in grid.flatten() if v > 0))
    tokens.append(TOK_SEP)
    for v in present_vals[:8]:
        tokens.append(GRID_VAL_BASE + v)
    tokens.append(TOK_SEP)

    return tokens[:max_tokens]


# ══════════════════════════════════════════════════════════════════════════════
# Hypothesis
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Hypothesis:
    """A single hypothesis about game rules."""
    rule_text: str               # "If X then Y"
    predicted_action: int        # ACTION1-7
    confidence: float            # 0.0-1.0
    times_tested: int = 0
    times_confirmed: int = 0
    avg_consistency: float = 0.0
    last_tested_step: int = -1
    is_validated: bool = False

    @property
    def success_rate(self) -> float:
        return self.times_confirmed / max(self.times_tested, 1)

    def update(self, was_consistent: bool, step: int) -> None:
        self.times_tested += 1
        if was_consistent:
            self.times_confirmed += 1
        self.avg_consistency = (
            self.avg_consistency * (self.times_tested - 1) + (1.0 if was_consistent else 0.0)
        ) / self.times_tested
        self.last_tested_step = step
        self.is_validated = self.times_tested >= 3 and self.success_rate >= 0.67


# ══════════════════════════════════════════════════════════════════════════════
# Hypothesis Scratchpad
# ══════════════════════════════════════════════════════════════════════════════

class Scratchpad:
    """Maintains an evolving rulebook of validated hypotheses.

    The scratchpad grows as the agent explores, accumulating knowledge
    about the game's physics. Validated rules are fed back into the LRLM
    as context for future decisions.
    """

    def __init__(self, max_rules: int = 20) -> None:
        self.max_rules = max_rules
        self.hypotheses: list[Hypothesis] = []
        self.validated_rules: list[str] = []
        self.step_log: list[dict] = []

    def add_hypothesis(self, hyp: Hypothesis) -> None:
        """Add a new hypothesis to the scratchpad."""
        # Check if similar hypothesis exists
        for existing in self.hypotheses:
            if existing.rule_text == hyp.rule_text:
                # Update existing instead of duplicating
                return
        self.hypotheses.append(hyp)
        # Keep only top-K by confidence
        self.hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        self.hypotheses = self.hypotheses[:self.max_rules]

    def record_validation(self, step: int, action: int, outcome: str) -> None:
        """Record a validation event."""
        self.step_log.append({
            "step": int(step),
            "action": int(action),
            "outcome": str(outcome),
        })

    def get_validated_context(self) -> str:
        """Get validated rules as text context for the LRLM."""
        if not self.validated_rules:
            return ""
        return " . ".join(self.validated_rules[-10:])  # Last 10 rules

    def get_validated_tokens(self) -> list[int]:
        """Get validated rules as LRLM tokens."""
        context = self.get_validated_context()
        if not context:
            return []
        return text_to_tokens(context)

    def promote_validated(self) -> None:
        """Promote validated hypotheses to permanent rules."""
        for hyp in self.hypotheses:
            if hyp.is_validated and hyp.rule_text not in self.validated_rules:
                self.validated_rules.append(hyp.rule_text)
        self.validated_rules = self.validated_rules[-15:]  # Keep last 15

    def report(self) -> str:
        lines = [f"Scratchpad: {len(self.hypotheses)} hypotheses, "
                 f"{len(self.validated_rules)} validated rules"]
        for i, h in enumerate(self.hypotheses[:5]):
            status = "VALIDATED" if h.is_validated else f"tested={h.times_tested}"
            lines.append(f"  {i+1}. [{status}] {h.rule_text} "
                        f"(conf={h.confidence:.2f}, rate={h.success_rate:.2f})")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        data = {
            "validated_rules": self.validated_rules,
            "hypotheses": [
                {"rule": str(h.rule_text), "action": int(h.predicted_action),
                 "conf": float(h.confidence), "tested": int(h.times_tested),
                 "confirmed": int(h.times_confirmed), "validated": bool(h.is_validated)}
                for h in self.hypotheses
            ],
            "step_log": self.step_log[-100:],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.validated_rules = data.get("validated_rules", [])
        self.step_log = data.get("step_log", [])
        self.hypotheses = []
        for h in data.get("hypotheses", []):
            hyp = Hypothesis(
                rule_text=h["rule"],
                predicted_action=h["action"],
                confidence=h["conf"],
                times_tested=h["tested"],
                times_confirmed=h["confirmed"],
            )
            hyp.is_validated = h.get("validated", False)
            self.hypotheses.append(hyp)


# ══════════════════════════════════════════════════════════════════════════════
# Hypothesis Verification Engine
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerificationResult:
    """Result of a hypothesis verification cycle."""
    action: int
    confidence: float
    hypothesis_text: str
    world_model_agreement: float  # How much world model agrees
    is_verified: bool
    reasoning_trace: str


class HypothesisEngine:
    """Core engine that forces the LRLM to reason before acting.

    The Scientific Method Loop:
      1. LRLM generates a hypothesis about what action to take and why
      2. World model ensemble evaluates the predicted outcome
      3. If world model disagrees, hypothesis is rejected
      4. If hypothesis survives, action is executed
      5. Actual outcome is compared to prediction
      6. Scratchpad is updated with validated/invalidated rules
    """

    def __init__(
        self,
        lrlm: ARCDomainLLM,
        world_model: HypothesisEnsemble,
        device: torch.device,
        confidence_threshold: float = 0.15,
        verification_threshold: float = 0.3,
    ) -> None:
        self.lrlm = lrlm
        self.world_model = world_model
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.verification_threshold = verification_threshold
        self.scratchpad = Scratchpad()
        self.total_hypotheses = 0
        self.verified_hypotheses = 0
        self.rejected_hypotheses = 0

    @torch.no_grad()
    def generate_hypothesis(
        self,
        grid_tokens: list[int],
        available_actions: list[int],
        step: int,
        history: list[int],
        scratchpad_tokens: list[int] | None = None,
    ) -> tuple[str, int, float]:
        """Force the LRLM to generate a reasoning hypothesis before acting.

        Returns:
            (hypothesis_text, action, confidence)
        """
        # Build reasoning prompt
        prompt = [TOK_BOS]

        # Add scratchpad context if available
        if scratchpad_tokens:
            prompt.extend(text_to_tokens("known rules : "))
            prompt.extend(scratchpad_tokens[:40])
            prompt.append(TOK_SEP)

        prompt.extend(grid_tokens)
        prompt.append(TOK_PROBLEM)

        problem_text = (
            f"step {step} . available actions : {available_actions} . "
            f"hypothesize what each action does and select the best one ."
        )
        prompt.extend(text_to_tokens(problem_text))
        prompt.append(TOK_STEP)

        if history:
            recent = history[-5:]
            hist_text = f"recent actions : {recent} ."
            prompt.extend(text_to_tokens(hist_text))

        prompt.extend(text_to_tokens(
            "analyze the grid structure , identify objects and patterns , "
            "then write your hypothesis about what will happen ."
        ))

        prompt.append(TOK_ANSWER)

        # Autoregressively generate reasoning tokens
        input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        reasoning_tokens = []

        for _ in range(100):  # Max 100 reasoning tokens
            crop = input_ids[:, -508:]
            logits, _ = self.lrlm(crop)
            next_logits = logits[0, -1, :] / 0.7

            # Top-k
            values, _ = torch.topk(next_logits, min(50, next_logits.size(-1)))
            if values[-1] > float("-inf"):
                next_logits[next_logits < values[-1]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() in (TOK_EOS, TOK_PAD):
                break

            reasoning_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # If we hit an action token, extract the action
            tid = next_token.item()
            if tid in ACTION_ID_FROM_TOKEN:
                action = ACTION_ID_FROM_TOKEN[tid]
                if action in available_actions:
                    break

        # Extract hypothesis text
        hypothesis_text = tokens_to_text(reasoning_tokens)

        # Extract action from generated tokens
        action = None
        for tid in reversed(reasoning_tokens):
            if tid in ACTION_ID_FROM_TOKEN:
                candidate = ACTION_ID_FROM_TOKEN[tid]
                if candidate in available_actions:
                    action = candidate
                    break

        if action is None:
            action = np.random.choice(available_actions)

        # Compute confidence from the final logits
        with torch.no_grad():
            crop = input_ids[:, -508:]
            logits, _ = self.lrlm(crop)
            action_mask = torch.full_like(logits[0, -1, :], float("-inf"))
            for a in available_actions:
                if a in ACTION_TOKEN_MAP:
                    action_mask[ACTION_TOKEN_MAP[a]] = 0.0
            masked = logits[0, -1, :] + action_mask
            probs = F.softmax(masked / 0.5, dim=-1)
            confidence = probs.max().item()

        self.total_hypotheses += 1
        return hypothesis_text, action, confidence

    @torch.no_grad()
    def verify_with_world_model(
        self,
        grid: np.ndarray,
        action: int,
    ) -> float:
        """Use world model ensemble to verify if an action is consistent.

        Returns:
            consistency_score: 0.0 (contradiction) to 1.0 (strong agreement)
        """
        grid_tensor = torch.tensor(grid, dtype=torch.long, device=self.device).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)

        try:
            # Get ensemble predictions
            latent = self.world_model.encode(grid_tensor)
            predictions = self.world_model.predict_next_ensemble(latent, action_tensor)

            # Ensemble agreement: low variance = high agreement
            variance = predictions.var(dim=0).mean().item()
            consistency = 1.0 / (1.0 + variance * 10)  # Sigmoid-like mapping

            # Also check reward prediction
            reward_preds = self.world_model.predict_reward_ensemble(latent, action_tensor)
            reward_mean = reward_preds.mean().item()
            reward_var = reward_preds.var().item()

            # High reward prediction + low variance = good action
            reward_score = reward_mean * (1.0 - reward_var)

            # Combined score
            return consistency * 0.6 + reward_score * 0.4

        except Exception:
            return 0.5  # Neutral when world model fails

    def verify_hypothesis(
        self,
        grid: np.ndarray,
        action: int,
        hypothesis_text: str,
        step: int,
    ) -> VerificationResult:
        """Full hypothesis verification cycle.

        1. World model evaluates the action
        2. Scratchpad cross-references with known rules
        3. Confidence threshold check
        """
        # World model verification
        wm_score = self.verify_with_world_model(grid, action)

        # Scratchpad cross-reference
        scratchpad_score = 0.5  # Default neutral
        if self.scratchpad.validated_rules:
            # Check if any validated rule contradicts this hypothesis
            for rule in self.scratchpad.validated_rules:
                # Simple keyword overlap check
                overlap = len(set(rule.lower().split()) & set(hypothesis_text.lower().split()))
                if overlap > 3:
                    scratchpad_score = 0.8  # Corroborated by known rules
                    break

        # Combined verification
        combined = wm_score * 0.7 + scratchpad_score * 0.3
        is_verified = combined >= self.verification_threshold

        if is_verified:
            self.verified_hypotheses += 1
        else:
            self.rejected_hypotheses += 1

        return VerificationResult(
            action=action,
            confidence=combined,
            hypothesis_text=hypothesis_text,
            world_model_agreement=wm_score,
            is_verified=is_verified,
            reasoning_trace=f"wm={wm_score:.3f} sp={scratchpad_score:.3f} "
                           f"combined={combined:.3f} {'VERIFIED' if is_verified else 'REJECTED'}",
        )

    def update_from_outcome(
        self,
        hypothesis_text: str,
        action: int,
        actual_reward: float,
        actual_grid: np.ndarray,
        prev_grid: np.ndarray,
        step: int,
    ) -> None:
        """Update scratchpad based on actual outcome.

        Compare predicted state to actual state and update hypothesis scores.
        """
        # Check if the grid actually changed as expected
        grid_changed = not np.array_equal(prev_grid, actual_grid)
        got_reward = actual_reward > 0

        # Find the matching hypothesis
        for hyp in self.scratchpad.hypotheses:
            if hyp.predicted_action == action:
                was_consistent = grid_changed or got_reward
                hyp.update(was_consistent, step)
                break
        else:
            # Create new hypothesis
            rule = hypothesis_text[:100] if hypothesis_text else f"action {action} causes change"
            hyp = Hypothesis(
                rule_text=rule,
                predicted_action=action,
                confidence=0.5,
            )
            hyp.update(grid_changed or got_reward, step)
            self.scratchpad.add_hypothesis(hyp)

        self.scratchpad.record_validation(step, action,
            "reward" if got_reward else ("changed" if grid_changed else "noop"))

        # Promote validated hypotheses
        self.scratchpad.promote_validated()

    def report(self) -> str:
        lines = [
            f"Hypothesis Engine:",
            f"  Total: {self.total_hypotheses} | "
            f"Verified: {self.verified_hypotheses} | "
            f"Rejected: {self.rejected_hypotheses}",
            f"  Verification rate: "
            f"{self.verified_hypotheses / max(self.total_hypotheses, 1) * 100:.1f}%",
            self.scratchpad.report(),
        ]
        return "\n".join(lines)
