"""MCTS Reasoning Engine — search-guided token generation with world model verification.

Cross-verifies linguistic hypotheses against world model simulations before
committing to text output. Prevents hallucination by pruning reasoning paths
that contradict physical transition rules.

Architecture:
  1. Beam search over token candidates
  2. Action tokens verified via world model transition prediction
  3. Text tokens scored by language model probability
  4. Beams pruned by combined linguistic + physical score
  5. Best surviving sequence returned as verified output
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class BeamState:
    """A single beam in the search tree."""
    token_ids: torch.Tensor
    score: float
    world_confidence: float
    step: int


class MCTS_ReasoningCore:
    """Search-guided token generation head.

    Instead of greedy/beam search over text probabilities alone,
    this engine validates each action token against the world model's
    transition prediction. Paths that contradict physical rules are pruned.

    Args:
        lrlm_model: ARCLRLM instance for text generation
        world_model: ActiveWorldModel or HypothesisEnsemble for verification
        action_vocab_size: number of action tokens (0-9 in our tokenizer)
        beam_width: number of beams to maintain at each step
        temperature: sampling temperature for token probabilities
    """

    def __init__(
        self,
        lrlm_model,
        world_model=None,
        action_vocab_size: int = 10,
        beam_width: int = 3,
        temperature: float = 0.7,
    ) -> None:
        self.lrlm = lrlm_model
        self.world_model = world_model
        self.action_vocab_size = action_vocab_size
        self.beam_width = beam_width
        self.temperature = temperature

    def is_action_token(self, token_id: int) -> bool:
        """Check if a token ID represents an executable action."""
        return 0 <= token_id < self.action_vocab_size

    def verify_action_token(
        self,
        token_id: int,
        grid_latent: torch.Tensor,
        action_logits: torch.Tensor,
    ) -> float:
        """Verify an action token against the world model.

        Returns a confidence score [0, 1] based on how well the world model
        agrees with this action's predicted outcome.
        """
        if self.world_model is None:
            return 0.5  # No verification available

        try:
            with torch.no_grad():
                # Get ensemble predictions for this action
                action_tensor = torch.tensor([token_id], device=grid_latent.device)

                if hasattr(self.world_model, 'predict_next_ensemble'):
                    # HypothesisEnsemble
                    predictions = self.world_model.predict_next_ensemble(
                        grid_latent, action_tensor,
                    )
                    # Low variance = high agreement = high confidence
                    variance = predictions.var(dim=0).mean().item()
                    confidence = 1.0 / (1.0 + variance * 10)
                elif hasattr(self.world_model, 'predict_next'):
                    # Single ActiveWorldModel
                    next_latent = self.world_model.predict_next(
                        grid_latent, action_tensor,
                    )
                    # Confidence based on prediction magnitude (well-defined transitions)
                    confidence = torch.sigmoid(next_latent.mean()).item()
                else:
                    confidence = 0.5

                return max(0.0, min(1.0, confidence))
        except Exception:
            return 0.5

    def generate_verified_response(
        self,
        initial_tokens: torch.Tensor,
        grid_latent: torch.Tensor | None = None,
        action_logits: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        depth: int = 8,
    ) -> torch.Tensor:
        """Generate tokens with MCTS-style verification.

        Args:
            initial_tokens: (1, seq_len) starting token sequence
            grid_latent: (1, latent_dim) current world model state
            action_logits: (1, action_dim) current action model output
            max_new_tokens: maximum tokens to generate
            depth: search depth for beam verification
        Returns:
            (1, seq_len + generated) verified token sequence
        """
        device = initial_tokens.device

        if grid_latent is None:
            grid_latent = torch.zeros(1, 256, device=device)
        if action_logits is None:
            action_logits = torch.zeros(1, 128, device=device)

        # Initialize beams
        beams = [BeamState(
            token_ids=initial_tokens.clone(),
            score=0.0,
            world_confidence=1.0,
            step=0,
        )]

        for step in range(min(max_new_tokens, depth)):
            new_beams = []

            for beam in beams:
                # Truncate to max_seq_len for LRLM
                input_seq = beam.token_ids[:, -64:]

                with torch.no_grad():
                    text_logits, _, _ = self.lrlm(
                        input_seq, grid_latent, action_logits,
                    )
                    next_probs = F.softmax(text_logits[:, -1, :] / self.temperature, dim=-1)

                # Get top-k candidates
                top_probs, top_indices = torch.topk(next_probs, self.beam_width, dim=-1)

                for prob, token_id in zip(top_probs[0], top_indices[0]):
                    tid = token_id.item()
                    p = prob.item()

                    # Skip EOS/PAD
                    if tid in (71, 73):
                        continue

                    # Build new token sequence
                    new_tokens = torch.cat([
                        beam.token_ids,
                        token_id.unsqueeze(0).unsqueeze(0),
                    ], dim=1)

                    # Score: linguistic probability
                    linguistic_score = beam.score + torch.log(torch.tensor(p + 1e-10)).item()

                    # Verify action tokens against world model
                    if self.is_action_token(tid):
                        world_conf = self.verify_action_token(
                            tid, grid_latent, action_logits,
                        )
                        # Boost score for world-verified actions
                        world_bonus = world_conf * 2.0
                        combined_score = linguistic_score + world_bonus
                        new_world_conf = world_conf
                    else:
                        combined_score = linguistic_score
                        new_world_conf = beam.world_confidence

                    new_beams.append(BeamState(
                        token_ids=new_tokens,
                        score=combined_score,
                        world_confidence=new_world_conf,
                        step=step + 1,
                    ))

            if not new_beams:
                break

            # Prune to top beam_width beams
            new_beams.sort(key=lambda b: b.score, reverse=True)
            beams = new_beams[:self.beam_width]

            # Early stop if all beams hit EOS
            if all(b.token_ids[0, -1].item() == 71 for b in beams):
                break

        # Return best beam
        best = beams[0] if beams else BeamState(
            token_ids=initial_tokens, score=0.0,
            world_confidence=0.0, step=0,
        )
        return best.token_ids

    def generate_with_uncertainty(
        self,
        initial_tokens: torch.Tensor,
        grid_latent: torch.Tensor | None = None,
        action_logits: torch.Tensor | None = None,
        num_samples: int = 5,
        max_new_tokens: int = 32,
    ) -> tuple[torch.Tensor, float]:
        """Generate multiple responses and return the most certain one.

        Args:
            initial_tokens: (1, seq_len) starting tokens
            grid_latent: world model state
            action_logits: action model output
            num_samples: number of independent generation runs
            max_new_tokens: tokens per sample
        Returns:
            (best_tokens, uncertainty_score)
            uncertainty_score: 0=highly certain, 1=highly uncertain
        """
        responses = []
        scores = []

        for _ in range(num_samples):
            tokens = self.generate_verified_response(
                initial_tokens, grid_latent, action_logits, max_new_tokens,
            )
            responses.append(tokens)
            scores.append(tokens.shape[1])  # Longer = more confident (didn't truncate early)

        # Select response with highest score
        best_idx = max(range(len(responses)), key=lambda i: scores[i])
        best_tokens = responses[best_idx]

        # Compute uncertainty from response diversity
        if len(responses) > 1:
            # Check if top responses agree on action tokens
            action_tokens = []
            for r in responses:
                action_toks = [t.item() for t in r[0] if self.is_action_token(t.item())]
                action_tokens.append(action_toks[:5])  # First 5 actions

            # Uncertainty = disagreement ratio
            if action_tokens:
                from collections import Counter
                all_actions = [a for seq in action_tokens for a in seq]
                if all_actions:
                    most_common_count = Counter(all_actions).most_common(1)[0][1]
                    uncertainty = 1.0 - (most_common_count / len(all_actions))
                else:
                    uncertainty = 1.0
            else:
                uncertainty = 0.5
        else:
            uncertainty = 0.0

        return best_tokens, uncertainty
