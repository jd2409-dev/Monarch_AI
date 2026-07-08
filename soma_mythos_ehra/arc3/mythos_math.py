"""Mythos Math World Model — Transition Dynamics for Theorem Proving.

Predicts the resulting goal state if a specific mathematical tactic
is applied to the current proof state. Mirrors the ARC-AGI-3 world model
but operates on semantic latent spaces rather than pixel grids.

Architecture: (state_latent + tactic_embed) → MLP → predicted_next_latent
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Lean 4 tactic vocabulary (expandable)
LEAN_TACTICS = {
    "intro": 0,
    "apply": 1,
    "exact": 2,
    "rw": 3,        # rewrite
    "simp": 4,      # simplification
    "omega": 5,     # linear arithmetic
    "linarith": 6,  # linear arithmetic
    "ring": 7,      # ring normalization
    "norm_num": 8,  # numeric normalization
    "cases": 9,
    "induction": 10,
    "constructor": 11,
    "exists": 12,
    "have": 13,
    "let": 14,
    "calc": 15,
    "sorry": 16,    # placeholder (avoid in proofs)
    "assumption": 17,
    "trivial": 18,
    " rfl": 19,     # reflexivity
    "symm": 20,     # symmetry
    "trans": 21,    # transitivity
    "congr": 22,    # congruence
    "ext": 23,      # extensionality
    "fin_cases": 24,
    "decide": 25,
    "norm_cast": 26,
    "push_cast": 27,
    "field_simp": 28,
    "nlinarith": 29,
    "positivity": 30,
    "aesop": 31,
}

NUM_TACTICS = len(LEAN_TACTICS)


class MythosMathWorldModel(nn.Module):
    """
    Mythos (World Model) for Mathematics.
    
    Predicts the next logical latent state given a current state and a tactic.
    Uses stop-gradient on target encodings for stable training.
    
    Architecture:
        - Tactic embedding: tactic_id → d_model vector
        - State encoder: current_state → d_model vector
        - Transition predictor: (state, tactic) → next_state prediction
        - Reward predictor: (state, tactic) → success probability
    """

    def __init__(self, d_model: int = 512, num_tactics: int = NUM_TACTICS):
        super().__init__()
        self.d_model = d_model
        
        # Tactic embedding
        self.tactic_embedding = nn.Embedding(num_tactics, d_model)
        
        # Transition predictor: (state + tactic) → next_state
        self.transition_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Reward predictor: (state + tactic) → success probability
        self.reward_predictor = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
        # Goal state decoder: latent → text representation
        self.goal_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)

    def forward(self, state_latent: torch.Tensor, tactic_id: torch.Tensor) -> dict:
        """
        Predict next state given current state and tactic.
        
        Args:
            state_latent: (batch, d_model) current state embedding from SOMA
            tactic_id: (batch,) tactic index
            
        Returns:
            dict with keys:
                - next_latent: (batch, d_model) predicted next state
                - success_prob: (batch, 1) probability of tactic success
                - decoded_goal: (batch, d_model) decoded goal representation
        """
        # Embed tactic
        tactic_feat = self.tactic_embedding(tactic_id)  # (batch, d_model)
        
        # Concatenate state and tactic
        fused = torch.cat([state_latent, tactic_feat], dim=-1)  # (batch, d_model*2)
        
        # Predict next latent
        next_latent = self.transition_predictor(fused)  # (batch, d_model)
        next_latent = self.norm(next_latent + state_latent)  # Residual connection
        
        # Predict success probability
        success_prob = self.reward_predictor(fused)  # (batch, 1)
        
        # Decode goal representation
        decoded_goal = self.goal_decoder(next_latent)  # (batch, d_model)
        
        return {
            "next_latent": next_latent,
            "success_prob": success_prob,
            "decoded_goal": decoded_goal,
        }

    def predict_ensemble(self, state_latent: torch.Tensor, tactic_id: torch.Tensor, 
                        num_samples: int = 5) -> dict:
        """
        Monte Carlo ensemble prediction for uncertainty estimation.
        
        Args:
            state_latent: (batch, d_model) current state
            tactic_id: (batch,) tactic index
            num_samples: number of forward passes for uncertainty
            
        Returns:
            dict with ensemble statistics
        """
        predictions = []
        success_probs = []
        
        self.train()  # Enable dropout for MC sampling
        for _ in range(num_samples):
            with torch.no_grad():
                out = self.forward(state_latent, tactic_id)
                predictions.append(out["next_latent"])
                success_probs.append(out["success_prob"])
        
        self.eval()
        
        # Stack predictions
        pred_stack = torch.stack(predictions)  # (num_samples, batch, d_model)
        prob_stack = torch.stack(success_probs)  # (num_samples, batch, 1)
        
        return {
            "mean_latent": pred_stack.mean(dim=0),
            "std_latent": pred_stack.std(dim=0),
            "mean_success": prob_stack.mean(dim=0),
            "std_success": prob_stack.std(dim=0),
            "all_predictions": pred_stack,
        }


class MythosMathEnsemble(nn.Module):
    """
    Ensemble of MythosMathWorldModels for uncertainty estimation.
    Mirrors HypothesisEnsemble from ARC-AGI-3.
    """

    def __init__(self, num_models: int = 5, d_model: int = 512):
        super().__init__()
        self.models = nn.ModuleList([
            MythosMathWorldModel(d_model=d_model)
            for _ in range(num_models)
        ])

    def forward(self, state_latent: torch.Tensor, tactic_id: torch.Tensor) -> dict:
        """
        Get ensemble predictions with uncertainty estimates.
        """
        all_next_latents = []
        all_success_probs = []
        
        for model in self.models:
            out = model(state_latent, tactic_id)
            all_next_latents.append(out["next_latent"])
            all_success_probs.append(out["success_prob"])
        
        # Stack and compute statistics
        latent_stack = torch.stack(all_next_latents)
        prob_stack = torch.stack(all_success_probs)
        
        return {
            "mean_latent": latent_stack.mean(dim=0),
            "std_latent": latent_stack.std(dim=0),
            "mean_success": prob_stack.mean(dim=0),
            "std_success": prob_stack.std(dim=0),
            "consistency": 1.0 - prob_stack.std(dim=0).mean(),  # Low variance = high consistency
        }
