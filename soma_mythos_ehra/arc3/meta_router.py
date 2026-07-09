"""Meta-Cognitive Router — Neuro-Symbolic Gatekeeper Network.

Analyzes the linguistic and structural properties of a prompt by measuring
token-level entropy and hidden state patterns from the LRLM's transformer
layers. Routes to System 1 (Creative) or System 2 (Symbolic) without any
keyword matching.

Architecture:
  1. Forward pass through LRLM transformer layers
  2. Capture hidden state from final layer
  3. Compute token distribution entropy
  4. Lightweight classifier head → routing score [0,1]
  5. Score < 0.5 → Creative (System 1), Score >= 0.5 → Symbolic (System 2)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HiddenStateExtractor:
    """Hooks into LRLM transformer to extract hidden states."""
    
    def __init__(self, lrlm):
        self.lrlm = lrlm
        self.hidden_states = []
        self._hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on transformer blocks."""
        for block in self.lrlm.transformer.h:
            handle = block.register_forward_hook(self._hook_fn)
            self._hook_handles.append(handle)
    
    def _hook_fn(self, module, input, output):
        """Capture output hidden states from each block."""
        self.hidden_states.append(output)
    
    def clear(self):
        """Clear captured hidden states."""
        self.hidden_states = []
    
    def get_last_hidden(self) -> torch.Tensor:
        """Get the final hidden state from the last transformer block."""
        if not self.hidden_states:
            return None
        return self.hidden_states[-1]
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []


class MetaCognitiveRouter(nn.Module):
    """
    Neuro-Symbolic Gatekeeper Network.
    
    Analyzes prompts through the LRLM's own transformer layers to determine
    whether the input requires fluid creative synthesis (System 1) or strict
    symbolic verification (System 2).
    
    No keyword matching. Pure entropy-based meta-cognition.
    """
    
    def __init__(self, lrlm, d_model: int = 512, vocab_size: int = 8192):
        super().__init__()
        self.lrlm = lrlm
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Hidden state extractor
        self.extractor = HiddenStateExtractor(lrlm)
        
        # Entropy features → routing decision
        # Takes entropy statistics from the LRLM's output distribution
        # and hidden state patterns to decide routing
        self.entropy_classifier = nn.Sequential(
            nn.Linear(8, 64),    # 8 entropy-based features
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Hidden state classifier
        # Takes the final hidden state representation
        self.hidden_classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Ensemble weight (learned)
        self.entropy_weight = nn.Parameter(torch.tensor(0.5))
        self.hidden_weight = nn.Parameter(torch.tensor(0.5))
    
    def _compute_entropy_features(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy-based features from LRLM output distribution.
        
        Args:
            logits: (batch, seq_len, vocab_size) output logits
            
        Returns:
            (batch, 8) feature vector
        """
        # Get probabilities for the last position
        probs = F.softmax(logits[:, -1, :], dim=-1)  # (batch, vocab_size)
        
        # 1. Shannon entropy
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch,)
        
        # 2. Max probability (confidence)
        max_prob = torch.max(probs, dim=-1).values  # (batch,)
        
        # 3. Number of tokens with probability > 1% (effective vocabulary)
        effective_vocab = (probs > 0.01).float().sum(dim=-1)  # (batch,)
        
        # 4. Top-5 token concentration
        top5_probs = torch.topk(probs, min(5, probs.size(-1)), dim=-1).values
        top5_concentration = top5_probs.sum(dim=-1)  # (batch,)
        
        # 5. Gini impurity (measure of distribution uniformity)
        gini = 1.0 - torch.sum(probs ** 2, dim=-1)  # (batch,)
        
        # 6. Perplexity proxy (exp of entropy)
        perplexity = torch.exp(entropy)  # (batch,)
        perplexity_normalized = torch.log(perplexity + 1) / 9.0  # Normalize
        
        # 7. Token distribution skewness (approximation)
        mean_prob = probs.mean(dim=-1)
        skew = torch.mean((probs - mean_prob.unsqueeze(-1)) ** 3, dim=-1) / (torch.std(probs, dim=-1) ** 3 + 1e-9)
        skew = torch.clamp(skew, -10, 10)  # Clip extremes
        
        # 8. Low-entropy token ratio (tokens with prob > 10%)
        high_conf_ratio = (probs > 0.1).float().sum(dim=-1) / probs.size(-1)
        
        # Stack features
        features = torch.stack([
            entropy / 9.0,           # Normalized entropy
            max_prob,                 # Confidence
            effective_vocab / probs.size(-1),  # Vocabulary usage
            top5_concentration,       # Top-5 concentration
            gini,                     # Impurity
            perplexity_normalized,    # Perplexity
            torch.tanh(skew),         # Skewness
            high_conf_ratio,          # High confidence ratio
        ], dim=-1)  # (batch, 8)
        
        return features
    
    def _compute_hidden_features(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Compute features from LRLM hidden state.
        
        Args:
            hidden_state: (batch, seq_len, d_model) last transformer block output
            
        Returns:
            (batch, d_model) processed hidden features
        """
        # Use the last token's hidden state
        last_hidden = hidden_state[:, -1, :]  # (batch, d_model)
        
        # L2 normalize for stability
        last_hidden = F.normalize(last_hidden, p=2, dim=-1)
        
        return last_hidden
    
    def forward(self, input_ids: torch.Tensor) -> dict:
        """
        Analyze input and determine routing.
        
        Args:
            input_ids: (batch, seq_len) token indices
            
        Returns:
            dict with:
                - routing_score: (batch,) in [0,1], 0=creative, 1=symbolic
                - entropy_score: (batch,) from entropy features
                - hidden_score: (batch,) from hidden state
                - routing_decision: str "CREATIVE" or "SYMBOLIC"
                - confidence: (batch,) routing confidence
        """
        self.lrlm.eval()
        self.extractor.clear()
        
        # Forward through LRLM to get logits and capture hidden states
        with torch.no_grad():
            logits, _ = self.lrlm(input_ids)
        
        # Get last hidden state
        hidden_state = self.extractor.get_last_hidden()
        
        # Compute entropy features
        entropy_features = self._compute_entropy_features(logits)
        entropy_score = self.entropy_classifier(entropy_features).squeeze(-1)
        
        # Compute hidden features
        if hidden_state is not None:
            hidden_features = self._compute_hidden_features(hidden_state)
            hidden_score = self.hidden_classifier(hidden_features).squeeze(-1)
        else:
            hidden_score = entropy_score  # Fallback
        
        # Ensemble routing score
        w_entropy = torch.sigmoid(self.entropy_weight)
        w_hidden = torch.sigmoid(self.hidden_weight)
        w_total = w_entropy + w_hidden + 1e-9
        
        routing_score = (w_entropy * entropy_score + w_hidden * hidden_score) / w_total
        
        # Confidence is inverse of agreement disagreement
        confidence = 1.0 - torch.abs(entropy_score - hidden_score)
        
        # Determine routing decision (per batch element)
        if routing_score.dim() > 0:
            score = routing_score.mean().item()
        else:
            score = routing_score.item()
        
        routing_decision = "SYMBOLIC" if score >= 0.5 else "CREATIVE"
        
        return {
            "routing_score": routing_score,
            "entropy_score": entropy_score,
            "hidden_score": hidden_score,
            "routing_decision": routing_decision,
            "confidence": confidence,
            "raw_entropy": entropy_features,
        }
    
    def determine_system_route(self, input_ids: torch.Tensor) -> float:
        """
        Simple interface: returns routing score.
        
        Args:
            input_ids: (batch, seq_len) token indices
            
        Returns:
            float in [0,1] where 0=creative, 1=symbolic
        """
        result = self.forward(input_ids)
        score = result["routing_score"]
        if score.dim() > 0:
            return score.mean().item()
        return score.item()
    
    def analyze_prompt(self, input_ids: torch.Tensor) -> dict:
        """
        Full analysis of prompt intent.
        
        Args:
            input_ids: (batch, seq_len) token indices
            
        Returns:
            dict with detailed analysis
        """
        result = self.forward(input_ids)
        
        return {
            "routing_score": result["routing_score"].item() if result["routing_score"].dim() == 0 else result["routing_score"].mean().item(),
            "decision": result["routing_decision"],
            "confidence": result["confidence"].item() if result["confidence"].dim() == 0 else result["confidence"].mean().item(),
            "entropy_component": result["entropy_score"].item() if result["entropy_score"].dim() == 0 else result["entropy_score"].mean().item(),
            "hidden_component": result["hidden_score"].item() if result["hidden_score"].dim() == 0 else result["hidden_score"].mean().item(),
        }
