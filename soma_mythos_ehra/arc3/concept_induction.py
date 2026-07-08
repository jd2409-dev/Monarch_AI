"""Mythos Concept Induction — Dynamic Concept Invention Module.

Detects stalls in reasoning loops and proposes brand-new abstract tokens.
When the model hits a logical wall, this module creates a novel mathematical
concept, injects it into the vocabulary, and tests if it helps.

Architecture: Stall Detection → Concept Proposal → Token Injection → Validation
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass


@dataclass
class InductedConcept:
    """Represents an inducted concept."""
    token_id: int
    name: str
    description: str
    embedding_vector: torch.Tensor
    success_rate: float = 0.0
    usage_count: int = 0


class MythosConceptInductor(nn.Module):
    """
    Mythos Layer for Concept Invention.
    
    Detects stalls in reasoning loops and proposes brand-new abstract tokens.
    Dynamically registers novel symbolic entities to bypass logical walls.
    """

    def __init__(self, vocab_size: int = 8192, d_model: int = 512, 
                 max_new_concepts: int = 128):
        super().__init__()
        self.d_model = d_model
        self.max_new_concepts = max_new_concepts
        
        # Stall detection network
        self.stall_detector = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Concept proposal network
        self.concept_generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # Concept validity predictor
        self.validity_predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Registry of inducted concepts
        self.inducted_concepts: list[InductedConcept] = []
        self.next_concept_id = vocab_size  # Start after existing vocab
        
        # Embedding for new concepts (learned)
        self.concept_embeddings = nn.Embedding(max_new_concepts, d_model)

    def detect_stall(self, state_sequence: torch.Tensor, 
                    window_size: int = 5) -> float:
        """
        Detect if the reasoning process is stalled.
        
        Args:
            state_sequence: (seq_len, d_model) sequence of state embeddings
            window_size: Number of recent states to consider
            
        Returns:
            Stall probability (0-1)
        """
        if len(state_sequence) < window_size:
            return 0.0
        
        # Get recent states
        recent = state_sequence[-window_size:]
        
        # Compute state variance (low variance = stall)
        state_var = recent.var(dim=0).mean()
        
        # Compute state progression (small changes = stall)
        state_diffs = torch.diff(recent, dim=0)
        avg_diff = state_diffs.abs().mean()
        
        # Concatenate features
        features = torch.cat([
            recent[-1],  # Current state
            recent[-2] if len(recent) > 1 else recent[-1],  # Previous state
        ]).unsqueeze(0)
        
        # Predict stall
        with torch.no_grad():
            stall_prob = self.stall_detector(features).item()
        
        # Boost probability if variance is low
        if state_var < 0.01:
            stall_prob = min(1.0, stall_prob + 0.3)
        
        return stall_prob

    def propose_concept(self, current_state: torch.Tensor, 
                       context: str = "") -> Optional[InductedConcept]:
        """
        Propose a new concept when stalled.
        
        Args:
            current_state: (d_model,) current state embedding
            context: Optional context string
            
        Returns:
            InductedConcept or None if no concept proposed
        """
        if len(self.inducted_concepts) >= self.max_new_concepts:
            return None
        
        # Generate concept vector
        with torch.no_grad():
            concept_vector = self.concept_generator(current_state.unsqueeze(0))
            concept_vector = concept_vector.squeeze(0)
        
        # Predict validity
        validity = self.validity_predictor(concept_vector.unsqueeze(0)).item()
        
        if validity < 0.3:
            return None  # Low validity, don't create concept
        
        # Create concept
        concept_id = self.next_concept_id
        self.next_concept_id += 1
        
        # Generate name
        concept_name = f"CONCEPT_{len(self.inducted_concepts)}"
        
        # Store concept
        concept = InductedConcept(
            token_id=concept_id,
            name=concept_name,
            description=f"Inducted concept from context: {context[:50]}",
            embedding_vector=concept_vector,
        )
        
        self.inducted_concepts.append(concept)
        
        # Update concept embedding
        with torch.no_grad():
            self.concept_embeddings.weight[concept_id - self.concept_embeddings.weight.shape[0] + len(self.inducted_concepts) - 1] = concept_vector
        
        return concept

    def validate_concept(self, concept: InductedConcept, 
                        success: bool) -> None:
        """
        Update concept based on usage outcome.
        
        Args:
            concept: The concept that was used
            success: Whether the concept helped
        """
        concept.usage_count += 1
        
        # Update success rate with exponential moving average
        alpha = 0.1
        concept.success_rate = (1 - alpha) * concept.success_rate + alpha * (1.0 if success else 0.0)
        
        # Remove concept if success rate too low after enough uses
        if concept.usage_count > 10 and concept.success_rate < 0.2:
            self.inducted_concepts.remove(concept)

    def get_concept_embedding(self, token_id: int) -> Optional[torch.Tensor]:
        """Get embedding for an inducted concept."""
        for concept in self.inducted_concepts:
            if concept.token_id == token_id:
                return concept.embedding_vector
        return None

    def get_all_concepts(self) -> list[dict]:
        """Get all inducted concepts."""
        return [
            {
                "id": c.token_id,
                "name": c.name,
                "description": c.description,
                "success_rate": c.success_rate,
                "usage_count": c.usage_count,
            }
            for c in self.inducted_concepts
        ]
