"""SOMA Creative Engine — System 1: Fluid Creative Synthesis.

Generates multi-layered conceptual outlines and deep prose using the
scratch-built 5.4M LRLM. Forces structured reasoning before generation
to avoid generic, predictable phrasing.

Architecture: Topic → Concept Map → Outline → Prose Expansion
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


class SOMACreativeEngine:
    """
    SOMA Creative Layer.
    
    Forces the scratch-built 5.4M LRLM to generate multi-layered conceptual
    outliners for essays, philosophy, and breakthroughs. Uses a two-phase
    approach: conceptual skeleton generation, then prose expansion.
    """

    def __init__(self, lrlm, tokenizer):
        """
        Initialize Creative Engine.
        
        Args:
            lrlm: ARCDomainLLM instance
            tokenizer: SharedTokenizer instance
        """
        self.lrlm = lrlm
        self.tokenizer = tokenizer
        self.device = next(lrlm.parameters()).device
        
        # Creative writing prompts
        self.outline_prompt = "[BOS] concept_map topic : {topic} . core_theses : [ "
        self.essay_prompt = " write_essay theme : {outline} . text : "

    def generate_astonishing_essay(self, topic_str: str, 
                                  target_length: int = 1000,
                                  temperature: float = 0.8,
                                  top_k: int = 50) -> str:
        """
        Generate a structured essay on the given topic.
        
        Args:
            topic_str: Essay topic
            target_length: Target word count
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated essay text
        """
        print(f"\nSOMA Core Initiating Deep Ideation on: '{topic_str}'")
        
        # Phase 1: Generate conceptual skeleton
        outline = self._generate_outline(topic_str, temperature, top_k)
        print(f"Conceptual Matrix Formulated: {outline[:100]}...")
        
        # Phase 2: Expand skeleton into prose
        essay = self._expand_to_prose(outline, target_length, temperature, top_k)
        
        return essay

    def _generate_outline(self, topic: str, temperature: float, 
                         top_k: int) -> str:
        """Generate structured conceptual outline."""
        prompt = self.outline_prompt.format(topic=topic)
        context_tokens = self.tokenizer.encode(prompt)
        if context_tokens.dim() == 1:
            context_tokens = context_tokens.unsqueeze(0)
        context_tokens = context_tokens.to(self.device)
        
        # Generate skeleton (stop at "]")
        skeleton_tokens = self._autoregressive_gen(
            context_tokens, 
            max_len=128, 
            stop_token="]",
            temperature=temperature,
            top_k=top_k,
        )
        
        return self.tokenizer.decode(skeleton_tokens)

    def _expand_to_prose(self, outline: str, target_length: int,
                        temperature: float, top_k: int) -> str:
        """Expand conceptual outline into full prose."""
        prompt = self.essay_prompt.format(outline=outline)
        context_tokens = self.tokenizer.encode(prompt)
        if context_tokens.dim() == 1:
            context_tokens = context_tokens.unsqueeze(0)
        context_tokens = context_tokens.to(self.device)
        
        # Generate prose
        essay_tokens = self._autoregressive_gen(
            context_tokens,
            max_len=target_length // 4,  # Approx tokens per word
            stop_token="[EOS]",
            temperature=temperature,
            top_k=top_k,
        )
        
        return self.tokenizer.decode(essay_tokens)

    def _autoregressive_gen(self, context: torch.Tensor, max_len: int,
                           stop_token: str, temperature: float = 0.8,
                           top_k: int = 50) -> list:
        """
        Autoregressive token generation with top-k sampling.
        
        Args:
            context: (1, seq_len) input tokens
            max_len: Maximum generation length
            stop_token: Token to stop at
            temperature: Sampling temperature
            top_k: Top-k filtering
            
        Returns:
            list of generated token IDs
        """
        generated = []
        stop_token_id = self.tokenizer.token_to_id.get(stop_token, -1) if hasattr(self.tokenizer, 'token_to_id') else -1
        
        # Dummy inputs for grid/diff (not used in creative mode)
        dummy_grid = torch.zeros(1, 64, dtype=torch.long).to(self.device)
        dummy_diff = torch.zeros(1, 256).to(self.device)
        
        self.lrlm.eval()
        
        for _ in range(max_len):
            with torch.no_grad():
                # Forward pass
                logits = self.lrlm(context)  # (1, seq_len, vocab_size)
                next_token_logits = logits[:, -1, :]  # (1, vocab_size)
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
            # Check stop condition
            if next_token == stop_token_id:
                break
            
            generated.append(next_token)
            
            # Append to context
            next_tensor = torch.tensor([[next_token]], dtype=torch.long).to(self.device)
            context = torch.cat([context, next_tensor], dim=1)
        
        return generated

    def generate_concept_map(self, topic: str, num_concepts: int = 5) -> dict:
        """
        Generate a structured concept map for a topic.
        
        Args:
            topic: Main topic
            num_concepts: Number of sub-concepts to generate
            
        Returns:
            dict with concept hierarchy
        """
        prompt = f"[BOS] concept_map topic : {topic} . num_concepts : {num_concepts} . concepts : [ "
        context_tokens = self.tokenizer.encode(prompt)
        if context_tokens.dim() == 1:
            context_tokens = context_tokens.unsqueeze(0)
        context_tokens = context_tokens.to(self.device)
        
        # Generate concept list
        concept_tokens = self._autoregressive_gen(
            context_tokens,
            max_len=256,
            stop_token="]",
            temperature=0.7,
            top_k=30,
        )
        
        concept_text = self.tokenizer.decode(concept_tokens)
        
        # Parse concepts (simple split)
        concepts = [c.strip() for c in concept_text.split(",") if c.strip()]
        
        return {
            "topic": topic,
            "concepts": concepts[:num_concepts],
            "raw": concept_text,
        }

    def generate_research_question(self, topic: str) -> str:
        """Generate a novel research question for a topic."""
        prompt = f"[BOS] research_question topic : {topic} . question : "
        context_tokens = self.tokenizer.encode(prompt)
        if context_tokens.dim() == 1:
            context_tokens = context_tokens.unsqueeze(0)
        context_tokens = context_tokens.to(self.device)
        
        question_tokens = self._autoregressive_gen(
            context_tokens,
            max_len=64,
            stop_token="?",
            temperature=0.9,
            top_k=40,
        )
        
        return self.tokenizer.decode(question_tokens) + "?"

    def generate_hypothesis(self, observation: str) -> str:
        """Generate a scientific hypothesis from an observation."""
        prompt = f"[BOS] hypothesis observation : {observation} . hypothesis : "
        context_tokens = self.tokenizer.encode(prompt)
        if context_tokens.dim() == 1:
            context_tokens = context_tokens.unsqueeze(0)
        context_tokens = context_tokens.to(self.device)
        
        hypothesis_tokens = self._autoregressive_gen(
            context_tokens,
            max_len=128,
            stop_token="[EOS]",
            temperature=0.8,
            top_k=50,
        )
        
        return self.tokenizer.decode(hypothesis_tokens)
