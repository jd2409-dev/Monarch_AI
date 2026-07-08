"""SOMA Math Encoder — Perception Layer for Lean 4 Theorem Proving.

Maps the SOMA (Perception) abstraction from ARC-AGI-3 pixel processing
to Lean 4 goal state encoding. Transforms raw Lean goal text into dense
semantic embeddings ready for the Mythos World Model.

Architecture: Token Embedding → GRU → Dense Projection → (1, 512) latent
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SOMAMathEncoder(nn.Module):
    """
    SOMA (Perception Layer) for Mathematics.
    
    Translates raw Lean goal states into dense semantic embeddings.
    Input: Raw Lean goal text (e.g., "⊢ n + 0 = n")
    Output: (1, d_model) latent vector ready for Mythos World Model
    """

    def __init__(self, vocab_size: int = 8192, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Token embedding with learned positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(512, d_model)  # Max seq len 512
        self.dropout = nn.Dropout(dropout)
        
        # GRU for sequence encoding (captures sequential dependencies)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True,
            num_layers=2,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=False,
        )
        
        # Final projection to ensure consistent output shape
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode Lean goal tokens into dense latent representation.
        
        Args:
            token_ids: (batch, seq_len) token indices from tokenizer
            
        Returns:
            (batch, d_model) latent vector
        """
        batch_size, seq_len = token_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens + positions
        token_emb = self.embedding(token_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        # GRU encoding
        gru_out, hidden = self.gru(x)  # hidden: (num_layers, batch, d_model)
        
        # Take the last hidden state from the top layer
        last_hidden = hidden[-1]  # (batch, d_model)
        
        # Project and normalize
        out = self.projection(last_hidden)
        out = self.norm(out)
        
        return out

    def encode_text(self, text: str, tokenizer) -> torch.Tensor:
        """
        Convenience method: encode raw text string.
        
        Args:
            text: Raw Lean goal text
            tokenizer: SharedTokenizer instance
            
        Returns:
            (1, d_model) latent vector
        """
        self.eval()
        with torch.no_grad():
            tokens = tokenizer.encode(text)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            tokens = tokens.to(next(self.parameters()).device)
            return self.forward(tokens)
