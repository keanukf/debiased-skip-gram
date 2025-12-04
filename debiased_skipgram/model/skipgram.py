"""Skip-gram model implementation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Literal


class SkipGram(nn.Module):
    """Skip-gram model with separate center and context embeddings."""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize Skip-gram model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Two embedding matrices: center (U) and context (V)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with uniform distribution [-0.5/dim, 0.5/dim]."""
        init_range = 0.5 / self.embedding_dim
        
        nn.init.uniform_(self.center_embeddings.weight, -init_range, init_range)
        nn.init.uniform_(self.context_embeddings.weight, -init_range, init_range)
    
    def forward(
        self, 
        center_ids: torch.Tensor, 
        context_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dot products between center and context embeddings.
        
        Args:
            center_ids: (batch_size,) tensor of center word indices
            context_ids: (batch_size, 1 + num_negatives) tensor of context word indices
                        First column is positive context, rest are negatives
        
        Returns:
            scores: (batch_size, 1 + num_negatives) tensor of dot products
        """
        # Get embeddings
        center = self.center_embeddings(center_ids)  # (batch, dim)
        context = self.context_embeddings(context_ids)  # (batch, 1+neg, dim)
        
        # Compute dot products: (batch, 1+neg, dim) @ (batch, dim, 1) -> (batch, 1+neg, 1)
        # Then squeeze to (batch, 1+neg)
        scores = torch.bmm(
            context, 
            center.unsqueeze(2)
        ).squeeze(2)  # (batch, 1+neg)
        
        # Clamp for numerical stability
        scores = torch.clamp(scores, -10.0, 10.0)
        
        return scores
    
    def get_embeddings(
        self, 
        combine: Literal["center", "context", "average", "concat"] = "center"
    ) -> np.ndarray:
        """
        Get final word embeddings.
        
        Args:
            combine: How to combine center and context embeddings
                - "center": Use only center embeddings (U)
                - "context": Use only context embeddings (V)
                - "average": Average of U and V
                - "concat": Concatenate U and V (doubles dimension)
        
        Returns:
            embeddings: (vocab_size, dim) or (vocab_size, 2*dim) numpy array
        """
        with torch.no_grad():
            center_emb = self.center_embeddings.weight.cpu().numpy()
            context_emb = self.context_embeddings.weight.cpu().numpy()
        
        if combine == "center":
            return center_emb
        elif combine == "context":
            return context_emb
        elif combine == "average":
            return (center_emb + context_emb) / 2.0
        elif combine == "concat":
            return np.concatenate([center_emb, context_emb], axis=1)
        else:
            raise ValueError(f"Unknown combine method: {combine}")

