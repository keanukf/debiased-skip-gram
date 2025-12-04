"""Negative sampling implementations."""

import torch
import numpy as np
from collections import Counter
from typing import Literal


class NegativeSampler:
    """Negative sampler for Skip-gram training."""
    
    def __init__(
        self,
        word_counts: Counter,
        vocab_size: int,
        mode: Literal["frequency", "uniform"] = "frequency",
        table_size: int = 10000000
    ):
        """
        Initialize negative sampling distribution.
        
        Args:
            word_counts: Counter mapping word indices to frequencies
            vocab_size: Size of vocabulary
            mode: "frequency" for P_n(j) ∝ f(j)^0.75, "uniform" for P_n(j) = 1/V
            table_size: Size of pre-computed sampling table (larger = more accurate)
        """
        self.vocab_size = vocab_size
        self.mode = mode
        self.table_size = table_size
        
        if mode == "frequency":
            # Build unigram table for frequency-based sampling
            # P_n(j) ∝ f(j)^0.75
            self._build_frequency_table(word_counts)
        else:
            # Uniform sampling: P_n(j) = 1/V
            self.uniform_probs = torch.ones(vocab_size) / vocab_size
    
    def _build_frequency_table(self, word_counts: Counter):
        """
        Build unigram table for efficient O(1) sampling.
        
        Similar to word2vec's implementation: create a large table where
        each word appears proportional to f(j)^0.75.
        """
        # Compute unigram distribution raised to 0.75
        total_count = sum(word_counts.values())
        unigram_probs = np.array([
            (word_counts.get(i, 0) / total_count) ** 0.75
            for i in range(self.vocab_size)
        ])
        
        # Normalize
        unigram_probs = unigram_probs / unigram_probs.sum()
        
        # Build sampling table
        # Each word appears in the table proportional to its probability
        table = []
        for word_idx in range(self.vocab_size):
            count = int(unigram_probs[word_idx] * self.table_size)
            table.extend([word_idx] * count)
        
        # Pad to exact table_size
        while len(table) < self.table_size:
            table.append(np.random.randint(0, self.vocab_size))
        
        self.sampling_table = np.array(table[:self.table_size], dtype=np.int64)
        self.table_idx = 0
    
    def sample(self, batch_size: int, num_negatives: int) -> torch.Tensor:
        """
        Sample negative examples.
        
        Args:
            batch_size: Number of samples to generate
            num_negatives: Number of negative samples per example
        
        Returns:
            negatives: (batch_size, num_negatives) tensor of negative word indices
        """
        if self.mode == "frequency":
            # Use pre-computed table for O(1) sampling
            total_samples = batch_size * num_negatives
            
            # Sample from table
            indices = np.random.randint(0, self.table_size, size=total_samples)
            negatives = self.sampling_table[indices]
            
            # Reshape to (batch_size, num_negatives)
            negatives = negatives.reshape(batch_size, num_negatives)
            
            return torch.from_numpy(negatives).long()
        
        else:  # uniform
            # Sample uniformly from vocabulary
            negatives = torch.multinomial(
                self.uniform_probs,
                num_samples=batch_size * num_negatives,
                replacement=True
            )
            
            # Reshape to (batch_size, num_negatives)
            negatives = negatives.reshape(batch_size, num_negatives)
            
            return negatives

