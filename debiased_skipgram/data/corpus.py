"""Corpus processing and vocabulary building."""

import re
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np
from pathlib import Path

from .download import download_text8, CACHE_DIR


class Corpus:
    """Corpus processor for Skip-gram training."""
    
    def __init__(self, path: str = None, min_count: int = 5):
        """
        Load corpus and build vocabulary.
        
        Args:
            path: Path to corpus file. If None, uses Text8.
            min_count: Minimum word frequency to include in vocabulary.
        """
        if path is None:
            path = str(download_text8())
        
        self.min_count = min_count
        self.word_counts = Counter()
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []
        self.vocab_size = 0
        self.subsampling_probs: Dict[int, float] = {}
        self.noise_distribution: np.ndarray = None
        self.total_words = 0
        
        # Load and process corpus
        self._load_corpus(path)
        self._build_vocabulary()
        self._compute_subsampling_probs()
        self._compute_noise_distribution()
    
    def _load_corpus(self, path: str):
        """Load corpus from file."""
        print(f"Loading corpus from {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize: split on whitespace and lowercase
        words = text.lower().split()
        self.word_counts = Counter(words)
        print(f"Total tokens: {len(words)}")
        print(f"Unique words: {len(self.word_counts)}")
    
    def _build_vocabulary(self):
        """Build vocabulary from word counts."""
        # Filter by min_count
        filtered_words = {
            word: count 
            for word, count in self.word_counts.items() 
            if count >= self.min_count
        }
        
        # Sort by frequency (descending)
        sorted_words = sorted(
            filtered_words.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Build mappings
        self.word2idx = {}
        self.idx2word = []
        
        # Add special tokens if needed (not used in standard word2vec, but good practice)
        # For now, just add regular words
        
        for word, count in sorted_words:
            idx = len(self.idx2word)
            self.word2idx[word] = idx
            self.idx2word.append(word)
        
        self.vocab_size = len(self.idx2word)
        print(f"Vocabulary size (min_count={self.min_count}): {self.vocab_size}")
        
        # Update word_counts to only include vocabulary words
        self.word_counts = {word: filtered_words[word] for word in self.idx2word}
    
    def _compute_subsampling_probs(self):
        """
        Compute subsampling probabilities for frequent words.
        
        P(keep) = sqrt(t/f) + t/f where t=1e-5
        """
        t = 1e-5
        total_count = sum(self.word_counts.values())
        
        self.subsampling_probs = {}
        for word in self.idx2word:
            f = self.word_counts[word] / total_count
            prob = np.sqrt(t / f) + (t / f)
            # Clamp to [0, 1]
            prob = min(1.0, max(0.0, prob))
            idx = self.word2idx[word]
            self.subsampling_probs[idx] = prob
    
    def _compute_noise_distribution(self):
        """
        Compute noise distribution for negative sampling.
        
        P_n(j) ∝ f(j)^0.75
        """
        # Compute unigram distribution raised to 0.75
        total_count = sum(self.word_counts.values())
        unigram_probs = np.array([
            (self.word_counts[word] / total_count) ** 0.75
            for word in self.idx2word
        ])
        
        # Normalize
        self.noise_distribution = unigram_probs / unigram_probs.sum()
    
    def get_word_pairs(self, window_size: int = 5) -> List[Tuple[int, int]]:
        """
        Generate (center, context) pairs with dynamic window.
        
        Uses subsampling for frequent words during pair generation.
        
        Args:
            window_size: Maximum window size (actual window is random 1 to window_size)
        
        Returns:
            List of (center_word_idx, context_word_idx) pairs
        """
        # Reload corpus to get word sequence
        # We need the actual sequence, not just counts
        corpus_path = str(download_text8())
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        words = text.lower().split()
        
        # Convert to indices, filtering OOV words and applying subsampling
        word_indices = []
        for word in words:
            if word in self.word2idx:
                idx = self.word2idx[word]
                # Apply subsampling
                if np.random.random() < self.subsampling_probs[idx]:
                    word_indices.append(idx)
        
        self.total_words = len(word_indices)
        print(f"Words after subsampling: {self.total_words}")
        
        # Generate pairs
        pairs = []
        for i, center_idx in enumerate(word_indices):
            # Random window size between 1 and window_size
            window = np.random.randint(1, window_size + 1)
            
            # Context words before
            start = max(0, i - window)
            for j in range(start, i):
                context_idx = word_indices[j]
                pairs.append((center_idx, context_idx))
            
            # Context words after
            end = min(len(word_indices), i + window + 1)
            for j in range(i + 1, end):
                context_idx = word_indices[j]
                pairs.append((center_idx, context_idx))
        
        print(f"Generated {len(pairs)} word pairs")
        return pairs
    
    def get_word_frequency(self, word: str) -> int:
        """Get frequency of a word in the corpus."""
        return self.word_counts.get(word, 0)
    
    def word_to_idx(self, word: str) -> int:
        """Convert word to index."""
        return self.word2idx.get(word.lower(), -1)
    
    def idx_to_word(self, idx: int) -> str:
        """Convert index to word."""
        if 0 <= idx < len(self.idx2word):
            return self.idx2word[idx]
        return None

