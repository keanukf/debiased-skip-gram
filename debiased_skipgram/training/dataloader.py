"""Data loaders for Skip-gram training."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


class SkipGramDataset(Dataset):
    """Dataset yielding (center_word, context_word) pairs."""
    
    def __init__(self, word_pairs: List[Tuple[int, int]]):
        """
        Initialize dataset.
        
        Args:
            word_pairs: List of (center_word_idx, context_word_idx) tuples
        """
        self.word_pairs = word_pairs
    
    def __len__(self):
        return len(self.word_pairs)
    
    def __getitem__(self, idx):
        return self.word_pairs[idx]


class SkipGramBatchCollator:
    """
    Collate function that adds negative samples to each batch.
    
    Takes list of (center, context) pairs and returns:
    - center_ids: (batch_size,)
    - all_context_ids: (batch_size, 1 + num_negatives) where first column is positive
    """
    
    def __init__(self, negative_sampler, num_negatives: int):
        """
        Initialize collator.
        
        Args:
            negative_sampler: NegativeSampler instance
            num_negatives: Number of negative samples per positive pair
        """
        self.negative_sampler = negative_sampler
        self.num_negatives = num_negatives
    
    def __call__(self, batch):
        """
        Collate batch and add negative samples.
        
        Args:
            batch: List of (center_idx, context_idx) tuples
        
        Returns:
            center_ids: (batch_size,) tensor
            all_context_ids: (batch_size, 1 + num_negatives) tensor
        """
        # Extract centers and positives
        centers = torch.tensor([pair[0] for pair in batch], dtype=torch.long)
        positives = torch.tensor([pair[1] for pair in batch], dtype=torch.long).unsqueeze(1)
        
        # Sample negatives
        negatives = self.negative_sampler.sample(len(batch), self.num_negatives)
        
        # Concatenate positives and negatives
        # Shape: (batch_size, 1 + num_negatives)
        all_contexts = torch.cat([positives, negatives], dim=1)
        
        return centers, all_contexts


def create_dataloader(
    word_pairs: List[Tuple[int, int]],
    negative_sampler,
    num_negatives: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for Skip-gram training.
    
    Args:
        word_pairs: List of (center, context) pairs
        negative_sampler: NegativeSampler instance
        num_negatives: Number of negative samples
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader instance
    """
    dataset = SkipGramDataset(word_pairs)
    collator = SkipGramBatchCollator(negative_sampler, num_negatives)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=False  # Set to True if using CUDA
    )
    
    return dataloader

