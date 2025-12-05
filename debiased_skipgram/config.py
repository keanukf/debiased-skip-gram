from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    # Corpus
    corpus: Literal["text8", "wikitext103"] = "text8"
    min_count: int = 5
    
    # Model
    embedding_dim: int = 100
    window_size: int = 5
    
    # Training
    negative_samples: int = 5
    epochs: int = 5
    batch_size: int = 16384
    learning_rate: float = 0.025
    lr_decay: bool = True
    
    # Debiasing
    negative_sampling: Literal["frequency", "uniform"] = "frequency"
    alpha: float = 0.0  # 0.0 = standard SGNS, >0 = debiased
    
    # System
    device: str = "mps"  # Use "mps" for M1 Mac, "cuda" for NVIDIA, "cpu" otherwise
    num_workers: int = 4
    seed: int = 42

