"""Word similarity evaluation."""

import csv
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path

from .metrics import spearman_correlation
from ..data.download import (
    download_simlex999,
    download_wordsim353,
    download_rarewords
)


def load_simlex999(path: str = None) -> List[Tuple[str, str, float]]:
    """
    Load SimLex-999 dataset.
    
    Args:
        path: Path to SimLex-999.txt. If None, downloads it.
    
    Returns:
        List of (word1, word2, human_score) tuples
    """
    if path is None:
        path = str(download_simlex999())
    
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            word1 = row['word1'].lower()
            word2 = row['word2'].lower()
            score = float(row['SimLex999'])
            pairs.append((word1, word2, score))
    
    return pairs


def load_wordsim353(path: str = None) -> List[Tuple[str, str, float]]:
    """
    Load WordSim-353 dataset.
    
    Args:
        path: Path to WordSim-353 combined.csv. If None, downloads it.
    
    Returns:
        List of (word1, word2, human_score) tuples
    """
    if path is None:
        path = str(download_wordsim353())
    
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Get the actual column names from the reader
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("Could not read column names from WordSim-353 file")
        
        # Find the score column (try different variations)
        score_col = None
        for col in fieldnames:
            if 'human' in col.lower() and ('mean' in col.lower() or 'score' in col.lower()):
                score_col = col
                break
        
        if score_col is None:
            raise ValueError(f"Could not find score column in WordSim-353. Available columns: {fieldnames}")
        
        # Find word columns (try different variations)
        word1_col = None
        word2_col = None
        for col in fieldnames:
            col_lower = col.lower()
            if 'word' in col_lower and ('1' in col_lower or 'one' in col_lower):
                word1_col = col
            elif 'word' in col_lower and ('2' in col_lower or 'two' in col_lower):
                word2_col = col
        
        if word1_col is None or word2_col is None:
            raise ValueError(f"Could not find word columns in WordSim-353. Available columns: {fieldnames}")
        
        for row in reader:
            word1 = row[word1_col].lower().strip()
            word2 = row[word2_col].lower().strip()
            score = float(row[score_col])
            pairs.append((word1, word2, score))
    
    return pairs


def load_rarewords(path: str = None) -> List[Tuple[str, str, float]]:
    """
    Load Stanford Rare Words dataset.
    
    Args:
        path: Path to rarewords.txt. If None, downloads it.
    
    Returns:
        List of (word1, word2, human_score) tuples
    """
    if path is None:
        path = str(download_rarewords())
    
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                word1 = parts[0].lower()
                word2 = parts[1].lower()
                try:
                    score = float(parts[2])
                    pairs.append((word1, word2, score))
                except ValueError:
                    continue
    
    return pairs


def evaluate_similarity(
    embeddings: np.ndarray,
    word2idx: Dict[str, int],
    dataset: List[Tuple[str, str, float]]
) -> Tuple[float, int, int]:
    """
    Evaluate word similarity using Spearman correlation.
    
    Args:
        embeddings: (vocab_size, dim) normalized embeddings
        word2idx: Vocabulary mapping
        dataset: List of (word1, word2, human_score) tuples
    
    Returns:
        (spearman_rho, num_found, num_total)
    """
    # Normalize embeddings (L2 norm)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    embeddings_norm = embeddings / norms
    
    model_scores = []
    human_scores = []
    num_found = 0
    num_total = len(dataset)
    
    for word1, word2, human_score in dataset:
        # Check if both words are in vocabulary
        idx1 = word2idx.get(word1, -1)
        idx2 = word2idx.get(word2, -1)
        
        if idx1 == -1 or idx2 == -1:
            continue
        
        # Compute cosine similarity
        emb1 = embeddings_norm[idx1]
        emb2 = embeddings_norm[idx2]
        cosine_sim = np.dot(emb1, emb2)
        
        model_scores.append(cosine_sim)
        human_scores.append(human_score)
        num_found += 1
    
    if num_found == 0:
        return 0.0, 0, num_total
    
    if len(model_scores) < 2:
        return 0.0, num_found, num_total
    
    # Compute Spearman correlation
    rho, _ = spearman_correlation(model_scores, human_scores)
    
    return rho, num_found, num_total

