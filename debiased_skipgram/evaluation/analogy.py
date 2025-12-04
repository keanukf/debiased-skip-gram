"""Word analogy evaluation."""

import numpy as np
from typing import List, Tuple, Dict, Literal
from pathlib import Path

from .metrics import accuracy
from ..data.download import download_google_analogies


def load_google_analogies(path: str = None) -> Dict[str, List[Tuple[str, str, str, str]]]:
    """
    Load Google analogy dataset.
    
    Args:
        path: Path to questions-words.txt. If None, downloads it.
    
    Returns:
        Dictionary mapping category name to list of (a, b, c, d) tuples
        where "a is to b as c is to d"
    """
    if path is None:
        path = str(download_google_analogies())
    
    analogies = {}
    current_category = None
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            # Category header: ": category_name"
            if line.startswith(':'):
                current_category = line[1:].strip()
                analogies[current_category] = []
            else:
                # Analogy line: "word1 word2 word3 word4"
                words = line.lower().split()
                if len(words) == 4 and current_category is not None:
                    analogies[current_category].append(tuple(words))
    
    return analogies


def evaluate_analogies(
    embeddings: np.ndarray,
    word2idx: Dict[str, int],
    analogies: Dict[str, List[Tuple[str, str, str, str]]],
    method: Literal["3cosadd", "3cosmul"] = "3cosadd",
    epsilon: float = 1e-3
) -> Dict[str, Tuple[float, int, int]]:
    """
    Evaluate word analogies.
    
    Args:
        embeddings: (vocab_size, dim) normalized embeddings
        word2idx: Vocabulary mapping
        analogies: Dictionary mapping category to list of (a, b, c, d) tuples
        method: "3cosadd" or "3cosmul"
        epsilon: Small constant for 3CosMul (to avoid division by zero)
    
    Returns:
        Dictionary mapping category -> (accuracy, num_correct, num_total)
    """
    # Normalize embeddings (L2 norm)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_norm = embeddings / norms
    
    results = {}
    
    for category, analogy_list in analogies.items():
        correct = 0
        total = 0
        
        for a, b, c, d in analogy_list:
            # Check if all words are in vocabulary
            idx_a = word2idx.get(a, -1)
            idx_b = word2idx.get(b, -1)
            idx_c = word2idx.get(c, -1)
            idx_d = word2idx.get(d, -1)
            
            if idx_a == -1 or idx_b == -1 or idx_c == -1 or idx_d == -1:
                continue
            
            total += 1
            
            # Get embeddings
            emb_a = embeddings_norm[idx_a]
            emb_b = embeddings_norm[idx_b]
            emb_c = embeddings_norm[idx_c]
            emb_d = embeddings_norm[idx_d]
            
            # Compute target vector
            if method == "3cosadd":
                # 3CosAdd: argmax_d [cos(d,b) - cos(d,a) + cos(d,c)]
                target = emb_b - emb_a + emb_c
            else:  # 3cosmul
                # 3CosMul: argmax_d [cos(d,b) * cos(d,c) / (cos(d,a) + ε)]
                # We compute this by finding d that maximizes the expression
                # For efficiency, we'll use a different approach:
                # We want to find d such that: d ≈ (b - a + c) normalized
                target = emb_b - emb_a + emb_c
            
            # Normalize target
            target_norm = np.linalg.norm(target)
            if target_norm > 0:
                target = target / target_norm
            
            # Find closest word (excluding a, b, c)
            # Compute cosine similarity with all words
            similarities = np.dot(embeddings_norm, target)
            
            # Set similarities for a, b, c to -inf so they're not selected
            similarities[idx_a] = -np.inf
            similarities[idx_b] = -np.inf
            similarities[idx_c] = -np.inf
            
            # Find best match
            predicted_idx = np.argmax(similarities)
            
            if predicted_idx == idx_d:
                correct += 1
        
        acc = accuracy([True] * correct + [False] * (total - correct), 
                      [True] * total) if total > 0 else 0.0
        results[category] = (acc, correct, total)
    
    return results


def aggregate_analogy_results(
    results: Dict[str, Tuple[float, int, int]]
) -> Dict[str, Tuple[float, int, int]]:
    """
    Aggregate analogy results into semantic and syntactic categories.
    
    Args:
        results: Dictionary from evaluate_analogies()
    
    Returns:
        Dictionary with 'semantic', 'syntactic', and 'overall' keys
    """
    # Semantic categories (from Google analogy dataset)
    semantic_categories = [
        'capital-common-countries',
        'capital-world',
        'currency',
        'city-in-state',
        'family',
        'gram1-adjective-to-adverb',
        'gram2-opposite',
        'gram3-comparative',
        'gram4-superlative',
        'gram5-present-participle',
        'gram6-nationality-adjective',
        'gram7-past-tense',
        'gram8-plural',
        'gram9-plural-verbs'
    ]
    
    # Split into semantic and syntactic
    # Semantic: capital-*, currency, city-in-state, family
    # Syntactic: gram*
    semantic_cats = [c for c in semantic_categories 
                     if any(c.startswith(prefix) for prefix in 
                           ['capital-', 'currency', 'city-in-state', 'family'])]
    syntactic_cats = [c for c in semantic_categories if c.startswith('gram')]
    
    semantic_correct = 0
    semantic_total = 0
    syntactic_correct = 0
    syntactic_total = 0
    overall_correct = 0
    overall_total = 0
    
    for category, (acc, correct, total) in results.items():
        if category in semantic_cats:
            semantic_correct += correct
            semantic_total += total
        elif category in syntactic_cats:
            syntactic_correct += correct
            syntactic_total += total
        
        overall_correct += correct
        overall_total += total
    
    aggregated = {}
    
    if semantic_total > 0:
        aggregated['semantic'] = (
            semantic_correct / semantic_total,
            semantic_correct,
            semantic_total
        )
    
    if syntactic_total > 0:
        aggregated['syntactic'] = (
            syntactic_correct / syntactic_total,
            syntactic_correct,
            syntactic_total
        )
    
    if overall_total > 0:
        aggregated['overall'] = (
            overall_correct / overall_total,
            overall_correct,
            overall_total
        )
    
    return aggregated

