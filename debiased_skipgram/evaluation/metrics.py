"""Evaluation metrics utilities."""

from scipy.stats import spearmanr
import numpy as np
from typing import List, Tuple


def spearman_correlation(
    x: List[float],
    y: List[float]
) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation coefficient.
    
    Args:
        x: First variable
        y: Second variable
    
    Returns:
        (correlation, p-value)
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    if len(x) < 2:
        return 0.0, 1.0
    
    correlation, p_value = spearmanr(x, y)
    
    # Handle NaN (can occur if all values are the same)
    if np.isnan(correlation):
        correlation = 0.0
    if np.isnan(p_value):
        p_value = 1.0
    
    return float(correlation), float(p_value)


def accuracy(y_true: List, y_pred: List) -> float:
    """
    Compute accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if len(y_true) == 0:
        return 0.0
    
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

