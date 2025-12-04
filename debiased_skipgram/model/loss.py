"""Loss functions for Skip-gram training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SGNSLoss(nn.Module):
    """Standard Skip-gram Negative Sampling loss with binary targets."""
    
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute standard SGNS loss.
        
        L = -log(σ(s_pos)) - Σ log(σ(-s_neg))
        
        Args:
            scores: (batch_size, 1 + num_negatives) tensor
                   First column is positive score, rest are negative scores
        
        Returns:
            loss: Scalar loss value
        """
        # Split positive and negative scores
        positive_scores = scores[:, 0]  # (batch_size,)
        negative_scores = scores[:, 1:]  # (batch_size, num_negatives)
        
        # Positive loss: -log(σ(s_pos))
        # Use logsigmoid for numerical stability
        positive_loss = -F.logsigmoid(positive_scores)
        
        # Negative loss: -Σ log(σ(-s_neg))
        # For each negative, compute -log(σ(-s_neg)) = -log(1 - σ(s_neg))
        # Using logsigmoid: log(σ(-s_neg)) = logsigmoid(-s_neg)
        negative_loss = -F.logsigmoid(-negative_scores)
        
        # Sum over negatives and average over batch
        loss = positive_loss.mean() + negative_loss.sum(dim=1).mean()
        
        return loss


class DebiasedSGNSLoss(nn.Module):
    """Debiased Skip-gram loss with soft targets."""
    
    def __init__(self, alpha: float, num_negatives: int):
        """
        Initialize debiased loss.
        
        Args:
            alpha: Debiasing parameter (0.0 = standard, >0 = debiased)
            num_negatives: Number of negative samples
        """
        super().__init__()
        self.alpha = alpha
        self.num_negatives = num_negatives
        
        # Pre-compute soft targets
        # positive_target = 1 - α·k/(k+1)
        # negative_target = α/(k+1)
        k = num_negatives
        self.positive_target = 1.0 - alpha * k / (k + 1)
        self.negative_target = alpha / (k + 1)
    
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute debiased SGNS loss with soft targets.
        
        For soft targets, use:
        L = -[t·log(σ(s)) + (1-t)·log(σ(-s))]
        
        Where t is the soft target (positive_target or negative_target).
        
        Args:
            scores: (batch_size, 1 + num_negatives) tensor
                   First column is positive score, rest are negative scores
        
        Returns:
            loss: Scalar loss value
        """
        # Split positive and negative scores
        positive_scores = scores[:, 0]  # (batch_size,)
        negative_scores = scores[:, 1:]  # (batch_size, num_negatives)
        
        # Positive loss: -[t_pos·log(σ(s_pos)) + (1-t_pos)·log(σ(-s_pos))]
        # = -t_pos·log(σ(s_pos)) - (1-t_pos)·log(σ(-s_pos))
        # = -t_pos·logsigmoid(s_pos) - (1-t_pos)·logsigmoid(-s_pos)
        positive_loss = (
            -self.positive_target * F.logsigmoid(positive_scores) -
            (1.0 - self.positive_target) * F.logsigmoid(-positive_scores)
        )
        
        # Negative loss: -[t_neg·log(σ(s_neg)) + (1-t_neg)·log(σ(-s_neg))]
        # For each negative
        negative_loss = (
            -self.negative_target * F.logsigmoid(negative_scores) -
            (1.0 - self.negative_target) * F.logsigmoid(-negative_scores)
        )
        
        # Sum over negatives and average over batch
        loss = positive_loss.mean() + negative_loss.sum(dim=1).mean()
        
        return loss

