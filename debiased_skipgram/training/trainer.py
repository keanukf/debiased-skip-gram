"""Training loop for Skip-gram model."""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
import numpy as np

from debiased_skipgram.config import Config


class Trainer:
    """Trainer for Skip-gram model."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        dataloader: DataLoader,
        config: Config
    ):
        """
        Initialize trainer.
        
        Args:
            model: SkipGram model
            loss_fn: Loss function (SGNSLoss or DebiasedSGNSLoss)
            dataloader: DataLoader for training data
            config: Configuration object
        """
        self.model = model
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Use Adam optimizer
        # Note: Embeddings use sparse=False because Adam doesn't support sparse gradients
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Training history
        self.history = {
            'loss': [],
            'epoch_losses': []
        }
        
        # Initial learning rate for decay
        self.initial_lr = config.learning_rate
        self.min_lr = 0.0001 * config.learning_rate
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train for configured number of epochs.
        
        Returns:
            Dictionary with training history
        """
        print(f"\nStarting training for {self.config.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Initial learning rate: {self.initial_lr}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Total batches per epoch: {len(self.dataloader)}\n")
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_loss = self._train_epoch(epoch)
            self.history['epoch_losses'].append(epoch_loss)
            
            # Learning rate decay
            if self.config.lr_decay:
                self._update_learning_rate(epoch)
        
        print("\nTraining completed!")
        return self.history
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch}/{self.config.epochs}",
            unit="batch"
        )
        
        for batch_idx, (center_ids, context_ids) in enumerate(pbar):
            # Move to device
            center_ids = center_ids.to(self.device)
            context_ids = context_ids.to(self.device)
            
            # Forward pass
            scores = self.model(center_ids, context_ids)
            
            # Compute loss
            loss = self.loss_fn(scores)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            self.history['loss'].append(loss.item())
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        avg_epoch_loss = total_loss / num_batches
        print(f"Epoch {epoch} average loss: {avg_epoch_loss:.4f}")
        
        return avg_epoch_loss
    
    def _update_learning_rate(self, epoch: int):
        """Update learning rate with linear decay."""
        if self.config.lr_decay:
            # Linear decay from initial_lr to min_lr over epochs
            progress = (epoch - 1) / self.config.epochs
            new_lr = self.initial_lr - progress * (self.initial_lr - self.min_lr)
            new_lr = max(new_lr, self.min_lr)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            if epoch % 1 == 0:  # Print every epoch
                print(f"Learning rate: {new_lr:.6f}")

