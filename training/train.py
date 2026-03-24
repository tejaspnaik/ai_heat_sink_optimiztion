"""
Training loop for PINN with checkpoint support for Colab.
"""
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import json


class PINNTrainer:
    """Trainer class with checkpoint support for Colab interruptions."""
    
    def __init__(self, model, optimizer, loss_fn, device='cpu', 
                 checkpoint_dir='checkpoints', checkpoint_interval=500):
        """
        Initialize trainer.
        
        Args:
            model: PINN model
            optimizer: PyTorch optimizer
            loss_fn: Loss function
            device: 'cpu' or 'cuda'
            checkpoint_dir: Directory for checkpoints
            checkpoint_interval: Save checkpoint every N epochs
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = {
            'epoch': [],
            'loss_total': [],
            'loss_pde': [],
            'loss_bc': [],
            'loss_data': [],
            'val_loss': []
        }
    
    def train_epoch(self, collocation_loader, bc_data, data_loader=None):
        """
        Train for one epoch.
        
        Args:
            collocation_loader: DataLoader for collocation points
            bc_data: Dictionary with boundary condition data
            data_loader: Optional DataLoader for CFD data
        
        Returns:
            Loss values
        """
        self.model.train()
        
        total_loss = 0
        total_pde_loss = 0
        total_bc_loss = 0
        total_data_loss = 0
        n_batches = 0
        
        for batch_idx, x_collocation in enumerate(collocation_loader):
            x_collocation = x_collocation.to(self.device).requires_grad_(True)
            
            # Forward pass
            y_pred = self.model(x_collocation)
            
            # Compute PDE residuals (simplified - implement full physics as needed)
            residuals = self._compute_pde_residuals(x_collocation, y_pred)
            loss_pde = self.loss_fn.pde_loss(residuals)
            
            # Boundary conditions
            loss_bc = self._compute_bc_loss(bc_data, y_pred)
            
            # Data loss (if available)
            loss_data = torch.tensor(0.0, device=self.device)
            if data_loader is not None:
                try:
                    x_data, y_data = next(data_loader_iter)
                    x_data = x_data.to(self.device)
                    y_data = y_data.to(self.device)
                    y_pred_data = self.model(x_data)
                    loss_data = self.loss_fn.data_loss(y_data, y_pred_data)
                except (StopIteration, NameError):
                    loss_data = torch.tensor(0.0, device=self.device)
            
            # Total loss
            loss = loss_pde + loss_bc + loss_data
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_pde_loss += loss_pde.item()
            total_bc_loss += loss_bc.item()
            total_data_loss += loss_data.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_pde_loss = total_pde_loss / n_batches
        avg_bc_loss = total_bc_loss / n_batches
        avg_data_loss = total_data_loss / n_batches
        
        self.epoch += 1
        
        return {
            'total': avg_loss,
            'pde': avg_pde_loss,
            'bc': avg_bc_loss,
            'data': avg_data_loss
        }
    
    def _compute_pde_residuals(self, x, y):
        """Compute PDE residuals."""
        # Simplified version - implement full physics equations here
        u = y[:, 0:1]
        v = y[:, 1:2]
        p = y[:, 2:3]
        T = y[:, 3:4]
        
        # Dummy residuals - replace with actual physics
        residuals = [
            torch.zeros_like(u),
            torch.zeros_like(v),
            torch.zeros_like(p),
            torch.zeros_like(T)
        ]
        
        return residuals
    
    def _compute_bc_loss(self, bc_data, y_pred):
        """Compute boundary condition loss."""
        # Simplified BC loss
        bc_residuals = [torch.zeros_like(y_pred[:, :1])]
        loss = self.loss_fn.boundary_loss(bc_residuals)
        return loss
    
    def save_checkpoint(self, filename=None):
        """Save model and training state."""
        if filename is None:
            filename = f'checkpoint_epoch_{self.epoch}.pt'
        
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'history': self.training_history
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
        
        return filepath
    
    def load_checkpoint(self, filepath):
        """Load model and training state from checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['history']
        
        print(f"Checkpoint loaded: {filepath}, epoch: {self.epoch}")
    
    def save_model(self, filename='pinn_model.pt'):
        """Save only model parameters."""
        filepath = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved: {filepath}")
    
    def load_model(self, filepath):
        """Load model parameters."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded: {filepath}")
    
    def record_history(self, losses, val_loss=None):
        """Record training history."""
        self.training_history['epoch'].append(self.epoch)
        self.training_history['loss_total'].append(losses['total'])
        self.training_history['loss_pde'].append(losses['pde'])
        self.training_history['loss_bc'].append(losses['bc'])
        self.training_history['loss_data'].append(losses['data'])
        if val_loss is not None:
            self.training_history['val_loss'].append(val_loss)
    
    def get_history(self):
        """Get training history."""
        return self.training_history
    
    def get_summary(self):
        """Get training summary."""
        if len(self.training_history['loss_total']) == 0:
            return "No training history"
        
        return {
            'total_epochs': self.epoch,
            'final_loss': self.training_history['loss_total'][-1],
            'best_loss': self.best_loss,
            'loss_reduction': (
                self.training_history['loss_total'][0] 
                / self.training_history['loss_total'][-1]
            )
        }


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience=10, min_delta=0.0):
        """
        Args:
            patience: Epochs with no improvement before stopping
            min_delta: Minimum improvement threshold
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
    
    def __call__(self, current_loss):
        """
        Check if should stop.
        
        Returns:
            True if should stop
        """
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            return self.wait >= self.patience


class LRScheduler:
    """Learning rate scheduler."""
    
    def __init__(self, optimizer, schedule_type='exponential', **kwargs):
        """
        Args:
            optimizer: PyTorch optimizer
            schedule_type: 'exponential', 'cosine', 'step'
            **kwargs: Additional parameters
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.initial_lr = optimizer.defaults['lr']
        self.epoch = 0
        self.kwargs = kwargs
    
    def step(self):
        """Update learning rate."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.epoch += 1
    
    def get_lr(self):
        """Compute current learning rate."""
        if self.schedule_type == 'exponential':
            decay = self.kwargs.get('decay', 0.95)
            return self.initial_lr * (decay ** self.epoch)
        
        elif self.schedule_type == 'step':
            step_size = self.kwargs.get('step_size', 10)
            gamma = self.kwargs.get('gamma', 0.1)
            return self.initial_lr * (gamma ** (self.epoch // step_size))
        
        elif self.schedule_type == 'cosine':
            T_max = self.kwargs.get('T_max', 100)
            eta_min = self.kwargs.get('eta_min', 0)
            return eta_min + (self.initial_lr - eta_min) * (
                1 + np.cos(np.pi * self.epoch / T_max)
            ) / 2
        
        return self.initial_lr
