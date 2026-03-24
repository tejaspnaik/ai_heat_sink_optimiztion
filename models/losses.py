"""
Custom loss functions for PINN training.
"""
import torch
import torch.nn.functional as F


class PINNLoss:
    """Compute weighted loss combining physics and data constraints."""
    
    def __init__(self, weight_pde=1.0, weight_bc=10.0, weight_data=1.0):
        """
        Initialize loss weights.
        
        Args:
            weight_pde: Weight for PDE residual loss
            weight_bc: Weight for boundary condition loss
            weight_data: Weight for data/measurement loss
        """
        self.weight_pde = weight_pde
        self.weight_bc = weight_bc
        self.weight_data = weight_data
    
    @staticmethod
    def mse(y_true, y_pred):
        """Mean squared error loss."""
        return torch.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mae(y_true, y_pred):
        """Mean absolute error loss."""
        return torch.mean(torch.abs(y_true - y_pred))
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Root mean squared error loss."""
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def relative_error(y_true, y_pred):
        """Relative L2 error: ||y_true - y_pred|| / ||y_true||."""
        numerator = torch.norm(y_true - y_pred, p=2)
        denominator = torch.norm(y_true, p=2)
        return numerator / (denominator + 1e-8)
    
    def pde_loss(self, residuals):
        """
        Compute PDE residual loss.
        
        Args:
            residuals: Tuple of residuals (residual_u, residual_v, etc.)
        
        Returns:
            Weighted sum of residuals
        """
        loss = 0.0
        for residual in residuals:
            loss += self.mse(residual, torch.zeros_like(residual))
        
        return self.weight_pde * loss / len(residuals)
    
    def boundary_loss(self, bc_residuals):
        """
        Compute boundary condition loss.
        
        Args:
            bc_residuals: Tuple of BC residuals
        
        Returns:
            Weighted sum of BC residuals
        """
        loss = 0.0
        for residual in bc_residuals:
            loss += self.mse(residual, torch.zeros_like(residual))
        
        return self.weight_bc * loss / len(bc_residuals)
    
    def data_loss(self, y_true, y_pred):
        """
        Compute data/measurement loss.
        
        Args:
            y_true: Ground truth data from CFD
            y_pred: Network predictions
        
        Returns:
            Weighted data loss
        """
        return self.weight_data * self.mse(y_true, y_pred)
    
    def total_loss(self, pde_res, bc_res, y_true=None, y_pred=None):
        """
        Compute total weighted loss.
        
        Args:
            pde_res: PDE residuals
            bc_res: Boundary condition residuals
            y_true: Ground truth (optional)
            y_pred: Predictions (optional)
        
        Returns:
            Total loss
        """
        loss_pde = self.pde_loss(pde_res)
        loss_bc = self.boundary_loss(bc_res)
        
        if y_true is not None and y_pred is not None:
            loss_data = self.data_loss(y_true, y_pred)
            return loss_pde + loss_bc + loss_data
        
        return loss_pde + loss_bc
    
    def update_weights(self, weight_pde=None, weight_bc=None, weight_data=None):
        """Dynamically update loss weights during training."""
        if weight_pde is not None:
            self.weight_pde = weight_pde
        if weight_bc is not None:
            self.weight_bc = weight_bc
        if weight_data is not None:
            self.weight_data = weight_data


class AdaptiveLoss(PINNLoss):
    """
    Adaptive loss weighting that adjusts weights based on loss components.
    Helps prevent one component from dominating training.
    """
    
    def __init__(self, initial_weight_pde=1.0, initial_weight_bc=10.0, 
                 initial_weight_data=1.0, adaptation_rate=0.01):
        """
        Initialize adaptive loss.
        
        Args:
            initial_weight_pde: Initial PDE weight
            initial_weight_bc: Initial BC weight
            initial_weight_data: Initial data weight
            adaptation_rate: Rate of weight adaptation (0-1)
        """
        super().__init__(initial_weight_pde, initial_weight_bc, initial_weight_data)
        self.adaptation_rate = adaptation_rate
        self.history_pde = []
        self.history_bc = []
        self.history_data = []
    
    def adapt_weights(self, loss_pde, loss_bc, loss_data=None):
        """
        Adapt loss weights based on current losses.
        Increases weight of smallest loss to balance components.
        
        Args:
            loss_pde: Current PDE loss
            loss_bc: Current BC loss
            loss_data: Current data loss (optional)
        """
        self.history_pde.append(loss_pde.item())
        self.history_bc.append(loss_bc.item())
        if loss_data is not None:
            self.history_data.append(loss_data.item())
        
        # Normalize losses for comparison
        max_loss = max(loss_pde.item(), loss_bc.item())
        if loss_data is not None:
            max_loss = max(max_loss, loss_data.item())
        
        normalized_pde = loss_pde.item() / (max_loss + 1e-8)
        normalized_bc = loss_bc.item() / (max_loss + 1e-8)
        
        # Increase weight for smaller normalized losses
        if normalized_pde < normalized_bc:
            self.weight_pde += self.adaptation_rate
        else:
            self.weight_bc += self.adaptation_rate
        
        # Normalize weights
        total_weight = self.weight_pde + self.weight_bc
        if loss_data is not None:
            total_weight += self.weight_data
        
        self.weight_pde /= total_weight
        self.weight_bc /= total_weight
        if loss_data is not None:
            self.weight_data /= total_weight
    
    def get_weight_history(self):
        """Return history of losses for visualization."""
        return {
            'pde': self.history_pde,
            'bc': self.history_bc,
            'data': self.history_data
        }


def conservation_loss(u_x, v_y):
    """
    Mass conservation loss: ∇·u = 0
    
    Args:
        u_x: ∂u/∂x
        v_y: ∂v/∂y
    
    Returns:
        MSE of divergence (should be zero)
    """
    divergence = u_x + v_y
    return torch.mean(divergence ** 2)


def energy_balance_loss(T_source, T_sink, flow_rate, cp=4186.0):
    """
    Energy balance loss: Q_in = Q_out + ΔT·ṁ·cp
    
    Args:
        T_source: Temperature at heat source
        T_sink: Temperature at heat sink
        flow_rate: Mass flow rate
        cp: Specific heat capacity
    
    Returns:
        Energy balance error
    """
    energy_transfer = (T_source - T_sink) * flow_rate * cp
    return torch.abs(energy_transfer)
