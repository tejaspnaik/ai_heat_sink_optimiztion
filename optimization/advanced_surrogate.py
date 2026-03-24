"""
Advanced Surrogate Models: POD (Proper Orthogonal Decomposition) and ROM (Reduced Order Models)
Enables ultra-fast surrogate models for design optimization and uncertainty quantification
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict
from scipy.linalg import svd as scipy_svd
from scipy.interpolate import griddata


class PODReducer:
    """
    Proper Orthogonal Decomposition - extract dominant modes from solutions
    Reduces high-dimensional solution space to low-dimensional modal coefficients
    """
    
    def __init__(self, n_modes: int = 10):
        """
        Args:
            n_modes: Number of POD modes to retain
        """
        self.n_modes = n_modes
        self.mean = None
        self.modes = None  # POD basis (N, n_modes)
        self.singular_values = None
        self.energy_content = None
    
    def fit(self, snapshots: np.ndarray, center: bool = True) -> Dict[str, float]:
        """
        Compute POD basis from solution snapshots
        
        Args:
            snapshots: Solution snapshots (n_samples, n_features)
                      where n_features = spatial_dim * num_variables
            center: Whether to center data before SVD
            
        Returns:
            Dictionary with POD statistics
        """
        n_samples, n_features = snapshots.shape
        
        # Center data
        if center:
            self.mean = np.mean(snapshots, axis=0)
            X = snapshots - self.mean
        else:
            self.mean = np.zeros(n_features)
            X = snapshots
        
        # Compute SVD
        U, S, Vt = scipy_svd(X.T @ X / n_samples, full_matrices=False)
        
        # Extract modes
        n_modes = min(self.n_modes, len(S))
        self.modes = U[:, :n_modes]  # POD basis vectors
        self.singular_values = np.sqrt(S[:n_modes])
        
        # Compute energy content
        total_energy = np.sum(S)
        self.energy_content = np.cumsum(S[:n_modes]) / total_energy
        
        return {
            'n_modes': n_modes,
            'energy_retained': self.energy_content[-1] * 100,
            'singular_values': self.singular_values,
        }
    
    def project(self, snapshots: np.ndarray) -> np.ndarray:
        """
        Project snapshots onto POD basis
        
        Args:
            snapshots: Snapshots to project (n_samples, n_features)
            
        Returns:
            Modal coefficients (n_samples, n_modes)
        """
        if self.modes is None:
            raise ValueError("POD basis not fitted. Call fit() first.")
        
        centered = snapshots - self.mean
        coefficients = centered @ self.modes
        
        return coefficients
    
    def reconstruct(self, coefficients: np.ndarray, truncate_tol: float = 1e-6) -> np.ndarray:
        """
        Reconstruct snapshots from modal coefficients
        
        Args:
            coefficients: Modal coefficients (n_samples, n_modes)
            truncate_tol: Tolerance for coefficients (set small coefficients to zero)
            
        Returns:
            Reconstructed snapshots (n_samples, n_features)
        """
        if self.modes is None:
            raise ValueError("POD basis not fitted.")
        
        # Truncate small coefficients
        coefficients_trunc = coefficients.copy()
        coefficients_trunc[np.abs(coefficients_trunc) < truncate_tol] = 0
        
        # Reconstruct
        reconstructed = coefficients_trunc @ self.modes.T + self.mean
        
        return reconstructed
    
    def reconstruction_error(self, snapshots: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for each snapshot
        
        Args:
            snapshots: Original snapshots
            
        Returns:
            Relative reconstruction errors
        """
        coefficients = self.project(snapshots)
        reconstructed = self.reconstruct(coefficients)
        
        errors = np.linalg.norm(snapshots - reconstructed, axis=1)
        norms = np.linalg.norm(snapshots, axis=1)
        relative_errors = errors / (norms + 1e-10)
        
        return relative_errors
    
    def get_modal_energy(self) -> np.ndarray:
        """
        Get cumulative energy content of modes
        """
        return self.energy_content if self.energy_content is not None else np.array([])


class GaussianProcessROM:
    """
    Gaussian Process Reduced Order Model
    Maps parameter space to modal coefficient space using GP regression
    """
    
    def __init__(self, n_modes: int = 10, kernel: str = 'rbf'):
        """
        Args:
            n_modes: Number of POD modes
            kernel: GP kernel type ('rbf', 'matern', 'rational_quadratic')
        """
        self.n_modes = n_modes
        self.kernel = kernel
        self.gps = []  # One GP per mode
    
    def fit(self, parameters: np.ndarray, modal_coefficients: np.ndarray):
        """
        Build GP surrogates for each modal coefficient
        
        Args:
            parameters: Design parameters (n_samples, n_params)
            modal_coefficients: POD coefficients (n_samples, n_modes)
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern
        
        if self.kernel == 'rbf':
            kernel_obj = RBF(length_scale=1.0)
        elif self.kernel == 'matern':
            kernel_obj = Matern(length_scale=1.0, nu=2.5)
        else:
            kernel_obj = RBF(length_scale=1.0)
        
        self.gps = []
        for mode_idx in range(self.n_modes):
            gp = GaussianProcessRegressor(kernel=kernel_obj, alpha=1e-6)
            gp.fit(parameters, modal_coefficients[:, mode_idx])
            self.gps.append(gp)
    
    def predict(self, parameters: np.ndarray, return_std: bool = False):
        """
        Predict modal coefficients for new parameters
        
        Args:
            parameters: Design parameters (n_samples, n_params)
            return_std: Whether to return standard deviation
            
        Returns:
            Modal coefficients (n_samples, n_modes)
        """
        predictions = []
        stds = []
        
        for gp in self.gps:
            if return_std:
                pred, std = gp.predict(parameters, return_std=True)
                stds.append(std)
            else:
                pred = gp.predict(parameters)
            predictions.append(pred)
        
        modal_coeff = np.column_stack(predictions)
        
        if return_std:
            return modal_coeff, np.column_stack(stds)
        return modal_coeff


class NeuralNetworkROM(nn.Module):
    """
    Neural Network ROM: Maps parameters → POD modal coefficients
    Lightweight network for ultrafast surrogate modeling
    """
    
    def __init__(self, n_params: int, n_modes: int, hidden_sizes: List[int] = None):
        """
        Args:
            n_params: Number of design parameters
            n_modes: Number of POD modes
            hidden_sizes: Hidden layer dimensions
        """
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        
        layers = []
        prev_size = n_params
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, n_modes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """
        Predict modal coefficients from parameters
        
        Args:
            parameters: Design parameters (batch, n_params)
            
        Returns:
            Modal coefficients (batch, n_modes)
        """
        return self.network(parameters)
    
    def train_rom(self, param_train: torch.Tensor, coeff_train: torch.Tensor,
                 param_val: Optional[torch.Tensor] = None,
                 coeff_val: Optional[torch.Tensor] = None,
                 epochs: int = 100, lr: float = 0.001) -> Dict[str, List[float]]:
        """
        Train the ROM
        
        Args:
            param_train: Training parameters
            coeff_train: Training modal coefficients
            param_val: Validation parameters
            coeff_val: Validation coefficients
            epochs: Training epochs
            lr: Learning rate
            
        Returns:
            Training history
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.train()
            pred = self.forward(param_train)
            loss = criterion(pred, coeff_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            history['train_loss'].append(loss.item())
            
            # Validation
            if param_val is not None and coeff_val is not None:
                self.eval()
                with torch.no_grad():
                    pred_val = self.forward(param_val)
                    val_loss = criterion(pred_val, coeff_val)
                    history['val_loss'].append(val_loss.item())
        
        return history


class KarhunenLoeveSurrogate:
    """
    Karhunen-Loève expansion for probabilistic surrogate modeling
    """
    
    def __init__(self, n_modes: int = 10):
        self.n_modes = n_modes
        self.pod = PODReducer(n_modes)
        self.modal_coeff_mean = None
        self.modal_coeff_cov = None
    
    def fit(self, snapshots: np.ndarray, parameters: Optional[np.ndarray] = None):
        """
        Fit KL expansion to solution snapshots
        
        Args:
            snapshots: Solution snapshots
            parameters: Optional parameter values for each snapshot
        """
        # Compute POD
        stats = self.pod.fit(snapshots)
        
        # Compute modal coefficient statistics
        coefficients = self.pod.project(snapshots)
        self.modal_coeff_mean = np.mean(coefficients, axis=0)
        self.modal_coeff_cov = np.cov(coefficients.T)
        
        return stats
    
    def sample(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random samples using KL expansion
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            (samples, modal_coefficients)
        """
        # Sample modal coefficients from Gaussian
        coefficients = np.random.multivariate_normal(
            self.modal_coeff_mean,
            self.modal_coeff_cov,
            n_samples
        )
        
        # Reconstruct to full space
        samples = self.pod.reconstruct(coefficients)
        
        return samples, coefficients


class RBFInterpolantROM:
    """
    Radial Basis Function interpolation for ROM
    High-dimensional scattered data interpolation
    """
    
    def __init__(self, rbf_type: str = 'thin_plate_spline'):
        """
        Args:
            rbf_type: Type of RBF kernel
        """
        self.rbf_type = rbf_type
        self.interpolator = None
    
    def fit(self, parameters: np.ndarray, coefficients: np.ndarray):
        """
        Fit RBF interpolant
        
        Args:
            parameters: Design parameters (n_samples, n_params)
            coefficients: Modal coefficients (n_samples, n_modes)
        """
        from scipy.interpolate import Rbf
        
        # Fit one RBF per mode
        self.rbf_per_mode = []
        
        for mode_idx in range(coefficients.shape[1]):
            rbf = Rbf(*parameters.T, coefficients[:, mode_idx],
                     function=self.rbf_type)
            self.rbf_per_mode.append(rbf)
    
    def predict(self, parameters: np.ndarray) -> np.ndarray:
        """
        Predict modal coefficients via RBF interpolation
        
        Args:
            parameters: Query parameters
            
        Returns:
            Modal coefficients
        """
        predictions = []
        
        for rbf in self.rbf_per_mode:
            pred = rbf(*parameters.T)
            predictions.append(pred)
        
        return np.column_stack(predictions)


class AdaptiveROMBuilder:
    """
    Adaptive ROM construction with greedy sample selection
    Builds efficient ROMs with minimum sample requirements
    """
    
    def __init__(self, pod_reducer: PODReducer, n_modes: int = 10):
        self.pod = pod_reducer
        self.n_modes = n_modes
        self.selected_samples = []
        self.selected_indices = []
    
    def greedy_selection(self, snapshots: np.ndarray, n_samples: int = 50,
                        criterion: str = 'error') -> List[int]:
        """
        Greedily select representative samples for ROM training
        
        Args:
            snapshots: All available snapshots
            n_samples: Number of samples to select
            criterion: Selection criterion ('error', 'leverage', 'uncertainty')
            
        Returns:
            Indices of selected samples
        """
        n_total = snapshots.shape[0]
        remaining_indices = list(range(n_total))
        selected_indices = []
        
        # Select first sample (max norm)
        norms = np.linalg.norm(snapshots, axis=1)
        first_idx = np.argmax(norms)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Fit POD on selected
        self.pod.fit(snapshots[selected_indices])
        
        for _ in range(n_samples - 1):
            # Compute criterion for each remaining sample
            scores = []
            
            for idx in remaining_indices:
                snapshot = snapshots[idx:idx+1]
                
                if criterion == 'error':
                    # Reconstruction error
                    coeff = self.pod.project(snapshot)
                    recon = self.pod.reconstruct(coeff)
                    error = np.linalg.norm(snapshot - recon)
                    scores.append(error)
                
                elif criterion == 'leverage':
                    # Hat matrix diagonal (leverage)
                    coeff = self.pod.project(snapshot)
                    scores.append(np.linalg.norm(coeff))
                
                elif criterion == 'uncertainty':
                    # Distance to selected set
                    selected_set = snapshots[selected_indices]
                    distances = np.linalg.norm(
                        selected_set - snapshot.reshape(1, -1),
                        axis=1
                    )
                    scores.append(np.min(distances))
            
            # Select sample with highest score
            best_local_idx = np.argmax(scores)
            best_global_idx = remaining_indices[best_local_idx]
            
            selected_indices.append(best_global_idx)
            remaining_indices.pop(best_local_idx)
            
            # Refit POD with selected samples
            self.pod.fit(snapshots[selected_indices])
        
        self.selected_indices = selected_indices
        self.selected_samples = snapshots[selected_indices]
        
        return selected_indices


class HyperReducedROM:
    """
    Hyper-reduced ROM using Discrete Empirical Interpolation Method (DEIM)
    Reduces computational cost via sparse sampling in spatial domain
    """
    
    def __init__(self, pod_reducer: PODReducer):
        self.pod = pod_reducer
        self.deim_indices = []
    
    def compute_deim_indices(self, spatial_dimension: int, n_interpolation: int) -> List[int]:
        """
        Select interpolation points using DEIM algorithm
        
        Args:
            spatial_dimension: Dimension of spatial domain
            n_interpolation: Number of interpolation points
            
        Returns:
            Indices of selected interpolation points
        """
        modes = self.pod.modes[:spatial_dimension, :self.pod.n_modes]
        
        selected_indices = []
        residuals = np.zeros((spatial_dimension, 1))
        residuals[:] = modes[:, 0:1]  # Initial residual is first mode
        
        for k in range(n_interpolation):
            # Find index with max residual
            max_idx = np.argmax(np.abs(residuals))
            selected_indices.append(max_idx)
            
            # Update residuals (simplified)
            if k < self.pod.n_modes - 1:
                residuals[:] = modes[:, k+1:k+2]
        
        self.deim_indices = selected_indices
        return selected_indices
    
    def get_sparse_projection_matrix(self) -> np.ndarray:
        """
        Get sparse projection matrix for DEIM
        """
        if not self.deim_indices:
            raise ValueError("DEIM indices not computed. Call compute_deim_indices first.")
        
        n_interp = len(self.deim_indices)
        n_modes = self.pod.n_modes
        
        P = np.zeros((n_modes, n_interp))
        
        for i, idx in enumerate(self.deim_indices):
            P[i, i] = 1
        
        return P
