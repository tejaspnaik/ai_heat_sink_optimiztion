"""
Time-Dependent / Transient Physics-Informed Neural Network Solver
Supports unsteady flows and dynamic heat transfer problems
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional, Callable
import numpy as np
from dataclasses import dataclass


@dataclass
class TimeIntegrationConfig:
    """Configuration for time integration schemes"""
    scheme: str = "implicit_euler"  # implicit_euler, crank_nicolson, implicit_rk3
    dt: float = 0.01  # Time step size
    t_start: float = 0.0
    t_end: float = 1.0
    num_time_steps: int = 100
    num_sub_steps: int = 1  # Substeps for RK methods
    adapt_dt: bool = False  # Adaptive time stepping
    dt_max: float = 0.1
    dt_min: float = 0.001


class TransientTrainer:
    """
    Trainer for time-dependent PINN problems
    Handles temporal evolution of flow fields
    """
    
    def __init__(self, model: nn.Module, physics_constraint: Callable,
                 config: TimeIntegrationConfig = None):
        """
        Args:
            model: Neural network model taking (x, y, z, t) and outputting (u, v, w, p, T)
            physics_constraint: Physics constraint function (e.g., TransientPhysicsConstraints3D)
            config: Time integration configuration
        """
        self.model = model
        self.physics = physics_constraint
        self.config = config or TimeIntegrationConfig()
        
        self.time_steps = torch.linspace(
            self.config.t_start,
            self.config.t_end,
            self.config.num_time_steps
        )
        self.current_time = self.config.t_start
        self.step_count = 0
    
    def implicit_euler_step(self, x_train: torch.Tensor, t: float, dt: float,
                           optimizer: torch.optim.Optimizer, loss_fn: Callable,
                           previous_solution: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        One implicit Euler step: u^(n+1) = u^n + dt*f(u^(n+1), t^(n+1))
        Solved iteratively using implicit differentiation
        
        Args:
            x_train: Spatial coordinates (batch_size, 3)
            t: Current time
            dt: Time step size
            optimizer: PyTorch optimizer
            loss_fn: Loss function combining physics and data
            previous_solution: Solution at previous time step u^n
            
        Returns:
            Dictionary with loss metrics
        """
        t_next = t + dt
        device = x_train.device
        
        # Augment spatial coords with time
        t_tensor = torch.full((x_train.size(0), 1), t_next, device=device)
        xyt = torch.cat([x_train, t_tensor], dim=1)
        
        losses = {'total': 0, 'physics': 0, 'temporal': 0}
        
        for _ in range(self.config.num_sub_steps):
            optimizer.zero_grad()
            
            # Network prediction at t^(n+1)
            u_next = self.model(xyt)
            
            # Physics residuals at t^(n+1)
            physics_loss = loss_fn(u_next, xyt, self.physics)
            
            # Temporal coupling (if previous solution available)
            temporal_loss = torch.tensor(0.0, device=device)
            if previous_solution is not None:
                # Implicit Euler: (u^(n+1) - u^n) / dt - RHS(u^(n+1)) = 0
                # This is incorporated into the physics loss through time derivatives
                temporal_loss = torch.mean((u_next - previous_solution) ** 2) / (dt ** 2)
            
            total_loss = physics_loss + 0.1 * temporal_loss  # Weight temporal term
            
            total_loss.backward()
            optimizer.step()
            
            losses['physics'] += physics_loss.item()
            losses['temporal'] += temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else 0
            losses['total'] += total_loss.item()
        
        return {k: v / self.config.num_sub_steps for k, v in losses.items()}
    
    def crank_nicolson_step(self, x_train: torch.Tensor, t: float, dt: float,
                           optimizer: torch.optim.Optimizer, loss_fn: Callable,
                           previous_solution: torch.Tensor) -> Dict[str, float]:
        """
        Crank-Nicolson scheme: u^(n+1) = u^n + dt/2 * [f(u^n, t^n) + f(u^(n+1), t^(n+1))]
        Second-order accurate, better stability than implicit Euler
        
        Args:
            x_train: Spatial coordinates
            t: Current time
            dt: Time step size
            optimizer: PyTorch optimizer
            loss_fn: Loss function
            previous_solution: Solution at previous time step u^n
            
        Returns:
            Loss metrics
        """
        t_mid = t + 0.5 * dt
        t_next = t + dt
        device = x_train.device
        
        losses = {'total': 0, 'physics': 0, 'temporal': 0}
        
        for _ in range(self.config.num_sub_steps):
            optimizer.zero_grad()
            
            # Evaluate at midpoint and next step
            t_mid_tensor = torch.full((x_train.size(0), 1), t_mid, device=device)
            xyt_mid = torch.cat([x_train, t_mid_tensor], dim=1)
            
            u_mid = self.model(xyt_mid)
            
            # Physics residual and temporal coupling
            physics_loss = loss_fn(u_mid, xyt_mid, self.physics)
            
            # Temporal constraint: (u^(n+1) - u^n) / dt ≈ f(u^mid, t^mid)
            # Since we don't have explicit u^(n+1), use midpoint as approximation
            temporal_loss = torch.mean((u_mid - previous_solution) ** 2) / (dt ** 2)
            
            total_loss = physics_loss + 0.1 * temporal_loss
            
            total_loss.backward()
            optimizer.step()
            
            losses['physics'] += physics_loss.item()
            losses['temporal'] += temporal_loss.item()
            losses['total'] += total_loss.item()
        
        return {k: v / self.config.num_sub_steps for k, v in losses.items()}
    
    def rk3_step(self, x_train: torch.Tensor, t: float, dt: float,
                optimizer: torch.optim.Optimizer, loss_fn: Callable) -> Dict[str, float]:
        """
        Third-order explicit Runge-Kutta (RK3) step
        Good for moderate stiffness problems
        
        k1 = f(u^n, t^n)
        k2 = f(u^n + 0.5*dt*k1, t^n + 0.5*dt)
        k3 = f(u^n - dt*k1 + 2*dt*k2, t^n + dt)
        u^(n+1) = u^n + dt/6 * (k1 + 4*k2 + k3)
        
        Args:
            x_train: Spatial coordinates
            t: Current time
            dt: Time step size
            optimizer: PyTorch optimizer
            loss_fn: Loss function
            
        Returns:
            Loss metrics
        """
        device = x_train.device
        losses = {'total': 0, 'physics': 0}
        
        stages = [
            (t, 1.0),
            (t + 0.5 * dt, 0.5),
            (t + dt, 2.0),
        ]
        
        for stage_t, weight in stages:
            optimizer.zero_grad()
            
            t_tensor = torch.full((x_train.size(0), 1), stage_t, device=device)
            xyt = torch.cat([x_train, t_tensor], dim=1)
            
            u = self.model(xyt)
            physics_loss = loss_fn(u, xyt, self.physics)
            
            loss = weight * physics_loss
            loss.backward()
            optimizer.step()
            
            losses['physics'] += physics_loss.item() * weight
            losses['total'] += loss.item()
        
        return {k: v / sum(w for _, w in stages) for k, v in losses.items()}
    
    def adaptive_timestepping(self, error_estimate: float, dt_current: float) -> float:
        """
        Adjust time step based on error estimate
        Uses standard adaptive algorithm: dt_new = dt * (eps / error)^(1/p)
        
        Args:
            error_estimate: Estimated local truncation error
            dt_current: Current time step
            
        Returns:
            New time step size
        """
        p = 2  # Order for Crank-Nicolson
        eps = 1e-5  # Target error
        
        if error_estimate < eps:
            # Error is small, increase time step
            dt_new = dt_current * (eps / (2 * error_estimate)) ** (1 / p)
        else:
            # Error is large, decrease time step
            dt_new = dt_current * (eps / error_estimate) ** (1 / p)
        
        # Clamp to bounds
        dt_new = torch.clamp(torch.tensor(dt_new),
                            min=self.config.dt_min,
                            max=self.config.dt_max).item()
        
        return dt_new
    
    def evolve_solution(self, x_train: torch.Tensor, u_initial: torch.Tensor,
                       optimizer: torch.optim.Optimizer, loss_fn: Callable,
                       callback: Optional[Callable] = None) -> List[torch.Tensor]:
        """
        Evolve solution from t_start to t_end
        
        Args:
            x_train: Fixed spatial grid
            u_initial: Initial condition at t=0
            optimizer: PyTorch optimizer
            loss_fn: Loss function
            callback: Optional function called at each time step (for callbacks/logging)
            
        Returns:
            List of solutions at each time step
        """
        solutions = [u_initial.detach().clone()]
        current_solution = u_initial
        
        t = self.config.t_start
        dt = self.config.dt
        step = 0
        
        while t < self.config.t_end and step < self.config.num_time_steps:
            # Select stepping scheme
            if self.config.scheme == 'implicit_euler':
                losses = self.implicit_euler_step(x_train, t, dt, optimizer, loss_fn, current_solution)
            elif self.config.scheme == 'crank_nicolson':
                losses = self.crank_nicolson_step(x_train, t, dt, optimizer, loss_fn, current_solution)
            elif self.config.scheme == 'implicit_rk3':
                losses = self.rk3_step(x_train, t, dt, optimizer, loss_fn)
            else:
                raise ValueError(f"Unknown time integration scheme: {self.config.scheme}")
            
            # Get current solution
            t_tensor = torch.full((x_train.size(0), 1), t + dt, device=x_train.device)
            xyt = torch.cat([x_train, t_tensor], dim=1)
            current_solution = self.model(xyt).detach().clone()
            solutions.append(current_solution)
            
            # Adaptive time stepping
            if self.config.adapt_dt:
                dt_old = dt
                dt = self.adaptive_timestepping(losses.get('physics', 1e-3), dt)
                if callback:
                    callback(step, t, dt, losses, dt != dt_old)
            else:
                if callback:
                    callback(step, t, dt, losses)
            
            t += dt
            step += 1
        
        return solutions
    
    def compute_energy_residual(self, x_train: torch.Tensor, t_train: torch.Tensor,
                               net_output: torch.Tensor) -> torch.Tensor:
        """
        Compute energy conservation residual for transient flows
        """
        xyt = torch.cat([x_train, t_train.view(-1, 1)], dim=1)
        
        u = net_output[:, 0:1]
        v = net_output[:, 1:2]
        w = net_output[:, 2:3]
        T = net_output[:, 4:5]
        
        # Kinetic energy: E_k = 0.5 * ρ * (u² + v² + w²)
        KE = 0.5 * (u**2 + v**2 + w**2)
        
        # Time derivative of KE
        xyt.requires_grad_(True)
        dKE_dt = torch.autograd.grad(KE.sum(), xyt, create_graph=True)[0][:, 3:4]
        
        return dKE_dt


class DataAugmentationTemporal:
    """
    Augment spatial data with temporal dimension for unsteady problems
    """
    
    @staticmethod
    def create_spatiotemporal_data(x_data: np.ndarray, t_data: np.ndarray) -> np.ndarray:
        """
        Create spatiotemporal grid from spatial and temporal data
        
        Args:
            x_data: Spatial coordinates (N, 3) for (x, y, z)
            t_data: Time values (T,)
            
        Returns:
            Spatiotemporal coordinates (N*T, 4)
        """
        N = x_data.shape[0]
        T = t_data.shape[0]
        
        xyt_data = np.zeros((N * T, 4))
        
        for i in range(T):
            idx_start = i * N
            idx_end = (i + 1) * N
            xyt_data[idx_start:idx_end, :3] = x_data
            xyt_data[idx_start:idx_end, 3] = t_data[i]
        
        return xyt_data
    
    @staticmethod
    def create_temporal_sequence(u_data: np.ndarray, num_lookback: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create temporal sequences for recurrent-like processing
        
        Args:
            u_data: Solution data (T, N, variables)
            num_lookback: Number of past timesteps to include
            
        Returns:
            (input_sequences, output_sequences)
        """
        T, N, _ = u_data.shape
        n_seq = T - num_lookback
        
        inputs = np.zeros((n_seq, num_lookback, N, u_data.shape[2]))
        outputs = u_data[num_lookback:T]
        
        for i in range(n_seq):
            inputs[i] = u_data[i:i+num_lookback]
        
        return inputs, outputs


class WaveEquationPINN:
    """
    Specialized PINN for hyperbolic problems (wave equations, advection)
    ∂²u/∂t² - c²∇²u = 0
    """
    
    def __init__(self, model: nn.Module, c: float = 1.0):
        """
        Args:
            model: Neural network model
            c: Wave speed
        """
        self.model = model
        self.c = c
    
    def wave_residual(self, xyt: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute wave equation residual
        """
        xyt.requires_grad_(True)
        
        du_dxyt = torch.autograd.grad(u.sum(), xyt, create_graph=True, retain_graph=True)[0]
        du_dt = du_dxyt[:, -1:] if xyt.shape[1] > 3 else du_dxyt[:, 2:]
        
        # Second time derivative
        d2u_dt2 = torch.autograd.grad(du_dt.sum(), xyt, create_graph=True)[0][:, -1:]
        
        # Spatial Laplacian (simplified)
        d2u_dx2 = 0
        for i in range(xyt.shape[1] - 1):
            d2u = torch.autograd.grad(du_dxyt[:, i].sum(), xyt, create_graph=True)[0][:, i:i+1]
            d2u_dx2 += d2u
        
        residual = d2u_dt2 - (self.c ** 2) * d2u_dx2
        return residual


class SchrodingerEquationPINN:
    """
    Specialized PINN for Schrödinger equation (quantum flows)
    i∂ψ/∂t + (1/(2m))∇²ψ + V*ψ = 0
    """
    
    def __init__(self, model: nn.Module, m: float = 1.0, V: Optional[Callable] = None):
        """
        Args:
            model: Neural network predicting real and imaginary parts
            m: Effective mass
            V: Potential function V(x, y, z)
        """
        self.model = model
        self.m = m
        self.V = V or (lambda x: torch.zeros_like(x[:, 0:1]))
    
    def schrodinger_residual(self, xyt: torch.Tensor, psi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (real_residual, imag_residual)
        """
        xyt.requires_grad_(True)
        
        # Extract real and imaginary parts
        psi_real = psi[:, 0:1]
        psi_imag = psi[:, 1:2]
        
        # Time derivatives
        dpsi_r_dt = torch.autograd.grad(psi_real.sum(), xyt, create_graph=True, retain_graph=True)[0][:, -1:]
        dpsi_i_dt = torch.autograd.grad(psi_imag.sum(), xyt, create_graph=True, retain_graph=True)[0][:, -1:]
        
        # Laplacian (simplified)
        d2psi_r = 0
        d2psi_i = 0
        dpsi_r_dxyt = torch.autograd.grad(psi_real.sum(), xyt, create_graph=True, retain_graph=True)[0]
        dpsi_i_dxyt = torch.autograd.grad(psi_imag.sum(), xyt, create_graph=True, retain_graph=True)[0]
        
        for i in range(xyt.shape[1] - 1):
            d2psi_r += torch.autograd.grad(dpsi_r_dxyt[:, i].sum(), xyt, create_graph=True)[0][:, i:i+1]
            d2psi_i += torch.autograd.grad(dpsi_i_dxyt[:, i].sum(), xyt, create_graph=True)[0][:, i:i+1]
        
        V_val = self.V(xyt[:, :-1])
        
        # Schrödinger: i*dψ/dt + (1/2m)*∇²ψ + V*ψ = 0
        # Real: dpsi_i/dt + (1/2m)*d2psi_r + V*psi_r = 0
        # Imag: -dpsi_r/dt + (1/2m)*d2psi_i + V*psi_i = 0
        
        residual_r = dpsi_i_dt + (1 / (2 * self.m)) * d2psi_r + V_val * psi_real
        residual_i = -dpsi_r_dt + (1 / (2 * self.m)) * d2psi_i + V_val * psi_imag
        
        return residual_r, residual_i
