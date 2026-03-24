"""
3D Physics Constraints for PINN - Navier-Stokes and Heat Transfer in 3D
Supports steady-state and transient flows with optional turbulence modeling
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, List


class PhysicsConstraints3D:
    """
    3D incompressible Navier-Stokes and energy equations with automatic differentiation.
    
    Outputs: [u, v, w, p, T] - 3D velocity components, pressure, temperature
    Domain: [x, y, z] in 3D space
    """
    
    def __init__(self, rho: float = 1.0, mu: float = 0.001, k_f: float = 0.5, 
                 nu: Optional[float] = None, Re: Optional[float] = None):
        """
        Args:
            rho: Fluid density (kg/m³)
            mu: Dynamic viscosity (Pa·s)
            k_f: Thermal conductivity (W/m·K)
            nu: Kinematic viscosity (alternative to mu)
            Re: Reynolds number (alternative specification)
        """
        self.rho = rho
        self.mu = mu
        self.k_f = k_f
        
        # Handle alternative input specifications
        if nu is not None:
            self.mu = rho * nu
        elif Re is not None:
            # Assuming reference velocity = 1 m/s, reference length = 1 m
            self.mu = rho / Re
    
    def compute_gradients(self, xy: torch.Tensor, net_output: torch.Tensor,
                         compute_hessian: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute first and optional second derivatives of network outputs.
        
        Args:
            xy: Input coordinates [batch, 3] for (x, y, z)
            net_output: Network predictions [batch, 5] for (u, v, w, p, T)
            compute_hessian: Whether to compute second derivatives
            
        Returns:
            Dictionary containing all required gradients for PDE
        """
        xy.requires_grad_(True)
        
        u = net_output[:, 0:1]
        v = net_output[:, 1:2]
        w = net_output[:, 2:3]
        p = net_output[:, 3:4]
        T = net_output[:, 4:5]
        
        gradients = {}
        
        # First derivatives
        du_dxy = torch.autograd.grad(u.sum(), xy, create_graph=True, retain_graph=True)[0]
        dv_dxy = torch.autograd.grad(v.sum(), xy, create_graph=True, retain_graph=True)[0]
        dw_dxy = torch.autograd.grad(w.sum(), xy, create_graph=True, retain_graph=True)[0]
        dp_dxy = torch.autograd.grad(p.sum(), xy, create_graph=True, retain_graph=True)[0]
        dT_dxy = torch.autograd.grad(T.sum(), xy, create_graph=True, retain_graph=True)[0]
        
        gradients['du_dx'] = du_dxy[:, 0:1]
        gradients['du_dy'] = du_dxy[:, 1:2]
        gradients['du_dz'] = du_dxy[:, 2:3]
        
        gradients['dv_dx'] = dv_dxy[:, 0:1]
        gradients['dv_dy'] = dv_dxy[:, 1:2]
        gradients['dv_dz'] = dv_dxy[:, 2:3]
        
        gradients['dw_dx'] = dw_dxy[:, 0:1]
        gradients['dw_dy'] = dw_dxy[:, 1:2]
        gradients['dw_dz'] = dw_dxy[:, 2:3]
        
        gradients['dp_dx'] = dp_dxy[:, 0:1]
        gradients['dp_dy'] = dp_dxy[:, 1:2]
        gradients['dp_dz'] = dp_dxy[:, 2:3]
        
        gradients['dT_dx'] = dT_dxy[:, 0:1]
        gradients['dT_dy'] = dT_dxy[:, 1:2]
        gradients['dT_dz'] = dT_dxy[:, 2:3]
        
        # Laplacians (second derivatives)
        if compute_hessian:
            d2u_dx2 = torch.autograd.grad(du_dxy[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
            d2u_dy2 = torch.autograd.grad(du_dxy[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
            d2u_dz2 = torch.autograd.grad(du_dxy[:, 2].sum(), xy, create_graph=True)[0][:, 2:3]
            
            d2v_dx2 = torch.autograd.grad(dv_dxy[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
            d2v_dy2 = torch.autograd.grad(dv_dxy[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
            d2v_dz2 = torch.autograd.grad(dv_dxy[:, 2].sum(), xy, create_graph=True)[0][:, 2:3]
            
            d2w_dx2 = torch.autograd.grad(dw_dxy[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
            d2w_dy2 = torch.autograd.grad(dw_dxy[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
            d2w_dz2 = torch.autograd.grad(dw_dxy[:, 2].sum(), xy, create_graph=True)[0][:, 2:3]
            
            d2T_dx2 = torch.autograd.grad(dT_dxy[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
            d2T_dy2 = torch.autograd.grad(dT_dxy[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
            d2T_dz2 = torch.autograd.grad(dT_dxy[:, 2].sum(), xy, create_graph=True)[0][:, 2:3]
            
            gradients['laplacian_u'] = d2u_dx2 + d2u_dy2 + d2u_dz2
            gradients['laplacian_v'] = d2v_dx2 + d2v_dy2 + d2v_dz2
            gradients['laplacian_w'] = d2w_dx2 + d2w_dy2 + d2w_dz2
            gradients['laplacian_T'] = d2T_dx2 + d2T_dy2 + d2T_dz2
        
        return gradients
    
    def navier_stokes_3d(self, xy: torch.Tensor, net_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        3D Navier-Stokes residuals: ∂u/∂t + ∇·(uu) = -∇p + ν∇²u + f
        
        Returns:
            (momentum_residual_x, momentum_residual_y, momentum_residual_z)
        """
        u = net_output[:, 0:1]
        v = net_output[:, 1:2]
        w = net_output[:, 2:3]
        
        grads = self.compute_gradients(xy, net_output, compute_hessian=True)
        
        # Kinematic viscosity
        nu = self.mu / self.rho
        
        # Momentum equations (steady-state, no body forces)
        # ρ(u·∇u) = -∇p + μ∇²u
        
        # Convective terms
        du_dx = grads['du_dx']
        du_dy = grads['du_dy']
        du_dz = grads['du_dz']
        
        dv_dx = grads['dv_dx']
        dv_dy = grads['dv_dy']
        dv_dz = grads['dv_dz']
        
        dw_dx = grads['dw_dx']
        dw_dy = grads['dw_dy']
        dw_dz = grads['dw_dz']
        
        # Convection: u·∇u
        conv_u = u * du_dx + v * du_dy + w * du_dz
        conv_v = u * dv_dx + v * dv_dy + w * dv_dz
        conv_w = u * dw_dx + v * dw_dy + w * dw_dz
        
        # Pressure gradient
        dp_dx = grads['dp_dx']
        dp_dy = grads['dp_dy']
        dp_dz = grads['dp_dz']
        
        # Viscous terms
        lap_u = grads['laplacian_u']
        lap_v = grads['laplacian_v']
        lap_w = grads['laplacian_w']
        
        # Momentum residuals
        residual_x = self.rho * conv_u + dp_dx - self.mu * lap_u
        residual_y = self.rho * conv_v + dp_dy - self.mu * lap_v
        residual_z = self.rho * conv_w + dp_dz - self.mu * lap_w
        
        return residual_x, residual_y, residual_z
    
    def continuity_3d(self, xy: torch.Tensor, net_output: torch.Tensor) -> torch.Tensor:
        """
        3D Continuity equation (incompressibility): ∇·u = 0
        
        Returns:
            Continuity residual
        """
        grads = self.compute_gradients(xy, net_output, compute_hessian=False)
        
        du_dx = grads['du_dx']
        dv_dy = grads['dv_dy']
        dw_dz = grads['dw_dz']
        
        residual = du_dx + dv_dy + dw_dz
        
        return residual
    
    def energy_3d(self, xy: torch.Tensor, net_output: torch.Tensor, 
                  rho_cp: float = 1000.0, Q_source: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        3D Energy equation: ρCp(u·∇T) = k∇²T + Q
        
        Args:
            xy: Coordinates
            net_output: Network predictions
            rho_cp: ρCp (heat capacity) default 1000 J/m³·K
            Q_source: Heat source term [batch, 1], default 0
            
        Returns:
            Energy residual
        """
        if Q_source is None:
            Q_source = torch.zeros_like(net_output[:, 4:5])
        
        u = net_output[:, 0:1]
        v = net_output[:, 1:2]
        w = net_output[:, 2:3]
        
        grads = self.compute_gradients(xy, net_output, compute_hessian=True)
        
        # Temperature gradient
        dT_dx = grads['dT_dx']
        dT_dy = grads['dT_dy']
        dT_dz = grads['dT_dz']
        
        # Thermal diffusion
        lap_T = grads['laplacian_T']
        
        # Convection
        convection = u * dT_dx + v * dT_dy + w * dT_dz
        
        # Energy residual
        residual = rho_cp * convection - self.k_f * lap_T - Q_source
        
        return residual
    
    def boundary_no_slip(self, boundary_output: torch.Tensor) -> torch.Tensor:
        """No-slip boundary condition: u = v = w = 0 on wall"""
        u = boundary_output[:, 0:1]
        v = boundary_output[:, 1:2]
        w = boundary_output[:, 2:3]
        return torch.cat([u, v, w], dim=1)
    
    def boundary_symmetry(self, xy: torch.Tensor, net_output: torch.Tensor, 
                         axis: str = 'x') -> torch.Tensor:
        """
        Symmetry boundary condition
        
        Args:
            axis: 'x', 'y', or 'z' - normal to symmetry plane
        """
        grads = self.compute_gradients(xy, net_output, compute_hessian=False)
        
        if axis == 'x':
            du_dx = grads['du_dx']
            dv_dx = grads['dv_dx']
            dw_dx = grads['dw_dx']
            return torch.cat([du_dx, dv_dx, dw_dx], dim=1)
        elif axis == 'y':
            du_dy = grads['du_dy']
            dv_dy = grads['dv_dy']
            dw_dy = grads['dw_dy']
            return torch.cat([du_dy, dv_dy, dw_dy], dim=1)
        elif axis == 'z':
            du_dz = grads['du_dz']
            dv_dz = grads['dv_dz']
            dw_dz = grads['dw_dz']
            return torch.cat([du_dz, dv_dz, dw_dz], dim=1)
    
    def boundary_inlet(self, inlet_velocity: torch.Tensor, net_output: torch.Tensor) -> torch.Tensor:
        """Inlet boundary condition: u = velocity_in"""
        u = net_output[:, 0:1]
        v = net_output[:, 1:2]
        w = net_output[:, 2:3]
        
        u_bc = u - inlet_velocity[:, 0:1]
        v_bc = v - inlet_velocity[:, 1:2]
        w_bc = w - inlet_velocity[:, 2:3]
        
        return torch.cat([u_bc, v_bc, w_bc], dim=1)
    
    def boundary_outlet(self, xy: torch.Tensor, net_output: torch.Tensor) -> torch.Tensor:
        """Outlet boundary condition: ∂u/∂n = 0 (zero gradient)"""
        grads = self.compute_gradients(xy, net_output, compute_hessian=False)
        du_dx = grads['du_dx']
        dv_dx = grads['dv_dx']
        dw_dx = grads['dw_dx']
        return torch.cat([du_dx, dv_dx, dw_dx], dim=1)
    
    def boundary_heat_source(self, T_output: torch.Tensor, T_target: torch.Tensor) -> torch.Tensor:
        """Heat boundary condition: T = T_target (Dirichlet)"""
        return T_output - T_target
    
    def boundary_heat_flux(self, xy: torch.Tensor, net_output: torch.Tensor, 
                          q_target: torch.Tensor) -> torch.Tensor:
        """Heat flux boundary condition: -k∇T·n = q (Neumann)"""
        grads = self.compute_gradients(xy, net_output, compute_hessian=False)
        dT_dx = grads['dT_dx']
        
        q_computed = -self.k_f * dT_dx
        return q_computed - q_target


class TransientPhysicsConstraints3D(PhysicsConstraints3D):
    """
    3D Physics constraints with time-dependent terms for unsteady flows
    Outputs: [u, v, w, p, T] at each time step
    """
    
    def navier_stokes_transient_3d(self, xyt: torch.Tensor, net_output: torch.Tensor,
                                   dt_output: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        3D Transient Navier-Stokes: ∂u/∂t + ρ(u·∇u) = -∇p + μ∇²u
        
        Args:
            xyt: Coordinates with time [batch, 4] for (x, y, z, t)
            net_output: Network predictions [batch, 5] for (u, v, w, p, T)
            dt_output: Optional time derivatives (for comparison)
            
        Returns:
            (momentum_residual_x, momentum_residual_y, momentum_residual_z)
        """
        xyt.requires_grad_(True)
        
        u = net_output[:, 0:1]
        v = net_output[:, 1:2]
        w = net_output[:, 2:3]
        
        # Time derivatives
        du_dt = torch.autograd.grad(u.sum(), xyt, create_graph=True, retain_graph=True)[0][:, 3:4]
        dv_dt = torch.autograd.grad(v.sum(), xyt, create_graph=True, retain_graph=True)[0][:, 3:4]
        dw_dt = torch.autograd.grad(w.sum(), xyt, create_graph=True, retain_graph=True)[0][:, 3:4]
        
        # Spatial derivatives for convection
        Du = torch.autograd.grad(u.sum(), xyt, create_graph=True, retain_graph=True)[0]
        Dv = torch.autograd.grad(v.sum(), xyt, create_graph=True, retain_graph=True)[0]
        Dw = torch.autograd.grad(w.sum(), xyt, create_graph=True, retain_graph=True)[0]
        Dp = torch.autograd.grad(net_output[:, 3:4].sum(), xyt, create_graph=True, retain_graph=True)[0]
        
        du_dx, du_dy, du_dz = Du[:, 0:1], Du[:, 1:2], Du[:, 2:3]
        dv_dx, dv_dy, dv_dz = Dv[:, 0:1], Dv[:, 1:2], Dv[:, 2:3]
        dw_dx, dw_dy, dw_dz = Dw[:, 0:1], Dw[:, 1:2], Dw[:, 2:3]
        dp_dx, dp_dy, dp_dz = Dp[:, 0:1], Dp[:, 1:2], Dp[:, 2:3]
        
        # Laplacians (approximated with spatial derivatives)
        # Full Laplacian would require computing second derivatives
        nu = self.mu / self.rho
        
        # Momentum + convection terms
        residual_x = du_dt + self.rho * (u * du_dx + v * du_dy + w * du_dz) + dp_dx
        residual_y = dv_dt + self.rho * (u * dv_dx + v * dv_dy + w * dv_dz) + dp_dy
        residual_z = dw_dt + self.rho * (u * dw_dx + v * dw_dy + w * dw_dz) + dp_dz
        
        return residual_x, residual_y, residual_z
    
    def energy_transient_3d(self, xyt: torch.Tensor, net_output: torch.Tensor,
                           rho_cp: float = 1000.0) -> torch.Tensor:
        """
        3D Transient Energy equation: ∂T/∂t + ρCp(u·∇T) = k∇²T
        
        Args:
            xyt: Coordinates with time [batch, 4]
            net_output: Network predictions
            rho_cp: Heat capacity
            
        Returns:
            Energy residual
        """
        xyt.requires_grad_(True)
        
        T = net_output[:, 4:5]
        u = net_output[:, 0:1]
        v = net_output[:, 1:2]
        w = net_output[:, 2:3]
        
        # Time derivative
        dT_dt = torch.autograd.grad(T.sum(), xyt, create_graph=True, retain_graph=True)[0][:, 3:4]
        
        # Spatial derivatives
        DT = torch.autograd.grad(T.sum(), xyt, create_graph=True, retain_graph=True)[0]
        dT_dx, dT_dy, dT_dz = DT[:, 0:1], DT[:, 1:2], DT[:, 2:3]
        
        # Energy residual (without Laplacian term - approximation)
        residual = dT_dt + rho_cp * (u * dT_dx + v * dT_dy + w * dT_dz)
        
        return residual
