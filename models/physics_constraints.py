"""
Physics constraints for PINN: Navier-Stokes and Heat Equations
"""
import torch
import torch.nn as nn


class PhysicsConstraints:
    """Define PDE residuals for fluid dynamics and heat transfer."""
    
    def __init__(self, rho=1000.0, mu=0.001, k_thermal=0.6, cp=4186.0):
        """
        Initialize physics constants for liquid cooling.
        
        Args:
            rho: Density of fluid (kg/m^3) - default: water
            mu: Dynamic viscosity (Pa·s) - default: water at 20°C
            k_thermal: Thermal conductivity (W/m·K) - default: water
            cp: Specific heat capacity (J/kg·K) - default: water
        """
        self.rho = rho
        self.mu = mu
        self.k_thermal = k_thermal
        self.cp = cp
    
    @staticmethod
    def compute_jacobian(y, x, i, j):
        """Compute ∂y_i/∂x_j using automatic differentiation."""
        grad = torch.autograd.grad(
            y[:, i].sum(), x, create_graph=True, retain_graph=True
        )[0]
        return grad[:, j].unsqueeze(1)
    
    @staticmethod
    def compute_hessian(y, x, component, i, j):
        """Compute ∂²y_component/∂x_i∂x_j."""
        grad_first = torch.autograd.grad(
            y[:, component].sum(), x, create_graph=True, retain_graph=True
        )[0]
        grad_second = torch.autograd.grad(
            grad_first[:, i].sum(), x, create_graph=True, retain_graph=True
        )[0]
        return grad_second[:, j].unsqueeze(1)
    
    def navier_stokes_2d(self, x, y):
        """
        2D incompressible Navier-Stokes equations.
        
        y = [u, v, p, T]
        x = [x_coord, y_coord]
        
        Returns:
            [residual_u, residual_v, residual_continuity, residual_T]
        """
        u = y[:, 0:1]
        v = y[:, 1:2]
        p = y[:, 2:3]
        T = y[:, 3:4]
        
        # Momentum equation x-component:
        # ρ(u·∇)u = -∇p + μ∇²u
        u_x = self.compute_jacobian(y, x, 0, 0)
        u_y = self.compute_jacobian(y, x, 0, 1)
        u_xx = self.compute_hessian(y, x, 0, 0, 0)
        u_yy = self.compute_hessian(y, x, 0, 1, 1)
        
        p_x = self.compute_jacobian(y, x, 2, 0)
        
        residual_u = (
            self.rho * (u * u_x + v * u_y) + p_x 
            - self.mu * (u_xx + u_yy)
        )
        
        # Momentum equation y-component:
        # ρ(u·∇)v = -∇p + μ∇²v
        v_x = self.compute_jacobian(y, x, 1, 0)
        v_y = self.compute_jacobian(y, x, 1, 1)
        v_xx = self.compute_hessian(y, x, 1, 0, 0)
        v_yy = self.compute_hessian(y, x, 1, 1, 1)
        
        p_y = self.compute_jacobian(y, x, 2, 1)
        
        residual_v = (
            self.rho * (u * v_x + v * v_y) + p_y 
            - self.mu * (v_xx + v_yy)
        )
        
        # Continuity equation: ∇·u = 0
        residual_continuity = u_x + v_y
        
        # Energy equation: ρcₚ(u·∇)T = k∇²T
        T_x = self.compute_jacobian(y, x, 3, 0)
        T_y = self.compute_jacobian(y, x, 3, 1)
        T_xx = self.compute_hessian(y, x, 3, 0, 0)
        T_yy = self.compute_hessian(y, x, 3, 1, 1)
        
        residual_T = (
            self.rho * self.cp * (u * T_x + v * T_y) 
            - self.k_thermal * (T_xx + T_yy)
        )
        
        return residual_u, residual_v, residual_continuity, residual_T
    
    def navier_stokes_2d_steady(self, x, y):
        """
        2D steady-state incompressible Navier-Stokes (no time derivatives).
        Simplified for faster training in Colab.
        """
        return self.navier_stokes_2d(x, y)
    
    def boundary_condition_wall(self, y_pred, y_bc_value):
        """
        No-slip boundary condition: u = 0, v = 0 at wall.
        
        Args:
            y_pred: Network output [u, v, p, T]
            y_bc_value: Expected value (usually 0 for u,v)
        
        Returns:
            Residual for boundary condition
        """
        u = y_pred[:, 0:1]
        v = y_pred[:, 1:2]
        
        residual_u = u - y_bc_value
        residual_v = v - y_bc_value
        
        return residual_u, residual_v
    
    def boundary_condition_inlet(self, y_pred, inlet_velocity, inlet_temp):
        """
        Inlet boundary condition: specified velocity and temperature.
        
        Args:
            y_pred: Network output [u, v, p, T]
            inlet_velocity: [u_inlet, v_inlet]
            inlet_temp: Temperature at inlet
        
        Returns:
            Residuals for inlet conditions
        """
        u = y_pred[:, 0:1]
        v = y_pred[:, 1:2]
        T = y_pred[:, 3:4]
        
        residual_u = u - inlet_velocity[0]
        residual_v = v - inlet_velocity[1]
        residual_T = T - inlet_temp
        
        return residual_u, residual_v, residual_T
    
    def boundary_condition_outlet(self, x, y_pred, outlet_pressure):
        """
        Outlet boundary condition: specified pressure.
        
        Args:
            x: Input coordinates
            y_pred: Network output [u, v, p, T]
            outlet_pressure: Pressure at outlet
        
        Returns:
            Residual for outlet pressure condition
        """
        p = y_pred[:, 2:3]
        residual_p = p - outlet_pressure
        return residual_p
    
    def boundary_condition_heat_source(self, y_pred, T_wall, h_conv=1000):
        """
        Heat transfer boundary condition at wall.
        q = h(T_wall - T_fluid) or fixed temperature.
        
        Args:
            y_pred: Network output [u, v, p, T]
            T_wall: Wall temperature (K)
            h_conv: Convection coefficient (W/m²·K)
        
        Returns:
            Temperature residual
        """
        T = y_pred[:, 3:4]
        residual_T = T - T_wall
        return residual_T
