"""
3D Synthetic Data Generators for PINN Validation
Analytical solutions for 3D fluid flows and heat transfer
"""

import numpy as np
import torch
from typing import Tuple, Callable
from scipy.special import erfc, erfcinv


class SyntheticDataGenerator3D:
    """
    Generate 3D analytical solution data for PINN validation and training
    """
    
    @staticmethod
    def poiseuille_flow_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          channel_height: float = 1.0,
                          channel_width: float = 1.0,
                          pressure_gradient: float = -1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        3D Poiseuille flow between parallel plates
        Fully developed laminar flow in rectangular channel
        
        u(x,y,z) = (P/2μ) * (H² - y²) [analytical in y-direction only]
        
        Args:
            x: X-coordinates (n_samples,)
            y: Y-coordinates (n_samples,) - height direction
            z: Z-coordinates (n_samples,) - spanwise direction
            channel_height: Channel height (H)
            channel_width: Channel width (W)
            pressure_gradient: Pressure gradient (dP/dx)
            
        Returns:
            (u, v, w, p) - velocity components and pressure
        """
        # Normalize coordinates
        y_norm = np.clip(y / channel_height, 0, 1)
        z_norm = np.clip(z / channel_width, 0, 1)
        
        # Poiseuille velocity profile (parabolic in y)
        u = 6 * y_norm * (1 - y_norm)  # Maximum at center
        v = np.zeros_like(x)
        w = np.zeros_like(x)
        
        # Pressure field
        p = pressure_gradient * x
        
        return u, v, w, p
    
    @staticmethod
    def taylor_green_vortex_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                              t: np.ndarray = None,
                              nu: float = 0.01) -> Tuple[np.ndarray, ...]:
        """
        3D Taylor-Green vortex - analytical solution for inviscid Euler/viscous Navier-Stokes
        Classic test case for DNS validation and PINN training
        
        u = sin(x)*cos(y)*cos(z)*exp(-3*nu*t)
        v = -cos(x)*sin(y)*cos(z)*exp(-3*nu*t)
        w = 0
        p = (1/16)*(cos(2x) + cos(2y))*(2 + cos(2z))*exp(-6*nu*t)
        
        Args:
            x, y, z: Coordinates (n_samples,)
            t: Time (scalar or n_samples,), default 0
            nu: Kinematic viscosity
            
        Returns:
            (u, v, w, p, T) - velocity components, pressure, temperature
        """
        if t is None:
            t = np.zeros_like(x)
        
        # Ensure t is compatible
        if np.isscalar(t):
            t = np.full_like(x, t)
        
        # Decay factor for viscous case
        decay = np.exp(-3 * nu * t)
        decay_p = np.exp(-6 * nu * t)
        
        # Velocity
        u = np.sin(x) * np.cos(y) * np.cos(z) * decay
        v = -np.cos(x) * np.sin(y) * np.cos(z) * decay
        w = np.zeros_like(x)
        
        # Pressure
        p = (1/16) * (np.cos(2*x) + np.cos(2*y)) * (2 + np.cos(2*z)) * decay_p
        
        # Temperature: coupled to kinetic energy
        T = 0.5 * (u**2 + v**2) + 300  # Baseline 300K
        
        return u, v, w, p, T
    
    @staticmethod
    def cylinder_wake_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        cylinder_diameter: float = 0.1,
                        flow_velocity: float = 1.0,
                        re_d: float = 100.0) -> Tuple[np.ndarray, ...]:
        """
        3D cylinder wake flow - approximated using Gaussian profile
        Useful for testing separation and wake dynamics
        
        Args:
            x, y, z: Coordinates
            cylinder_diameter: Cylinder diameter (D)
            flow_velocity: Upstream flow velocity
            re_d: Reynolds number based on diameter
            
        Returns:
            (u, v, w, p, T)
        """
        # Kinematic viscosity from Re
        nu = (flow_velocity * cylinder_diameter) / re_d
        
        # Distance from cylinder (approximated)
        r = np.sqrt((y)**2 + (z)**2)
        r = np.maximum(r, cylinder_diameter/2)  # Avoid singularity
        
        # Gaussian wake deficit
        deficit = np.exp(-(r / (0.1 * np.abs(x)))**2)
        
        # Velocity field (axial-dominant with transverse components in wake)
        u = flow_velocity * (1 - 0.5 * deficit)
        
        # Transverse velocity (cross-flow in wake)
        u_theta = 0.1 * flow_velocity * deficit * z / (r + 1e-8)
        u_r = 0.1 * flow_velocity * deficit * y / (r + 1e-8)
        
        v = u_r
        w = u_theta
        
        # Pressure (wake suction)
        p = -0.5 * deficit * flow_velocity**2
        
        # Temperature
        T = 300 + 10 * deficit
        
        return u, v, w, p, T
    
    @staticmethod
    def heat_diffusion_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         t: np.ndarray = None,
                         alpha: float = 0.1,
                         T_initial: float = 100.0,
                         T_ambient: float = 300.0) -> Tuple[np.ndarray, ...]:
        """
        3D heat diffusion from point/line source
        Solution to ∂T/∂t = α∇²T
        
        Args:
            x, y, z: Coordinates
            t: Time (scalar or array)
            alpha: Thermal diffusivity
            T_initial: Initial temperature
            T_ambient: Ambient temperature
            
        Returns:
            (T, Tx, Ty, Tz) - temperature and spatial gradients
        """
        if t is None:
            t = np.ones_like(x) * 0.1
        
        if np.isscalar(t):
            t = np.full_like(x, t)
        
        # Distance from origin
        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.maximum(r, 1e-6)
        
        # Gaussian diffusion solution
        denominator = np.sqrt((4 * np.pi * alpha * t)**3)
        denominator = np.maximum(denominator, 1e-6)
        
        # Temperature field
        exp_arg = -(r**2) / (4 * alpha * t)
        exp_arg = np.minimum(exp_arg, 100)  # Avoid overflow
        
        T = T_ambient + (T_initial - T_ambient) * np.exp(exp_arg) / denominator
        
        # Gradients (analytical)
        denom_grad = 4 * alpha * t
        denom_grad = np.maximum(denom_grad, 1e-6)
        
        Tx = -(T_initial - T_ambient) * x / denom_grad * np.exp(exp_arg) / denominator
        Ty = -(T_initial - T_ambient) * y / denom_grad * np.exp(exp_arg) / denominator
        Tz = -(T_initial - T_ambient) * z / denom_grad * np.exp(exp_arg) / denominator
        
        return T, Tx, Ty, Tz
    
    @staticmethod
    def thermal_boundary_layer_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                 T_wall: float = 400.0,
                                 T_free: float = 300.0,
                                 u_free: float = 1.0,
                                 pr: float = 0.71,
                                 re_x: float = 10000.0) -> Tuple[np.ndarray, ...]:
        """
        3D thermal boundary layer over flat plate
        Blasius-Pohlhausen solution with temperature
        
        Args:
            x, y, z: Coordinates
            T_wall: Wall temperature
            T_free: Free-stream temperature
            u_free: Free-stream velocity
            pr: Prandtl number
            re_x: Reynolds number based on x
            
        Returns:
            (u, v, T, p)
        """
        # Thermal boundary layer thickness
        delta_t = 5.0 * np.sqrt(x / (u_free * re_x))
        delta_t = np.maximum(delta_t, 1e-6)
        
        # Normalized temperature profile
        eta = y / delta_t
        
        # Temperature profile (approximation)
        T_profile = T_wall + (T_free - T_wall) * (1 - np.exp(-eta**2))
        T = np.where(y > 0, T_profile, T_wall)
        
        # Velocity field (simplified)
        u = u_free * (1 - np.exp(-eta))
        v = -0.5 * np.sqrt(u_free / (x + 1e-6)) * y
        
        # Pressure (approximate)
        p = -0.5 * u**2
        
        return u, v, T, p
    
    @staticmethod
    def create_spatiotemporal_grid_3d(domain_3d: Tuple[Tuple[float, float], ...],
                                     time_range: Tuple[float, float],
                                     n_spatial: int = 20,
                                     n_time: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create 3D spatiotemporal grid for unsteady problems
        
        Args:
            domain_3d: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            time_range: (t_min, t_max)
            n_spatial: Points per spatial dimension
            n_time: Time steps
            
        Returns:
            (x_grid, y_grid, z_grid, t_grid) - 1D arrays for grid
        """
        x_min, x_max = domain_3d[0]
        y_min, y_max = domain_3d[1]
        z_min, z_max = domain_3d[2]
        t_min, t_max = time_range
        
        x = np.linspace(x_min, x_max, n_spatial)
        y = np.linspace(y_min, y_max, n_spatial)
        z = np.linspace(z_min, z_max, n_spatial)
        t = np.linspace(t_min, t_max, n_time)
        
        return x, y, z, t
    
    @staticmethod
    def generate_training_data_3d(method: str = 'taylor_green',
                                 domain: Tuple = ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
                                 n_points: int = 10000,
                                 **kwargs) -> Dict[str, np.ndarray]:
        """
        Generate complete training dataset in 3D
        
        Args:
            method: 'taylor_green', 'poiseuille', 'cylinder_wake', 'heat_diffusion'
            domain: 3D domain bounds
            n_points: Number of sample points
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Dictionary with 'coordinates' and 'fields'
        """
        # Generate random points in domain
        x = np.random.uniform(domain[0][0], domain[0][1], n_points)
        y = np.random.uniform(domain[1][0], domain[1][1], n_points)
        z = np.random.uniform(domain[2][0], domain[2][1], n_points)
        
        coordinates = np.column_stack([x, y, z])
        
        # Generate fields based on method
        if method == 'taylor_green':
            t = kwargs.get('t', 0.0)
            u, v, w, p, T = SyntheticDataGenerator3D.taylor_green_vortex_3d(x, y, z, t)
            fields = np.column_stack([u, v, w, p, T])
        
        elif method == 'poiseuille':
            u, v, w, p = SyntheticDataGenerator3D.poiseuille_flow_3d(x, y, z)
            T = np.full_like(x, 300.0)
            fields = np.column_stack([u, v, w, p, T])
        
        elif method == 'cylinder_wake':
            u, v, w, p, T = SyntheticDataGenerator3D.cylinder_wake_3d(x, y, z)
            fields = np.column_stack([u, v, w, p, T])
        
        elif method == 'heat_diffusion':
            t = kwargs.get('t', 0.1)
            T, Tx, Ty, Tz = SyntheticDataGenerator3D.heat_diffusion_3d(x, y, z, t)
            u, v, w = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
            p = np.zeros_like(x)
            fields = np.column_stack([u, v, w, p, T])
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'coordinates': coordinates,
            'fields': fields,
            'x': x, 'y': y, 'z': z
        }


class AnalyticalSolutionProvider3D:
    """
    Provides analytical solutions and residuals for PINN testing
    """
    
    def __init__(self, solution_type: str = 'taylor_green'):
        self.solution_type = solution_type
    
    def compute_residuals(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         derivatives: Dict[str, np.ndarray],
                         params: Dict = None) -> Dict[str, np.ndarray]:
        """
        Compute PDE residuals from analytical solution gradients
        
        Args:
            x, y, z: Coordinates
            derivatives: Dictionary with derivative values (du_dx, du_dy, etc.)
            params: Parameters (mu, rho, etc.)
            
        Returns:
            Dictionary with residuals for each equation
        """
        if params is None:
            params = {'mu': 0.001, 'rho': 1.0, 'k_f': 0.5}
        
        residuals = {}
        
        # Continuity: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
        residuals['continuity'] = (
            derivatives.get('du_dx', 0) +
            derivatives.get('dv_dy', 0) +
            derivatives.get('dw_dz', 0)
        )
        
        # Momentum equations (simplified)
        residuals['momentum_x'] = derivatives.get('du_dx', 0)
        residuals['momentum_y'] = derivatives.get('dv_dy', 0)
        residuals['momentum_z'] = derivatives.get('dw_dz', 0)
        
        # Energy equation (simplified)
        residuals['energy'] = derivatives.get('dT_dt', 0) if 'dT_dt' in derivatives else np.zeros_like(x)
        
        return residuals
    
    def validate_solution(self, pinn_solution: np.ndarray,
                         reference_solution: np.ndarray) -> Dict[str, float]:
        """
        Validate PINN solution against analytical reference
        
        Args:
            pinn_solution: PINN predictions (N, 5) for [u, v, w, p, T]
            reference_solution: Analytical solution (N, 5)
            
        Returns:
            Dictionary with error metrics
        """
        errors = np.abs(pinn_solution - reference_solution)
        
        return {
            'mae': np.mean(errors),
            'rmse': np.sqrt(np.mean(errors**2)),
            'max_error': np.max(errors),
            'velocity_error': np.mean(errors[:, :3]),
            'pressure_error': np.mean(errors[:, 3]),
            'temperature_error': np.mean(errors[:, 4]),
        }


from typing import Dict
