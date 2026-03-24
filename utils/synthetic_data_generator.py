"""
Generate synthetic 2D fluid dynamics data for initial PINN training.
Useful for validation before running CFD simulations.
"""
import numpy as np


class SyntheticDataGenerator2D:
    """Generate synthetic 2D flow and temperature data."""
    
    @staticmethod
    def poiseuille_flow(x, y, U_max=1.0, H=1.0):
        """
        Poiseuille flow (channel flow between parallel plates).
        
        u(y) = U_max * 4 * y * (H - y) / H^2
        v = 0
        p = -mu * grad(u)  (linear pressure drop)
        
        Args:
            x: x-coordinates
            y: y-coordinates
            U_max: Maximum velocity
            H: Channel height
        
        Returns:
            (u, v, p) fields
        """
        u = U_max * 4 * y * (H - y) / (H ** 2)
        u[y <= 0] = 0
        u[y >= H] = 0
        
        v = np.zeros_like(y)
        p = np.zeros_like(x)
        
        return u, v, p
    
    @staticmethod
    def temperature_linear(x, y, T_in=300, T_out=350, L=1.0):
        """
        Linear temperature rise along channel (simplified energy equation).
        
        Args:
            x: x-coordinates
            y: y-coordinates
            T_in: Inlet temperature (K)
            T_out: Outlet temperature (K)
            L: Channel length
        
        Returns:
            T field
        """
        T = T_in + (T_out - T_in) * x / L
        return T
    
    @staticmethod
    def cylinder_wake_approximation(x, y, cylinder_x=0.5, cylinder_r=0.1):
        """
        Approximate velocity field around a cylinder (simplified).
        Uses potential flow + boundary layer approximation.
        
        Args:
            x: x-coordinates
            y: y-coordinates
            cylinder_x: x-position of cylinder
            cylinder_r: Cylinder radius
        
        Returns:
            (u, v, p) fields
        """
        # Distance from cylinder center
        dx = x - cylinder_x
        dy = y
        r = np.sqrt(dx**2 + dy**2)
        
        # Potential flow
        U_inf = 1.0
        theta = np.arctan2(dy, dx)
        
        u_r = U_inf * (1 - (cylinder_r / r) ** 2) * np.cos(theta)
        u_theta = -U_inf * (1 + (cylinder_r / r) ** 2) * np.sin(theta)
        
        u = u_r * np.cos(theta) - u_theta * np.sin(theta)
        v = u_r * np.sin(theta) + u_theta * np.cos(theta)
        
        # Inside cylinder: no flow
        inside = r < cylinder_r
        u[inside] = 0
        v[inside] = 0
        
        # Pressure coefficient
        p = 1 - (u**2 + v**2) / (U_inf**2)
        
        return u, v, p
    
    @staticmethod
    def heat_source_diffusion(x, y, source_x=0.5, source_y=0.5, 
                              source_strength=100, diffusivity=0.1):
        """
        Temperature field from point heat source with diffusion.
        
        Args:
            x: x-coordinates
            y: y-coordinates
            source_x, source_y: Heat source position
            source_strength: Heat power (W)
            diffusivity: Thermal diffusivity (m²/s)
        
        Returns:
            T field
        """
        dx = x - source_x
        dy = y - source_y
        r = np.sqrt(dx**2 + dy**2)
        
        # Gaussian temperature distribution
        T_ambient = 300
        T = T_ambient + (source_strength / (4 * np.pi * diffusivity)) * np.exp(-r**2 / (4 * diffusivity))
        
        return T
    
    @staticmethod
    def create_dataset_2d(nx=32, ny=32, dataset_type='poiseuille'):
        """
        Create synthetic 2D dataset.
        
        Args:
            nx, ny: Grid resolution
            dataset_type: 'poiseuille', 'cylinder', 'combined'
        
        Returns:
            Dictionary with coordinates and fields
        """
        # Create grid
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        xx, yy = np.meshgrid(x, y)
        
        if dataset_type == 'poiseuille':
            u, v, p = SyntheticDataGenerator2D.poiseuille_flow(xx, yy)
            T = SyntheticDataGenerator2D.temperature_linear(xx, yy)
        
        elif dataset_type == 'cylinder':
            u, v, p = SyntheticDataGenerator2D.cylinder_wake_approximation(xx, yy)
            T = SyntheticDataGenerator2D.heat_source_diffusion(xx, yy)
        
        elif dataset_type == 'combined':
            u, v, p = SyntheticDataGenerator2D.poiseuille_flow(xx, yy)
            T = SyntheticDataGenerator2D.temperature_linear(xx, yy)
            # Add some variation
            u += 0.01 * np.random.randn(*u.shape)
            T += 5 * np.random.randn(*T.shape)
        
        # Flatten and stack
        coords = np.stack([xx.flatten(), yy.flatten()], axis=1)
        u_flat = u.flatten()
        v_flat = v.flatten()
        p_flat = p.flatten()
        T_flat = T.flatten()
        
        fields = np.stack([u_flat, v_flat, p_flat, T_flat], axis=1)
        
        return {
            'coordinates': coords,
            'fields': fields,
            'u': u_flat,
            'v': v_flat,
            'p': p_flat,
            'T': T_flat,
            'grid_shape': (ny, nx)
        }
    
    @staticmethod
    def create_collocation_points(n_collocation=5000, domain_bounds=None):
        """
        Create collocation points for PDE residuals.
        
        Args:
            n_collocation: Number of collocation points
            domain_bounds: [(x_min, x_max), (y_min, y_max)]
        
        Returns:
            Collocation points (n_collocation, 2)
        """
        if domain_bounds is None:
            domain_bounds = [(0, 1), (0, 1)]
        
        x_min, x_max = domain_bounds[0]
        y_min, y_max = domain_bounds[1]
        
        x = np.random.uniform(x_min, x_max, n_collocation)
        y = np.random.uniform(y_min, y_max, n_collocation)
        
        return np.stack([x, y], axis=1)
    
    @staticmethod
    def create_boundary_conditions(domain_bounds, n_bc=1000):
        """
        Create boundary condition points.
        
        Args:
            domain_bounds: [(x_min, x_max), (y_min, y_max)]
            n_bc: Number of BC points per boundary
        
        Returns:
            Dictionary with BC points for each boundary
        """
        x_min, x_max = domain_bounds[0]
        y_min, y_max = domain_bounds[1]
        
        # Left boundary (inlet): x = x_min
        x_left = np.full(n_bc, x_min)
        y_left = np.linspace(y_min, y_max, n_bc)
        inlet_bc = np.stack([x_left, y_left], axis=1)
        
        # Right boundary (outlet): x = x_max
        x_right = np.full(n_bc, x_max)
        y_right = np.linspace(y_min, y_max, n_bc)
        outlet_bc = np.stack([x_right, y_right], axis=1)
        
        # Bottom boundary (wall): y = y_min
        x_bottom = np.linspace(x_min, x_max, n_bc)
        y_bottom = np.full(n_bc, y_min)
        bottom_bc = np.stack([x_bottom, y_bottom], axis=1)
        
        # Top boundary (wall): y = y_max
        x_top = np.linspace(x_min, x_max, n_bc)
        y_top = np.full(n_bc, y_max)
        top_bc = np.stack([x_top, y_top], axis=1)
        
        return {
            'inlet': inlet_bc,
            'outlet': outlet_bc,
            'wall_bottom': bottom_bc,
            'wall_top': top_bc
        }
