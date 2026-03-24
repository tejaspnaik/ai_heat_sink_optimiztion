"""
Turbulence Models for PINN - k-ε and k-ω RANS models
Two-equation closure models for eddy viscosity computation
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import math


class TurbulenceModel:
    """Base class for turbulence models"""
    
    def __init__(self, rho: float = 1.0, mu: float = 0.001):
        self.rho = rho
        self.mu = mu
        self.nu = mu / rho  # Kinematic viscosity
    
    def compute_nu_t(self, strain_rate: torch.Tensor, k: torch.Tensor, 
                     dissipation: torch.Tensor) -> torch.Tensor:
        """Compute turbulent viscosity (to be overridden)"""
        raise NotImplementedError


class KepsilonModel(TurbulenceModel):
    """
    Standard k-ε turbulence model
    Equations:
        Dk/Dt = Production - ε + ∇·[(ν + σ_k*ν_t)∇k]
        Dε/Dt = (C_1ε*P - C_2ε*ε/k) + ∇·[(ν + σ_ε*ν_t)∇ε]
    
    where:
        P = ν_t*|S|² (production term)
        |S| = √(2*S_ij*S_ij) (strain rate magnitude)
        ν_t = C_μ * k²/ε (eddy viscosity)
    """
    
    # Model coefficients
    C_mu = 0.09
    C_1eps = 1.44
    C_2eps = 1.92
    sigma_k = 1.0
    sigma_eps = 1.3
    
    def __init__(self, rho: float = 1.0, mu: float = 0.001):
        super().__init__(rho, mu)
    
    def compute_nu_t(self, k: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """
        Compute turbulent kinematic viscosity
        ν_t = C_μ * k²/ε
        
        Args:
            k: Turbulent kinetic energy [batch, 1]
            epsilon: Turbulent dissipation rate [batch, 1]
            
        Returns:
            Turbulent viscosity [batch, 1]
        """
        # Avoid division by zero
        epsilon_safe = torch.clamp(epsilon, min=1e-8)
        nu_t = self.C_mu * (k ** 2) / epsilon_safe
        return nu_t
    
    def production_term(self, strain_rate: torch.Tensor, nu_t: torch.Tensor) -> torch.Tensor:
        """
        Production term: P = ν_t * 2|S|²
        
        Args:
            strain_rate: Strain rate tensor components [batch, 6] 
                        (S_xx, S_yy, S_zz, S_xy, S_xz, S_yz)
            nu_t: Turbulent viscosity [batch, 1]
            
        Returns:
            Production term [batch, 1]
        """
        # 2|S|² = 2 * S_ij * S_ij
        S_norm_sq = 2 * torch.sum(strain_rate ** 2, dim=1, keepdim=True)
        P = nu_t * S_norm_sq
        return P
    
    def k_equation(self, xy: torch.Tensor, k: torch.Tensor, epsilon: torch.Tensor,
                   strain_rate: torch.Tensor, nu_t: torch.Tensor) -> torch.Tensor:
        """
        Turbulent kinetic energy equation residual
        
        Args:
            xy: Spatial coordinates [batch, 2 or 3]
            k: Turbulent kinetic energy [batch, 1]
            epsilon: Dissipation rate [batch, 1]
            strain_rate: Strain rate tensor [batch, 6]
            nu_t: Turbulent viscosity [batch, 1]
            
        Returns:
            Residual of k equation
        """
        xy.requires_grad_(True)
        
        # Compute gradients of k
        dk_dxy = torch.autograd.grad(k.sum(), xy, create_graph=True, retain_graph=True)[0]
        d2k_dxy2 = 0
        
        for i in range(xy.shape[1]):
            d2k = torch.autograd.grad(dk_dxy[:, i].sum(), xy, create_graph=True)[0][:, i:i+1]
            d2k_dxy2 += d2k
        
        # Production
        P = self.production_term(strain_rate, nu_t)
        
        # Dissipation
        epsilon_safe = torch.clamp(epsilon, min=1e-8)
        D = epsilon_safe
        
        # Diffusion
        nu_eft_k = self.nu + self.sigma_k * nu_t
        diffusion = nu_eft_k * d2k_dxy2
        
        # Residual (steady state, no convection for now)
        residual = P - D + diffusion
        
        return residual
    
    def epsilon_equation(self, xy: torch.Tensor, k: torch.Tensor, epsilon: torch.Tensor,
                        strain_rate: torch.Tensor, nu_t: torch.Tensor) -> torch.Tensor:
        """
        Turbulent dissipation rate equation residual
        
        Args:
            xy: Spatial coordinates [batch, 2 or 3]
            k: Turbulent kinetic energy [batch, 1]
            epsilon: Dissipation rate [batch, 1]
            strain_rate: Strain rate tensor [batch, 6]
            nu_t: Turbulent viscosity [batch, 1]
            
        Returns:
            Residual of ε equation
        """
        xy.requires_grad_(True)
        k.requires_grad_(True)
        
        # Compute gradients of epsilon
        deps_dxy = torch.autograd.grad(epsilon.sum(), xy, create_graph=True, retain_graph=True)[0]
        d2eps_dxy2 = 0
        
        for i in range(xy.shape[1]):
            d2eps = torch.autograd.grad(deps_dxy[:, i].sum(), xy, create_graph=True)[0][:, i:i+1]
            d2eps_dxy2 += d2eps
        
        # Production
        P = self.production_term(strain_rate, nu_t)
        
        # Dissipation and production terms in ε equation
        k_safe = torch.clamp(k, min=1e-8)
        epsilon_safe = torch.clamp(epsilon, min=1e-8)
        
        source = self.C_1eps * P * epsilon_safe / k_safe - self.C_2eps * (epsilon_safe ** 2) / k_safe
        
        # Diffusion
        nu_eft_eps = self.nu + self.sigma_eps * nu_t
        diffusion = nu_eft_eps * d2eps_dxy2
        
        # Residual
        residual = source + diffusion
        
        return residual


class KomegaModel(TurbulenceModel):
    """
    Standard k-ω turbulence model (Wilcox)
    Equations:
        Dk/Dt = Production - β*k*ω + ∇·[(ν + σ_k*ν_t)∇k]
        Dω/Dt = (α*ω/k*Production - β*ω²) + ∇·[(ν + σ_ω*ν_t)∇ω]
    
    where:
        ν_t = k/ω (eddy viscosity)
        ω = ε/(β*k) (specific dissipation rate)
    """
    
    # Model coefficients (Wilcox 1998)
    alpha = 5.0 / 9.0
    beta = 0.075
    sigma_k = 2.0
    sigma_omega = 2.0
    
    def __init__(self, rho: float = 1.0, mu: float = 0.001):
        super().__init__(rho, mu)
    
    def compute_nu_t(self, k: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """
        Compute turbulent kinematic viscosity from k and ω
        ν_t = k/ω
        
        Args:
            k: Turbulent kinetic energy [batch, 1]
            omega: Specific dissipation rate [batch, 1]
            
        Returns:
            Turbulent viscosity [batch, 1]
        """
        omega_safe = torch.clamp(omega, min=1e-8)
        nu_t = k / omega_safe
        return nu_t
    
    def production_term(self, strain_rate: torch.Tensor, nu_t: torch.Tensor) -> torch.Tensor:
        """
        Production term in k-ω model
        P = ν_t * 2|S|²
        """
        S_norm_sq = 2 * torch.sum(strain_rate ** 2, dim=1, keepdim=True)
        P = nu_t * S_norm_sq
        return P
    
    def k_equation(self, xy: torch.Tensor, k: torch.Tensor, omega: torch.Tensor,
                   strain_rate: torch.Tensor, nu_t: torch.Tensor) -> torch.Tensor:
        """
        Turbulent kinetic energy equation for k-ω model
        
        Args:
            xy: Spatial coordinates [batch, 2 or 3]
            k: Turbulent kinetic energy [batch, 1]
            omega: Specific dissipation rate [batch, 1]
            strain_rate: Strain rate tensor [batch, 6]
            nu_t: Turbulent viscosity [batch, 1]
            
        Returns:
            Residual of k equation
        """
        xy.requires_grad_(True)
        
        # Gradients
        dk_dxy = torch.autograd.grad(k.sum(), xy, create_graph=True, retain_graph=True)[0]
        d2k_dxy2 = 0
        
        for i in range(xy.shape[1]):
            d2k = torch.autograd.grad(dk_dxy[:, i].sum(), xy, create_graph=True)[0][:, i:i+1]
            d2k_dxy2 += d2k
        
        # Production
        P = self.production_term(strain_rate, nu_t)
        
        # Dissipation: β*k*ω
        omega_safe = torch.clamp(omega, min=1e-8)
        dissipation = self.beta * k * omega_safe
        
        # Diffusion
        nu_eft_k = self.nu + self.sigma_k * nu_t
        diffusion = nu_eft_k * d2k_dxy2
        
        # Residual
        residual = P - dissipation + diffusion
        
        return residual
    
    def omega_equation(self, xy: torch.Tensor, k: torch.Tensor, omega: torch.Tensor,
                      strain_rate: torch.Tensor, nu_t: torch.Tensor) -> torch.Tensor:
        """
        Specific dissipation rate equation for k-ω model
        
        Args:
            xy: Spatial coordinates [batch, 2 or 3]
            k: Turbulent kinetic energy [batch, 1]
            omega: Specific dissipation rate [batch, 1]
            strain_rate: Strain rate tensor [batch, 6]
            nu_t: Turbulent viscosity [batch, 1]
            
        Returns:
            Residual of ω equation
        """
        xy.requires_grad_(True)
        
        # Gradients
        dom_dxy = torch.autograd.grad(omega.sum(), xy, create_graph=True, retain_graph=True)[0]
        d2om_dxy2 = 0
        
        for i in range(xy.shape[1]):
            d2om = torch.autograd.grad(dom_dxy[:, i].sum(), xy, create_graph=True)[0][:, i:i+1]
            d2om_dxy2 += d2om
        
        # Production effect on ω
        P = self.production_term(strain_rate, nu_t)
        
        k_safe = torch.clamp(k, min=1e-8)
        omega_safe = torch.clamp(omega, min=1e-8)
        
        source = self.alpha * (omega_safe / k_safe) * P - self.beta * (omega_safe ** 2)
        
        # Diffusion
        nu_eft_omega = self.nu + self.sigma_omega * nu_t
        diffusion = nu_eft_omega * d2om_dxy2
        
        # Residual
        residual = source + diffusion
        
        return residual


class HybridTurbulenceModel(TurbulenceModel):
    """
    Blended k-ε and k-ω model
    Uses k-ω near wall and k-ε in outer flow region
    Blend function: f = tanh(r_ω^4) where r_ω based on distance from boundary
    """
    
    def __init__(self, rho: float = 1.0, mu: float = 0.001):
        super().__init__(rho, mu)
        self.k_eps = KepsilonModel(rho, mu)
        self.k_omega = KomegaModel(rho, mu)
    
    def blend_function(self, distance_from_wall: torch.Tensor, C_blend: float = 1.0) -> torch.Tensor:
        """
        Blend function based on distance from wall
        f = tanh((d/C)^4) where d is distance from wall
        """
        arg = torch.clamp(distance_from_wall, min=1e-8) / C_blend
        blend = torch.tanh(arg ** 4)
        return blend  # 0 near wall (k-ω), 1 far from wall (k-ε)
    
    def compute_nu_t(self, k: torch.Tensor, epsilon: torch.Tensor, 
                    omega: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """
        Compute blended turbulent viscosity
        
        Args:
            k: Turbulent kinetic energy
            epsilon: Dissipation rate (for k-ε)
            omega: Specific dissipation (for k-ω)
            blend: Blending function [0, 1]
            
        Returns:
            Blended turbulent viscosity
        """
        nu_t_omega = self.k_omega.compute_nu_t(k, omega)
        nu_t_eps = self.k_eps.compute_nu_t(k, epsilon)
        
        nu_t = (1 - blend) * nu_t_omega + blend * nu_t_eps
        return nu_t


class SpalartAllmarasModel(TurbulenceModel):
    """
    Spalart-Allmaras one-equation turbulence model
    Simplified model with single variable ν̃ (modified eddy viscosity)
    Ideal for aerospace applications and DNS-RANS coupling
    
    Equation: Dν̃/Dt = c_b1*S*ν̃ + (1/σ)*∇·[(ν+ν̃)∇ν̃] - c_w1*f_w*(ν̃/d)²
    """
    
    c_b1 = 0.1355
    c_b2 = 0.622
    sigma = 2.0 / 3.0
    c_w1 = c_b1 / 0.41**2 + (1 + c_b2) / sigma
    c_w2 = 0.3
    c_w3 = 2.0
    
    def __init__(self, rho: float = 1.0, mu: float = 0.001):
        super().__init__(rho, mu)
    
    def compute_nu_t(self, nu_tilde: torch.Tensor, chi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute turbulent viscosity from ν̃
        ν_t = ν̃ * f_v1
        where f_v1 = χ³/(χ³ + c_v1³), χ = ν̃/ν
        """
        if chi is None:
            chi = torch.clamp(nu_tilde / self.nu, min=0, max=100)  # Avoid overflow
        
        c_v1 = 7.1
        chi_3 = chi ** 3
        f_v1 = chi_3 / (chi_3 + c_v1 ** 3)
        
        nu_t = nu_tilde * f_v1
        return nu_t
    
    def spalart_allmaras_equation(self, xy: torch.Tensor, nu_tilde: torch.Tensor,
                                  strain_rate_mag: torch.Tensor,
                                  distance_from_wall: torch.Tensor) -> torch.Tensor:
        """
        Spalart-Allmaras equation residual
        
        Args:
            xy: Spatial coordinates
            nu_tilde: Modified eddy viscosity
            strain_rate_mag: Magnitude of strain rate |S|
            distance_from_wall: Distance from boundary (d)
            
        Returns:
            Residual
        """
        xy.requires_grad_(True)
        
        # Compute Laplacian of ν̃
        dnu_dxy = torch.autograd.grad(nu_tilde.sum(), xy, create_graph=True, retain_graph=True)[0]
        d2nu_dxy2 = 0
        
        for i in range(xy.shape[1]):
            d2nu = torch.autograd.grad(dnu_dxy[:, i].sum(), xy, create_graph=True)[0][:, i:i+1]
            d2nu_dxy2 += d2nu
        
        # Chi = ν̃/ν
        chi = torch.clamp(nu_tilde / self.nu, min=0, max=100)
        
        # f_v2
        c_v2 = 0.7
        f_v2 = 1 - chi / (1 + chi * (self.c_b2 / self.sigma)**0.5)
        
        # f_w
        r = torch.clamp(nu_tilde / (strain_rate_mag * distance_from_wall**2 + 1e-8), min=0, max=10)
        g = r + self.c_w2 * (r**6 - r)
        f_w = g * ((1 + self.c_w3**6) / (g**6 + self.c_w3**6))**(1/6)
        
        # Source and sink terms
        production = self.c_b1 * strain_rate_mag * nu_tilde
        destruction = self.c_w1 * f_w * (nu_tilde / (distance_from_wall**2 + 1e-8))**2
        diffusion = (self.nu + nu_tilde) / self.sigma * d2nu_dxy2
        
        # Residual (steady state)
        residual = production - destruction + diffusion
        
        return residual
