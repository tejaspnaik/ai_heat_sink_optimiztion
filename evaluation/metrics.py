"""
Evaluation and validation utilities.
"""
import numpy as np
import torch


def compute_metrics(y_true, y_pred):
    """
    Compute error metrics.
    
    Args:
        y_true: Ground truth (N, D)
        y_pred: Predictions (N, D)
    
    Returns:
        Dictionary with metrics
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # Relative error
    rel_error = np.linalg.norm(y_true - y_pred) / (np.linalg.norm(y_true) + 1e-8)
    
    # Component-wise metrics
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'relative_error': rel_error
    }
    
    # Per-component metrics (u, v, p, T)
    component_names = ['u', 'v', 'p', 'T']
    for i in range(min(y_pred.shape[1], len(component_names))):
        metrics[f'mae_{component_names[i]}'] = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
        metrics[f'rmse_{component_names[i]}'] = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2))
    
    return metrics


def validate_against_cfd(pinn_pred, cfd_data):
    """
    Validate PINN predictions against CFD data.
    
    Args:
        pinn_pred: PINN predictions (N, 4)
        cfd_data: CFD ground truth (N, 4)
    
    Returns:
        Validation metrics
    """
    return compute_metrics(cfd_data, pinn_pred)


def compute_physics_residuals(x, y, physics_constraints):
    """
    Compute physics residuals at validation points.
    
    Args:
        x: Input points (N, 2)
        y: Model outputs (N, 4)
        physics_constraints: PhysicsConstraints instance
    
    Returns:
        Residual statistics
    """
    residuals = physics_constraints.navier_stokes_2d(x, y)
    
    residual_stats = {}
    residual_names = ['u_momentum', 'v_momentum', 'continuity', 'energy']
    
    for i, (res, name) in enumerate(zip(residuals, residual_names)):
        residual_stats[f'{name}_mean'] = np.mean(np.abs(res.detach().cpu().numpy()))
        residual_stats[f'{name}_max'] = np.max(np.abs(res.detach().cpu().numpy()))
        residual_stats[f'{name}_std'] = np.std(res.detach().cpu().numpy())
    
    return residual_stats


def compute_conservation_errors(y_pred):
    """
    Check conservation laws (mass, energy).
    
    Args:
        y_pred: Model predictions (N, 4)
    
    Returns:
        Conservation error metrics
    """
    u = y_pred[:, 0]
    v = y_pred[:, 1]
    T = y_pred[:, 2]
    
    # Approximate divergence (simplified)
    # In real implementation, use automatic differentiation
    
    return {
        'mass_conservation_approx': 0.0,  # Placeholder
        'energy_conservation_approx': 0.0  # Placeholder
    }


def compare_designs(design_history):
    """
    Analyze design optimization history.
    
    Args:
        design_history: List of design points and objectives
    
    Returns:
        Summary statistics
    """
    if len(design_history) == 0:
        return {}
    
    objectives = np.array([item['value'] for item in design_history])
    
    return {
        'initial_objective': objectives[0],
        'final_objective': objectives[-1],
        'improvement': (objectives[0] - objectives[-1]) / objectives[0] * 100,
        'best_objective': np.min(objectives),
        'mean_objective': np.mean(objectives),
        'std_objective': np.std(objectives)
    }


def compute_thermal_performance(T_field, Q, cp=4186, rho=1000, volume=1.0):
    """
    Compute thermal performance metrics.
    
    Args:
        T_field: Temperature field
        Q: Heat transfer rate (W)
        cp: Specific heat capacity
        rho: Density
        volume: Fluid volume
    
    Returns:
        Thermal metrics
    """
    T_mean = np.mean(T_field)
    T_max = np.max(T_field)
    T_min = np.min(T_field)
    T_std = np.std(T_field)
    
    # Heat capacity
    C = rho * volume * cp
    
    return {
        'T_mean': T_mean,
        'T_max': T_max,
        'T_min': T_min,
        'T_std': T_std,
        'temperature_uniformity': 1.0 / (1.0 + T_std / T_mean),  # 0-1 scale
        'heat_capacity': C
    }


def compute_flow_statistics(u_field, v_field):
    """
    Compute flow field statistics.
    
    Args:
        u_field: x-velocity field
        v_field: y-velocity field
    
    Returns:
        Flow statistics
    """
    velocity_magnitude = np.sqrt(u_field**2 + v_field**2)
    
    return {
        'u_mean': np.mean(u_field),
        'v_mean': np.mean(v_field),
        'u_max': np.max(u_field),
        'v_max': np.max(v_field),
        'velocity_mean': np.mean(velocity_magnitude),
        'velocity_max': np.max(velocity_magnitude),
        'velocity_std': np.std(velocity_magnitude)
    }
