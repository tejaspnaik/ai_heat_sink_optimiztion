"""PINN Cooling Project - Physics-Informed Neural Networks for Heat Sink Optimization"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .models import PINN, PhysicsConstraints
from .training import PINNTrainer
from .optimization import HeatSinkOptimizer, BayesianOptimizer
from .evaluation import compute_metrics

__all__ = [
    'PINN',
    'PhysicsConstraints',
    'PINNTrainer',
    'HeatSinkOptimizer',
    'BayesianOptimizer',
    'compute_metrics'
]
