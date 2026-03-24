"""Evaluation module"""
from .metrics import compute_metrics, validate_against_cfd, compute_conservation_errors
from .visualization import PINNVisualizer

__all__ = [
    'compute_metrics',
    'validate_against_cfd',
    'compute_conservation_errors',
    'PINNVisualizer'
]
