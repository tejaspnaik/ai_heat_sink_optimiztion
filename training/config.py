"""
Configuration defaults for PINN training and optimization.
"""
import yaml
from pathlib import Path


DEFAULT_CONFIG = {
    'training': {
        'epochs': 5000,
        'batch_size': 256,
        'learning_rate': 0.001,
        'optimizer': 'adam',  # 'adam' or 'sgd'
        'weight_decay': 0.0,
        'seed': 42,
        'checkpoint_interval': 500,
        'validate_interval': 500,
        'early_stopping_patience': 20,
    },
    
    'model': {
        'type': 'small',  # 'small', 'medium', 'large'
        'input_dim': 2,
        'output_dim': 4,
        'activation': 'tanh',
        'hidden_layers': [128, 128, 128, 128],  # Will be overridden by type
    },
    
    'data': {
        'collocation_points': 5000,
        'boundary_points': 1000,
        'dataset_type': 'synthetic',  # 'synthetic' or 'cfd'
        'train_split': 0.7,
        'val_split': 0.15,
        'normalize': True,
        'normalization_method': 'standardization',  # 'standardization' or 'minmax'
    },
    
    'physics': {
        'rho': 1000.0,  # Density (kg/m³)
        'mu': 0.001,    # Dynamic viscosity (Pa·s)
        'k_thermal': 0.6,  # Thermal conductivity (W/m·K)
        'cp': 4186.0,   # Specific heat capacity (J/kg·K)
        'domain_bounds': [[0, 1], [0, 1]],  # [(x_min, x_max), (y_min, y_max)]
    },
    
    'loss': {
        'weight_pde': 1.0,
        'weight_bc': 10.0,
        'weight_data': 1.0,
        'adaptive_weighting': False,
    },
    
    'optimization': {
        'algorithm': 'bayesian',  # 'bayesian', 'ga', 'pso'
        'n_iterations': 1000,
        'n_initial_samples': 10,
    },
    
    'paths': {
        'data_dir': 'data',
        'checkpoint_dir': 'checkpoints',
        'output_dir': 'output',
    },
    
    'device': 'cuda',  # 'cuda' or 'cpu'
}


class Config:
    """Configuration class for PINN training."""
    
    def __init__(self, config_dict=None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Dictionary with configuration values
        """
        self.config = DEFAULT_CONFIG.copy()
        
        if config_dict is not None:
            self._merge_config(self.config, config_dict)
    
    @staticmethod
    def _merge_config(base, update):
        """Recursively merge configuration dicts."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                Config._merge_config(base[key], value)
            else:
                base[key] = value
    
    def load_yaml(self, filepath):
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self._merge_config(self.config, config_dict)
    
    def save_yaml(self, filepath):
        """Save configuration to YAML file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key, default=None):
        """Get configuration value by key (dot notation supported)."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key, value):
        """Set configuration value by key (dot notation supported)."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def __getitem__(self, key):
        """Get configuration section."""
        return self.config.get(key, {})
    
    def __repr__(self):
        """String representation."""
        return str(self.config)


def get_default_config():
    """Get default configuration."""
    return Config()


def create_config_file(filepath):
    """Create default configuration file."""
    config = Config()
    config.save_yaml(filepath)
    print(f"Configuration file created: {filepath}")
