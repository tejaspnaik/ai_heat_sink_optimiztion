"""
Physics-Informed Neural Network (PINN) architecture for fluid dynamics.
Optimized for Google Colab memory constraints.
"""
import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Fully-connected neural network for PINN.
    Optimized for 2D fluid dynamics with 4 outputs: [u, v, p, T]
    """
    
    def __init__(self, input_dim=2, hidden_layers=[128, 128, 128, 128], 
                 output_dim=4, activation='tanh'):
        """
        Initialize PINN architecture.
        
        Args:
            input_dim: Input dimension (2 for 2D: x, y)
            hidden_layers: List of hidden layer sizes
            output_dim: Output dimension (4 for u, v, p, T)
            activation: Activation function ('tanh', 'relu', 'elu')
        """
        super(PINN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        
        # Choose activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.ModuleList(layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier/Glorot initialization for better convergence."""
        for layer in self.network[:-1]:  # All but output
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # Output layer: small initialization
        nn.init.xavier_uniform_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, x):
        """
        Forward pass through network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Normalize input to [-1, 1] for better activation function performance
        x = x.clone()
        
        # Pass through hidden layers with activation
        for i, layer in enumerate(self.network[:-1]):
            x = layer(x)
            x = self.activation(x)
        
        # Output layer (no activation)
        x = self.network[-1](x)
        
        return x
    
    @staticmethod
    def create_small_model(input_dim=2, output_dim=4):
        """
        Create smallest model for Colab free tier (memory-constrained).
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        
        Returns:
            PINN model optimized for Colab
        """
        return PINN(
            input_dim=input_dim,
            hidden_layers=[64, 64, 64],  # 3 layers, 64 units each
            output_dim=output_dim,
            activation='tanh'
        )
    
    @staticmethod
    def create_medium_model(input_dim=2, output_dim=4):
        """
        Create medium model for standard GPUs.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        
        Returns:
            PINN model for standard training
        """
        return PINN(
            input_dim=input_dim,
            hidden_layers=[128, 128, 128, 128],  # 4 layers, 128 units each
            output_dim=output_dim,
            activation='tanh'
        )
    
    @staticmethod
    def create_large_model(input_dim=2, output_dim=4):
        """
        Create larger model for high-end hardware.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        
        Returns:
            PINN model for high-fidelity training
        """
        return PINN(
            input_dim=input_dim,
            hidden_layers=[256, 256, 256, 256],  # 4 layers, 256 units each
            output_dim=output_dim,
            activation='tanh'
        )
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """Get model architecture information."""
        total_params = self.count_parameters()
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'total_parameters': total_params,
            'activation': self.activation_name,
            'num_layers': len(self.network),
            'memory_mb': total_params * 4 / 1e6  # Approximate for float32
        }


class ResidualPINN(nn.Module):
    """
    PINN with residual connections for improved training stability.
    """
    
    def __init__(self, input_dim=2, hidden_dim=128, num_blocks=4, output_dim=4):
        """
        Initialize residual PINN.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension for all layers
            num_blocks: Number of residual blocks
            output_dim: Output dimension
        """
        super(ResidualPINN, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim, hidden_dim)
            for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()
    
    @staticmethod
    def _make_residual_block(in_dim, out_dim):
        """Create a residual block."""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x):
        """Forward pass with residual connections."""
        x = self.activation(self.input_projection(x))
        
        for block in self.residual_blocks:
            x = x + block(x)
            x = self.activation(x)
        
        x = self.output_layer(x)
        return x
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
