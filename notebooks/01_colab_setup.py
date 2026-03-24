# Google Colab Setup for PINN Liquid Cooling Heat Sink Optimization
# Run this notebook cell by cell to initialize the environment

"""
CELL 1: Environment Setup and Package Installation
"""

# Check if running in Colab
try:
    from google.colab import drive
    IN_COLAB = True
    print("Running in Google Colab")
except:
    IN_COLAB = False
    print("Running locally")

# Mount Google Drive (if in Colab)
if IN_COLAB:
    drive.mount('/content/drive')
    print("Google Drive mounted")

# Install required packages
import subprocess
import sys

packages = [
    'torch',  # Will install via pip with CUDA support
    'numpy',
    'scipy',
    'matplotlib',
    'plotly',
    'pyyaml',
    'tqdm'
]

print("Installing packages...")
for package in packages:
    try:
        __import__(package)
        print(f"✓ {package} already installed")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("\nPackages installed successfully!")

"""
CELL 2: Setup Project Structure and Imports
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# Setup paths
if IN_COLAB:
    PROJECT_ROOT = '/content/drive/My Drive/pinn-cooling-project'
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
else:
    PROJECT_ROOT = Path.cwd()
    DATA_DIR = PROJECT_ROOT / 'data'
    CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
    OUTPUT_DIR = PROJECT_ROOT / 'output'

# Create directories
for directory in [DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"✓ Directory ready: {directory}")

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n✓ Using device: {device}")

if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

"""
CELL 3: Define PINN Neural Network Architecture
"""

import torch.nn as nn

class PINN(nn.Module):
    """Fully-connected neural network for PINN (optimized for Colab)."""
    
    def __init__(self, input_dim=2, hidden_layers=[128, 128, 128], 
                 output_dim=4, activation='tanh'):
        super(PINN, self).__init__()
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.ModuleList(layers)
        
        # Initialize weights
        for layer in self.network[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.network[:-1]):
            x = layer(x)
            x = self.activation(x)
        x = self.network[-1](x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Create model
model = PINN(input_dim=2, hidden_layers=[64, 64, 64], output_dim=4).to(device)
print(f"✓ Model created")
print(f"  Total parameters: {model.count_parameters():,}")

"""
CELL 4: Synthetic Data Generation for Testing
"""

def create_synthetic_dataset(nx=32, ny=32):
    """Create synthetic Poiseuille flow for testing."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    
    # Poiseuille flow
    H = 1.0
    U_max = 1.0
    u = U_max * 4 * yy * (H - yy) / (H ** 2)
    u[yy <= 0] = 0
    u[yy >= H] = 0
    
    v = np.zeros_like(u)
    p = np.zeros_like(u)
    
    # Temperature field
    T = 300 + 50 * xx
    
    coords = np.stack([xx.flatten(), yy.flatten()], axis=1)
    fields = np.stack([u.flatten(), v.flatten(), p.flatten(), T.flatten()], axis=1)
    
    return coords.astype(np.float32), fields.astype(np.float32)

# Generate synthetic data
x_data, y_data = create_synthetic_dataset(32, 32)
print(f"✓ Synthetic data generated")
print(f"  Coordinate shape: {x_data.shape}")
print(f"  Field shape: {y_data.shape}")

# Convert to tensors
x_tensor = torch.from_numpy(x_data).to(device)
y_tensor = torch.from_numpy(y_data).to(device)

"""
CELL 5: Simple Loss Function and Training Loop
"""

class SimplePhysicsLoss(nn.Module):
    """Simplified physics loss for testing."""
    
    def __init__(self, weight_pde=1.0, weight_bc=10.0, weight_data=1.0):
        super().__init__()
        self.weight_pde = weight_pde
        self.weight_bc = weight_bc
        self.weight_data = weight_data
    
    def forward(self, y_pred, y_true):
        # Data loss
        data_loss = torch.mean((y_pred - y_true) ** 2)
        
        # BC loss (simplified: enforce boundary values)
        boundary_loss = torch.mean(y_pred[0, :] ** 2)
        
        return (
            self.weight_data * data_loss + 
            self.weight_bc * boundary_loss
        )

loss_fn = SimplePhysicsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

print("✓ Loss function and optimizer initialized")

"""
CELL 6: Training with Checkpoint Support (For Colab)
"""

def train_step(model, x_batch, y_batch, optimizer, loss_fn, device):
    """Single training step."""
    x_batch = x_batch.to(device).requires_grad_(True)
    y_batch = y_batch.to(device)
    
    optimizer.zero_grad()
    
    y_pred = model(x_batch)
    loss = loss_fn(y_pred, y_batch)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_epoch(model, x_data, y_data, optimizer, loss_fn, batch_size=128, device='cpu'):
    """Train for one epoch."""
    n_samples = len(x_data)
    indices = np.random.permutation(n_samples)
    
    total_loss = 0
    n_batches = 0
    
    for i in range(0, n_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        x_batch = torch.from_numpy(x_data[batch_idx]).to(device)
        y_batch = torch.from_numpy(y_data[batch_idx]).to(device)
        
        loss = train_step(model, x_batch, y_batch, optimizer, loss_fn, device)
        total_loss += loss
        n_batches += 1
    
    return total_loss / n_batches

# Training loop with checkpoints
num_epochs = 100  # For testing; use 5000+ for real training
checkpoint_interval = 20

print(f"Starting training for {num_epochs} epochs...")
losses = []

for epoch in range(num_epochs):
    loss = train_epoch(model, x_data, y_data, optimizer, loss_fn, batch_size=64, device=device)
    losses.append(loss)
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}")
    
    # Save checkpoint
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': loss
        }, checkpoint_path)

print(f"✓ Training complete!")

"""
CELL 7: Evaluate and Visualize Results
"""

# Make predictions
model.eval()
with torch.no_grad():
    y_pred = model(x_tensor).cpu().numpy()
    y_true = y_data

# Compute metrics
mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

print(f"Evaluation Metrics:")
print(f"  MAE:  {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")

# Plot training loss
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'), dpi=150, bbox_inches='tight')
print(f"✓ Training loss plot saved")

# Plot predictions vs. ground truth
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

component_names = ['u', 'v', 'p', 'T']
for i in range(2):
    ax = axes[i]
    ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=10)
    
    min_val = min(y_true[:, i].min(), y_pred[:, i].min())
    max_val = max(y_true[:, i].max(), y_pred[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax.set_xlabel(f'Ground Truth ({component_names[i]})')
    ax.set_ylabel(f'PINN Prediction ({component_names[i]})')
    ax.set_title(f'Component {component_names[i]}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'predictions_vs_truth.png'), dpi=150, bbox_inches='tight')
print(f"✓ Prediction comparison plot saved")

"""
CELL 8: Checkpoint Recovery Example
"""

# Example: Load from checkpoint
checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_100.pt')
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"✓ Checkpoint loaded from epoch {checkpoint['epoch']}")
else:
    print(f"Checkpoint not found at {checkpoint_path}")

"""
CELL 9: Next Steps
"""

print("""
✓ PINN Colab Setup Complete!

Next Steps:
1. Replace synthetic data with real CFD data
2. Implement full physics constraints (Navier-Stokes + Heat equations)
3. Increase training epochs and model size
4. Add validation against CFD data
5. Train optimization engine using surrogate model

For production training:
- Use 5000+ epochs
- Use larger model (128-256 hidden units)
- Implement adaptive sampling
- Add early stopping
- Monitor validation metrics

Remember:
- Save checkpoints every 500 epochs
- Colab timeout is 12 hours (plan accordingly)
- Use GPU acceleration (K80/T4 on free tier)
""")
