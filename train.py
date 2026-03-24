"""
Main training script for PINN
Can be run locally or adapted for Colab
"""
import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from models import PINN, PhysicsConstraints
from models.losses import PINNLoss
from training import PINNTrainer, LRScheduler, EarlyStopping
from training.config import Config, get_default_config
from utils import DataNormalizer, DataSampler, DataLoader, SyntheticDataGenerator2D


def main(config_file=None, resume_checkpoint=None):
    """
    Main training function.
    
    Args:
        config_file: Path to YAML config file
        resume_checkpoint: Path to checkpoint to resume from
    """
    
    # Load configuration
    config = get_default_config()
    if config_file and os.path.exists(config_file):
        config.load_yaml(config_file)
    
    print("=" * 60)
    print("PINN Liquid Cooling Heat Sink Optimization")
    print("=" * 60)
    print(f"Config: {config['training']}")
    
    # Setup device
    device = torch.device(config.get('device', 'cpu'))
    if device.type == 'cuda':
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ Using CPU")
    
    # Create directories
    checkpoint_dir = Path(config.get('paths.checkpoint_dir', 'checkpoints'))
    output_dir = Path(config.get('paths.output_dir', 'output'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate or load data
    print("\nLoading data...")
    dataset_type = config.get('data.dataset_type', 'synthetic')
    
    if dataset_type == 'synthetic':
        data = SyntheticDataGenerator2D.create_dataset_2d(nx=64, ny=64)
        x_train = data['coordinates']
        y_train = data['fields']
    else:
        print(f"Dataset type '{dataset_type}' not yet implemented")
        return
    
    print(f"✓ Data loaded: x_shape={x_train.shape}, y_shape={y_train.shape}")
    
    # Normalize data
    print("\nNormalizing data...")
    x_normalizer = DataNormalizer(method='minmax')
    x_normalizer.fit(x_train)
    x_train_norm = x_normalizer.normalize(x_train)
    
    y_normalizer = DataNormalizer(method='standardization')
    y_normalizer.fit(y_train)
    y_train_norm = y_normalizer.normalize(y_train)
    
    # Create collocation points for physics residuals
    domain_bounds = config.get('physics.domain_bounds', [[0, 1], [0, 1]])
    n_collocation = config.get('data.collocation_points', 5000)
    x_collocation = DataSampler.uniform_domain_sampling(domain_bounds, n_collocation)
    x_collocation_norm = x_normalizer.normalize(x_collocation)
    
    print(f"✓ Collocation points: {x_collocation.shape}")
    
    # Create data loaders
    train_loader = DataLoader(x_collocation_norm, batch_size=config['training']['batch_size'])
    
    # Create model
    print("\nCreating model...")
    model_type = config.get('model.type', 'small')
    if model_type == 'small':
        model = PINN.create_small_model()
    elif model_type == 'medium':
        model = PINN.create_medium_model()
    elif model_type == 'large':
        model = PINN.create_large_model()
    else:
        model = PINN.create_small_model()
    
    model = model.to(device)
    print(f"✓ Model: {model_type} ({model.count_parameters()} params)")
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )
    
    loss_fn = PINNLoss(
        weight_pde=config.get('loss.weight_pde', 1.0),
        weight_bc=config.get('loss.weight_bc', 10.0),
        weight_data=config.get('loss.weight_data', 1.0)
    )
    
    scheduler = LRScheduler(optimizer, 'exponential', decay=0.995)
    early_stopping = EarlyStopping(patience=config['training'].get('early_stopping_patience', 20))
    
    # Create trainer
    trainer = PINNTrainer(
        model, optimizer, loss_fn, device=device,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=config['training']['checkpoint_interval']
    )
    
    # Resume from checkpoint if provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint)
    
    # Training loop
    print(f"\nTraining for {config['training']['epochs']} epochs...")
    print("-" * 60)
    
    try:
        for epoch in range(trainer.epoch, config['training']['epochs']):
            losses = trainer.train_epoch(train_loader, bc_data={})
            scheduler.step()
            trainer.record_history(losses)
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{config['training']['epochs']}")
                print(f"  Loss: {losses['total']:.6f}")
                print(f"  PDE: {losses['pde']:.6f}, BC: {losses['bc']:.6f}")
            
            # Check early stopping
            if early_stopping(losses['total']):
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
            
            # Save checkpoint
            if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
                trainer.save_checkpoint()
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(filename='pinn_model_final.pt')
    
    # Save training history
    history_file = output_dir / 'training_history.json'
    import json
    with open(history_file, 'w') as f:
        json.dump(trainer.get_history(), f)
    
    print(f"✓ Training history saved to {history_file}")
    
    # Summary
    summary = trainer.get_summary()
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return model, trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PINN for liquid cooling')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    
    args = parser.parse_args()
    
    main(config_file=args.config, resume_checkpoint=args.resume)
