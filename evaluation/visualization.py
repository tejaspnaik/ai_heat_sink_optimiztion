"""
Visualization utilities for PINN results.
"""
import numpy as np
import matplotlib.pyplot as plt


class PINNVisualizer:
    """Visualization utilities for PINN results."""
    
    @staticmethod
    def plot_training_history(history, figsize=(12, 4)):
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Total loss
        if 'loss_total' in history:
            axes[0].plot(history['epoch'], history['loss_total'], label='Total Loss')
            axes[0].plot(history['epoch'], history['loss_pde'], label='PDE Loss', alpha=0.7)
            axes[0].plot(history['epoch'], history['loss_bc'], label='BC Loss', alpha=0.7)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].set_yscale('log')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Loss components
        if 'loss_pde' in history and 'loss_bc' in history:
            axes[1].plot(history['epoch'], history['loss_pde'], label='PDE Loss')
            axes[1].plot(history['epoch'], history['loss_bc'], label='BC Loss')
            if 'loss_data' in history:
                axes[1].plot(history['epoch'], history['loss_data'], label='Data Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Loss Components')
            axes[1].set_yscale('log')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Validation loss
        if 'val_loss' in history and len(history['val_loss']) > 0:
            axes[2].plot(history['epoch'], history['val_loss'], label='Validation Loss', color='red')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Loss')
            axes[2].set_title('Validation Loss')
            axes[2].set_yscale('log')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_field(field, title, cmap='RdBu_r', figsize=(8, 6)):
        """
        Plot 2D field.
        
        Args:
            field: 2D field array
            title: Field title
            cmap: Colormap
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(field, cmap=cmap, aspect='auto', origin='lower')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_comparison(pinn_field, cfd_field, component_name='u', figsize=(15, 4)):
        """
        Compare PINN and CFD fields.
        
        Args:
            pinn_field: PINN prediction
            cfd_field: CFD ground truth
            component_name: Variable name (u, v, p, T)
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # PINN
        im1 = axes[0].imshow(pinn_field, cmap='RdBu_r', aspect='auto', origin='lower')
        axes[0].set_title(f'PINN: {component_name}')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])
        
        # CFD
        im2 = axes[1].imshow(cfd_field, cmap='RdBu_r', aspect='auto', origin='lower')
        axes[1].set_title(f'CFD: {component_name}')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference
        diff = pinn_field - cfd_field
        im3 = axes[2].imshow(diff, cmap='seismic', aspect='auto', origin='lower')
        axes[2].set_title(f'Difference: {component_name}')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_optimization_history(history, figsize=(10, 5)):
        """
        Plot optimization history.
        
        Args:
            history: Optimization history list
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        objectives = [item['value'] for item in history]
        iterations = range(len(objectives))
        
        ax.plot(iterations, objectives, marker='o', linestyle='-', markersize=4)
        ax.plot(iterations, np.minimum.accumulate(objectives), 'r--', label='Best so far')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('Optimization Progress')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_convergence(pinn_pred, cfd_true, figsize=(10, 5)):
        """
        Plot prediction vs. ground truth scatter.
        
        Args:
            pinn_pred: PINN predictions
            cfd_true: CFD ground truth
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(cfd_true, pinn_pred, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(cfd_true.min(), pinn_pred.min())
        max_val = max(cfd_true.max(), pinn_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        ax.set_xlabel('CFD Ground Truth')
        ax.set_ylabel('PINN Prediction')
        ax.set_title('Prediction Accuracy')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_velocity_vectors(u_field, v_field, skip=2, figsize=(10, 8)):
        """
        Plot velocity vector field.
        
        Args:
            u_field: u-component (2D array)
            v_field: v-component (2D array)
            skip: Plot every nth vector
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ny, nx = u_field.shape
        x = np.arange(0, nx, skip)
        y = np.arange(0, ny, skip)
        
        ax.quiver(x, y, u_field[::skip, ::skip], v_field[::skip, ::skip])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Velocity Field')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_metrics(metrics, figsize=(10, 6)):
        """
        Plot validation metrics.
        
        Args:
            metrics: Metrics dictionary
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        keys = list(metrics.keys())
        values = list(metrics.values())
        
        # Filter numeric values
        numeric_items = [(k, v) for k, v in zip(keys, values) if isinstance(v, (int, float))]
        
        if numeric_items:
            keys, values = zip(*numeric_items)
            
            bars = ax.bar(range(len(keys)), values)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=45, ha='right')
            ax.set_ylabel('Value')
            ax.set_title('Validation Metrics')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig


def save_figure(fig, filepath):
    """Save figure to file."""
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {filepath}")


def close_all_figures():
    """Close all matplotlib figures."""
    plt.close('all')
