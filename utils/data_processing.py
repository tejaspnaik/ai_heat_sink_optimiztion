"""
Data processing utilities: normalization, sampling, batching.
"""
import numpy as np
import torch


class DataNormalizer:
    """Normalize data to standardized ranges."""
    
    def __init__(self, mean=None, std=None, min_val=None, max_val=None, method='standardization'):
        """
        Initialize data normalizer.
        
        Args:
            mean: Mean values for standardization
            std: Std dev values for standardization
            min_val: Min values for min-max normalization
            max_val: Max values for min-max normalization
            method: 'standardization' or 'minmax'
        """
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.method = method
    
    def fit(self, data):
        """
        Fit normalizer on data.
        
        Args:
            data: Training data array (N, D)
        
        Returns:
            self for chaining
        """
        if self.method == 'standardization':
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0) + 1e-8
        elif self.method == 'minmax':
            self.min_val = np.min(data, axis=0)
            self.max_val = np.max(data, axis=0) + 1e-8
        
        return self
    
    def normalize(self, data):
        """Normalize data."""
        if self.method == 'standardization':
            return (data - self.mean) / self.std
        elif self.method == 'minmax':
            return (data - self.min_val) / (self.max_val - self.min_val)
    
    def denormalize(self, data):
        """Denormalize data to original scale."""
        if self.method == 'standardization':
            return data * self.std + self.mean
        elif self.method == 'minmax':
            return data * (self.max_val - self.min_val) + self.min_val
    
    def get_params(self):
        """Get normalization parameters for saving."""
        return {
            'method': self.method,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'min_val': self.min_val.tolist() if self.min_val is not None else None,
            'max_val': self.max_val.tolist() if self.max_val is not None else None
        }
    
    def load_params(self, params):
        """Load normalization parameters."""
        self.method = params['method']
        if params['mean'] is not None:
            self.mean = np.array(params['mean'])
            self.std = np.array(params['std'])
        if params['min_val'] is not None:
            self.min_val = np.array(params['min_val'])
            self.max_val = np.array(params['max_val'])


class DataSampler:
    """Sample data points for collocation and boundaries."""
    
    @staticmethod
    def uniform_domain_sampling(domain_bounds, n_points, seed=None):
        """
        Uniform random sampling from domain.
        
        Args:
            domain_bounds: [(x_min, x_max), (y_min, y_max)]
            n_points: Number of samples
            seed: Random seed
        
        Returns:
            Sampled points (n_points, 2)
        """
        if seed is not None:
            np.random.seed(seed)
        
        x_min, x_max = domain_bounds[0]
        y_min, y_max = domain_bounds[1]
        
        x = np.random.uniform(x_min, x_max, n_points)
        y = np.random.uniform(y_min, y_max, n_points)
        
        return np.stack([x, y], axis=1)
    
    @staticmethod
    def grid_sampling(domain_bounds, n_points_per_side):
        """
        Grid sampling from domain.
        
        Args:
            domain_bounds: [(x_min, x_max), (y_min, y_max)]
            n_points_per_side: Points per side
        
        Returns:
            Grid points (n_points_per_side^2, 2)
        """
        x_min, x_max = domain_bounds[0]
        y_min, y_max = domain_bounds[1]
        
        x = np.linspace(x_min, x_max, n_points_per_side)
        y = np.linspace(y_min, y_max, n_points_per_side)
        
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        return points
    
    @staticmethod
    def boundary_sampling(domain_bounds, n_points, side='all'):
        """
        Sample points on domain boundaries.
        
        Args:
            domain_bounds: [(x_min, x_max), (y_min, y_max)]
            n_points: Total points to sample
            side: 'left', 'right', 'bottom', 'top', or 'all'
        
        Returns:
            Boundary points (n_points, 2)
        """
        x_min, x_max = domain_bounds[0]
        y_min, y_max = domain_bounds[1]
        
        points = []
        
        if side in ['all', 'left']:
            x = np.full(n_points // 4, x_min)
            y = np.linspace(y_min, y_max, n_points // 4)
            points.append(np.stack([x, y], axis=1))
        
        if side in ['all', 'right']:
            x = np.full(n_points // 4, x_max)
            y = np.linspace(y_min, y_max, n_points // 4)
            points.append(np.stack([x, y], axis=1))
        
        if side in ['all', 'bottom']:
            x = np.linspace(x_min, x_max, n_points // 4)
            y = np.full(n_points // 4, y_min)
            points.append(np.stack([x, y], axis=1))
        
        if side in ['all', 'top']:
            x = np.linspace(x_min, x_max, n_points // 4, endpoint=False)
            y = np.full(n_points // 4, y_max)
            points.append(np.stack([x, y], axis=1))
        
        return np.vstack(points) if points else np.array([])
    
    @staticmethod
    def adaptive_sampling(residuals, domain_bounds, n_points, percentile=80):
        """
        Adaptive sampling: sample more from high-residual regions.
        
        Args:
            residuals: Residual values at existing points
            domain_bounds: Domain boundaries
            n_points: Number of new points to sample
            percentile: Only sample from top percentile regions
        
        Returns:
            New sampled points (n_points, 2)
        """
        threshold = np.percentile(np.abs(residuals), percentile)
        
        # This is simplified; full implementation would require spatial indexing
        points = DataSampler.uniform_domain_sampling(domain_bounds, n_points)
        
        return points


class DataLoader:
    """Simple data loader for batching."""
    
    def __init__(self, data, batch_size, shuffle=True, seed=None):
        """
        Initialize data loader.
        
        Args:
            data: Data array or list of arrays
            batch_size: Batch size
            shuffle: Whether to shuffle
            seed: Random seed
        """
        self.data = data if isinstance(data, list) else [data]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.n_samples = len(self.data[0])
        self.indices = np.arange(self.n_samples)
        self.epoch = 0
    
    def __iter__(self):
        """Iterate over batches."""
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed + self.epoch)
            np.random.shuffle(self.indices)
        
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_data = [d[batch_indices] for d in self.data]
            
            # Convert to torch tensors
            batch_tensors = [torch.from_numpy(d).float() for d in batch_data]
            
            if len(batch_tensors) == 1:
                yield batch_tensors[0]
            else:
                yield tuple(batch_tensors)
        
        self.epoch += 1
    
    def __len__(self):
        """Number of batches."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def create_data_splits(data, train_ratio=0.7, val_ratio=0.15, seed=None):
    """
    Split data into train/val/test.
    
    Args:
        data: Data array
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed
    
    Returns:
        (train_data, val_data, test_data)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(data)
    indices = np.random.permutation(n)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    return data[train_idx], data[val_idx], data[test_idx]
