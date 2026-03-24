"""
Heat sink design optimization using trained PINN surrogate model.
"""
import numpy as np
import torch
from abc import ABC, abstractmethod


class SurrogateModel:
    """Wrap trained PINN as fast surrogate for design optimization."""
    
    def __init__(self, model, normalizer_input=None, normalizer_output=None, device='cpu'):
        """
        Initialize surrogate model.
        
        Args:
            model: Trained PINN model
            normalizer_input: Input normalizer
            normalizer_output: Output normalizer
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.model.eval()
        self.normalizer_input = normalizer_input
        self.normalizer_output = normalizer_output
        self.device = device
    
    def predict(self, x):
        """
        Predict at given points.
        
        Args:
            x: Input points (N, 2)
        
        Returns:
            Predictions (N, 4)
        """
        # Normalize input if available
        if self.normalizer_input is not None:
            x_norm = self.normalizer_input.normalize(x)
        else:
            x_norm = x
        
        # Convert to tensor
        x_tensor = torch.from_numpy(x_norm).float().to(self.device)
        
        # Forward pass
        with torch.no_grad():
            y_pred = self.model(x_tensor).cpu().numpy()
        
        # Denormalize output if available
        if self.normalizer_output is not None:
            y_pred = self.normalizer_output.denormalize(y_pred)
        
        return y_pred
    
    def predict_field(self, x_coords, y_coords):
        """
        Predict on a 2D grid.
        
        Args:
            x_coords: x grid coordinates
            y_coords: y grid coordinates
        
        Returns:
            Predictions on grid (ny, nx, 4)
        """
        xx, yy = np.meshgrid(x_coords, y_coords)
        points = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        predictions = self.predict(points)
        
        ny, nx = xx.shape
        return predictions.reshape(ny, nx, 4)
    
    def compute_thermal_resistance(self, T_inlet, T_outlet, Q):
        """
        Compute thermal resistance.
        
        Args:
            T_inlet: Inlet temperature
            T_outlet: Outlet temperature
            Q: Heat transfer rate (W)
        
        Returns:
            Thermal resistance (K/W)
        """
        if Q == 0:
            return float('inf')
        return (T_outlet - T_inlet) / Q
    
    def compute_pressure_drop(self, p_inlet, p_outlet):
        """Compute pressure drop."""
        return abs(p_inlet - p_outlet)


class ObjectiveFunction:
    """Define objective functions for optimization."""
    
    @staticmethod
    def thermal_resistance(surrogate, design_params):
        """
        Minimize thermal resistance.
        
        Args:
            surrogate: Trained surrogate model
            design_params: Design parameters (geometry info)
        
        Returns:
            Thermal resistance value
        """
        # Evaluate surrogate at key points
        T_out = 350.0  # Placeholder
        T_in = 300.0
        Q = 100.0
        
        R_th = surrogate.compute_thermal_resistance(T_in, T_out, Q)
        return R_th
    
    @staticmethod
    def pressure_drop(surrogate, design_params):
        """Minimize pressure drop."""
        p_inlet = 101325
        p_outlet = 100000
        dp = surrogate.compute_pressure_drop(p_inlet, p_outlet)
        return dp
    
    @staticmethod
    def combined_objective(surrogate, design_params, w_thermal=0.7, w_pressure=0.3):
        """
        Combined objective: thermal performance + pressure drop.
        
        Args:
            surrogate: Trained surrogate model
            design_params: Design parameters
            w_thermal: Weight for thermal resistance
            w_pressure: Weight for pressure drop
        
        Returns:
            Combined objective value
        """
        R_th = ObjectiveFunction.thermal_resistance(surrogate, design_params)
        dp = ObjectiveFunction.pressure_drop(surrogate, design_params)
        
        # Normalize
        R_th_norm = R_th / 0.001  # Typical R_th ~ 0.001 K/W
        dp_norm = dp / 1000  # Typical dp ~ 1000 Pa
        
        return w_thermal * R_th_norm + w_pressure * dp_norm


class HeatSinkOptimizer(ABC):
    """Base class for heat sink design optimizers."""
    
    def __init__(self, surrogate, objective_fn, design_bounds, seed=None):
        """
        Args:
            surrogate: Trained surrogate model
            objective_fn: Objective function to minimize
            design_bounds: Bounds for design parameters [(min, max), ...]
            seed: Random seed
        """
        self.surrogate = surrogate
        self.objective_fn = objective_fn
        self.design_bounds = design_bounds
        self.seed = seed
        self.history = []
        
        if seed is not None:
            np.random.seed(seed)
    
    @abstractmethod
    def optimize(self, n_iterations):
        """Run optimization for n iterations."""
        pass
    
    def get_history(self):
        """Get optimization history."""
        return self.history


class BayesianOptimizer(HeatSinkOptimizer):
    """Bayesian optimization for heat sink design."""
    
    def __init__(self, surrogate, objective_fn, design_bounds, seed=None,
                 n_initial_samples=10):
        """
        Args:
            surrogate: Trained surrogate model
            objective_fn: Objective function
            design_bounds: Design parameter bounds
            seed: Random seed
            n_initial_samples: Initial samples before Bayesian optimization
        """
        super().__init__(surrogate, objective_fn, design_bounds, seed)
        self.n_initial_samples = n_initial_samples
        self.best_design = None
        self.best_value = float('inf')
    
    def optimize(self, n_iterations=100):
        """
        Run Bayesian optimization.
        
        Args:
            n_iterations: Number of optimization iterations
        
        Returns:
            Best design found
        """
        # Random initial sampling
        for _ in range(self.n_initial_samples):
            design = np.array([
                np.random.uniform(bound[0], bound[1]) 
                for bound in self.design_bounds
            ])
            
            value = self.objective_fn(self.surrogate, design)
            self.history.append({'design': design, 'value': value})
            
            if value < self.best_value:
                self.best_value = value
                self.best_design = design.copy()
        
        # Acquisition function optimization (simplified)
        for iteration in range(n_iterations):
            # Generate candidate designs
            candidates = np.array([
                np.random.uniform(bound[0], bound[1], 100)
                for bound in self.design_bounds
            ]).T
            
            # Evaluate candidates
            values = np.array([
                self.objective_fn(self.surrogate, cand)
                for cand in candidates
            ])
            
            # Select best candidate
            best_idx = np.argmin(values)
            best_candidate = candidates[best_idx]
            best_value = values[best_idx]
            
            self.history.append({'design': best_candidate, 'value': best_value})
            
            if best_value < self.best_value:
                self.best_value = best_value
                self.best_design = best_candidate.copy()
        
        return self.best_design, self.best_value


class EvolutionaryOptimizer(HeatSinkOptimizer):
    """Evolutionary/Genetic algorithm for design optimization."""
    
    def __init__(self, surrogate, objective_fn, design_bounds, seed=None,
                 population_size=20, mutation_rate=0.1):
        """
        Args:
            surrogate: Trained surrogate model
            objective_fn: Objective function
            design_bounds: Design parameter bounds
            seed: Random seed
            population_size: EA population size
            mutation_rate: Mutation probability
        """
        super().__init__(surrogate, objective_fn, design_bounds, seed)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = None
        self.fitness = None
    
    def _initialize_population(self):
        """Initialize random population."""
        n_params = len(self.design_bounds)
        self.population = np.array([
            np.array([
                np.random.uniform(bound[0], bound[1])
                for bound in self.design_bounds
            ])
            for _ in range(self.population_size)
        ])
    
    def _evaluate_fitness(self):
        """Evaluate fitness for entire population."""
        self.fitness = np.array([
            self.objective_fn(self.surrogate, design)
            for design in self.population
        ])
    
    def _selection(self):
        """Tournament selection."""
        selected_indices = []
        for _ in range(self.population_size):
            # Tournament: pick 2 random individuals, return better
            idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
            if self.fitness[idx1] < self.fitness[idx2]:
                selected_indices.append(idx1)
            else:
                selected_indices.append(idx2)
        
        return self.population[selected_indices]
    
    def _crossover(self, parents):
        """Uniform crossover."""
        offspring = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[i]
            
            # Uniform crossover
            mask = np.random.rand(len(parent1)) < 0.5
            child = np.where(mask, parent1, parent2)
            offspring.append(child)
        
        return np.array(offspring)
    
    def _mutation(self, population):
        """Gaussian mutation."""
        mutated = population.copy()
        for i in range(len(mutated)):
            if np.random.rand() < self.mutation_rate:
                param_idx = np.random.randint(len(self.design_bounds))
                bound = self.design_bounds[param_idx]
                mutation = np.random.normal(0, 0.01 * (bound[1] - bound[0]))
                mutated[i, param_idx] = np.clip(
                    mutated[i, param_idx] + mutation,
                    bound[0], bound[1]
                )
        
        return mutated
    
    def optimize(self, n_generations=50):
        """
        Run evolutionary optimization.
        
        Args:
            n_generations: Number of generations
        
        Returns:
            Best design found
        """
        self._initialize_population()
        best_design = None
        best_value = float('inf')
        
        for generation in range(n_generations):
            self._evaluate_fitness()
            
            # Track best
            gen_best_idx = np.argmin(self.fitness)
            gen_best_value = self.fitness[gen_best_idx]
            
            if gen_best_value < best_value:
                best_value = gen_best_value
                best_design = self.population[gen_best_idx].copy()
            
            self.history.append({
                'generation': generation,
                'best_value': best_value,
                'mean_fitness': np.mean(self.fitness)
            })
            
            # Selection, crossover, mutation
            selected = self._selection()
            offspring = self._crossover(selected)
            mutated = self._mutation(offspring)
            
            # Elitism: keep best individual
            worst_idx = np.argmax(self.fitness)
            mutated[0] = best_design.copy()
            
            self.population = mutated
        
        return best_design, best_value
