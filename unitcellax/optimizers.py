"""
Optimization algorithms for unitcellax with JAX integration.

This module provides abstract base classes and concrete implementations for
optimization algorithms that work seamlessly with JAX gradients and memory management.

Performance Notes:
    - These optimizers are designed for CPU computation with JAX
    - GPU acceleration is available through JAX backend configuration
    - For large-scale problems, consider GPU-enabled JAX installations
    - Memory management is handled automatically to prevent accumulation
"""

import gc
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import jax
import nlopt
import numpy as onp


class JAXOptimizer(ABC):
    """Abstract base class for JAX-aware optimizers.
    
    This class provides a common interface for optimization algorithms that work
    with JAX gradient computation and automatic memory management.
    """
    
    def __init__(self, memory_cleanup_freq: int = 5):
        """Initialize the JAX optimizer.
        
        Args:
            memory_cleanup_freq (int): Frequency of JAX cache clearing and garbage collection.
        """
        self.memory_cleanup_freq = memory_cleanup_freq
        self.epoch_counter = 0
        
    @abstractmethod
    def optimize(self, initial_guess: onp.ndarray) -> Tuple[onp.ndarray, float]:
        """Perform optimization starting from initial guess.
        
        Args:
            initial_guess (np.ndarray): Initial parameter values.
            
        Returns:
            Tuple[np.ndarray, float]: Optimal parameters and objective value.
        """
        pass
    
    def _jax_to_numpy(self, jax_array):
        """Convert JAX array to numpy to break autograd references and prevent memory leaks.
        
        Args:
            jax_array: JAX array to convert.
            
        Returns:
            np.ndarray: Converted numpy array with memory leak prevention.
        """
        # Prevent memory leaks by properly handling JAX to numpy conversion
        try:
            # Stop gradient computation to break autograd tape
            jax_array = jax.lax.stop_gradient(jax_array)
            
            # Use device_get to move from device and convert to numpy
            # This ensures memory is properly released
            numpy_array = jax.device_get(jax_array)
            
            # Create a copy to ensure complete separation from JAX memory
            result = onp.array(numpy_array, copy=True)
            
            # Explicitly delete intermediate references
            del numpy_array
            
            return result
            
        except Exception as e:
            # Fallback for edge cases
            try:
                # For scalar values
                if hasattr(jax_array, 'ndim') and jax_array.ndim == 0:
                    return float(jax_array)
                # For arrays, force copy
                return onp.array(jax_array, copy=True)
            except Exception:
                # Last resort
                return float(jax_array) if hasattr(jax_array, 'ndim') and jax_array.ndim == 0 else onp.array(jax_array)
    
    def _periodic_cleanup(self):
        """Perform periodic memory cleanup to prevent accumulation."""
        self.epoch_counter += 1
        if self.epoch_counter % self.memory_cleanup_freq == 0:
            print("!!Cache clear")
            # Clear JAX compilation cache
            if hasattr(jax._src.interpreters.xla, '_xla_callable'):
                jax._src.interpreters.xla._xla_callable.cache_clear()
            # Clear all JAX caches
            jax.clear_caches()
            # Force garbage collection
            gc.collect()
            gc.collect()
            gc.collect()


class NLoptJAXOptimizer(JAXOptimizer, nlopt.opt):
    """NLopt optimizer with JAX gradient computation and memory management.
    
    This class wraps NLopt algorithms to work seamlessly with JAX functions,
    handling gradient computation and memory management automatically.
    """
    
    def __init__(self, 
                 algorithm: int, 
                 n_vars: int, 
                 objective_fn: Callable,
                 constraints: Optional[List[Tuple[Callable, float]]] = None,
                 save_callback: Optional[Callable] = None,
                 memory_cleanup_freq: int = 5):
        """Initialize the NLopt JAX optimizer.
        
        Args:
            algorithm (int): NLopt algorithm identifier (e.g., nlopt.LD_MMA).
            n_vars (int): Number of optimization variables.
            objective_fn (Callable): Objective function that takes parameters and returns scalar.
            constraints (List[Tuple[Callable, float]], optional): List of (constraint_fn, tolerance) pairs.
            save_callback (Callable, optional): Callback function for saving intermediate results.
            memory_cleanup_freq (int): Frequency of memory cleanup operations.
        """
        # Initialize parent classes
        JAXOptimizer.__init__(self, memory_cleanup_freq)
        nlopt.opt.__init__(self, algorithm, n_vars)
        
        self.objective_fn = objective_fn
        self.constraints = constraints or []
        self.save_callback = save_callback
        
        # Set up the optimization
        self.set_min_objective(self._objective_wrapper)
        for constraint_fn, tolerance in self.constraints:
            self.add_inequality_constraint(self._constraint_wrapper(constraint_fn), tolerance)
    
    def _objective_wrapper(self, x: onp.ndarray, grad: onp.ndarray) -> float:
        """Internal wrapper for objective function with JAX handling and memory leak prevention.
        
        Args:
            x (np.ndarray): Current parameter values.
            grad (np.ndarray): Gradient array to fill.
            
        Returns:
            float: Objective function value.
        """
        # Ensure x is a proper numpy array to prevent JAX memory references
        x_copy = onp.array(x, copy=True)
        
        # Compute objective and gradient
        J, dJ = jax.value_and_grad(self.objective_fn)(x_copy)
        
        # Convert JAX arrays to numpy with proper memory management
        J_np = float(self._jax_to_numpy(J))
        dJ_np = self._jax_to_numpy(dJ)
        
        if grad.size > 0:
            grad[:] = dJ_np.copy()  # Ensure a copy is made
            
        # Explicitly delete intermediate JAX arrays to prevent leaks
        del J, dJ
        
        # Save visualization if callback provided
        if self.save_callback:
            self.save_callback(x_copy, self.epoch_counter)
            
        # Periodic memory cleanup
        self._periodic_cleanup()
        
        # Delete copies to ensure cleanup
        del x_copy, dJ_np
        
        # Force immediate garbage collection for critical arrays
        gc.collect(0)
            
        print(f"Objective: {J_np:.6e}")
        return J_np
    
    def _constraint_wrapper(self, constraint_fn: Callable) -> Callable:
        """Internal wrapper for constraint functions with JAX handling.
        
        Args:
            constraint_fn (Callable): Constraint function to wrap.
            
        Returns:
            Callable: Wrapped constraint function.
        """
        def wrapper(x: onp.ndarray, grad: onp.ndarray) -> float:
            # Ensure x is a proper numpy array to prevent JAX memory references
            x_copy = onp.array(x, copy=True)
            
            c, gradc = jax.value_and_grad(constraint_fn)(x_copy)
            
            # Convert JAX arrays to numpy with proper memory management
            c_np = float(self._jax_to_numpy(c))
            gradc_np = self._jax_to_numpy(gradc)
            
            if grad.size > 0:
                grad[:] = gradc_np
            
            # Explicitly delete intermediate JAX arrays to prevent leaks
            del c, gradc, x_copy, gradc_np
                
            print(f"Constraint: {c_np:.6e}")
            return c_np
        return wrapper
    
    def optimize(self, initial_guess: onp.ndarray) -> Tuple[onp.ndarray, float]:
        """Perform optimization using NLopt.
        
        Args:
            initial_guess (np.ndarray): Initial parameter values.
            
        Returns:
            Tuple[np.ndarray, float]: Optimal parameters and objective value.
        """
        try:
            # Call nlopt.opt.optimize() method correctly
            x_opt = nlopt.opt.optimize(self, initial_guess)
            opt_val = self.last_optimum_value()
            result = self.last_optimize_result()
            print(f"Optimization completed with result: {result}")
            print(f"Optimal objective value: {opt_val:.6e}")
            return x_opt, opt_val
        except Exception as e:
            print(f"Optimization failed: {e}")
            return initial_guess, float('inf')


class GCMMAOptimizer(NLoptJAXOptimizer):
    """GCMMA (Globally Convergent Method of Moving Asymptotes) optimizer for topology optimization.
    
    This class provides a convenient interface for topology optimization using the MMA algorithm
    with global convergence guarantees. Suitable for gradient-based optimization of structural
    compliance, volume constraints, and other topology optimization objectives.
    
    Performance Notes:
        - Optimized for CPU computation with JAX automatic differentiation
        - Memory-efficient with automatic JAX cache management
        - Supports large-scale problems with sparse matrix operations
        - For GPU acceleration, ensure JAX is configured with GPU backend
    
    Algorithm:
        Uses NLopt's LD_MMA (Method of Moving Asymptotes) algorithm which is equivalent
        to GCMMA for most topology optimization problems.
    """
    
    def __init__(self,
                 n_vars: int,
                 objective_fn: Callable,
                 volume_constraint_fn: Callable,
                 volume_fraction: float = 0.5,
                 algorithm: int = nlopt.LD_MMA,
                 save_callback: Optional[Callable] = None,
                 memory_cleanup_freq: int = 5):
        """Initialize topology optimizer.
        
        Args:
            n_vars (int): Number of design variables.
            objective_fn (Callable): Objective function (e.g., compliance).
            volume_constraint_fn (Callable): Volume constraint function.
            volume_fraction (float): Target volume fraction.
            algorithm (int): NLopt algorithm to use.
            save_callback (Callable, optional): Callback for saving intermediate results.
            memory_cleanup_freq (int): Memory cleanup frequency.
        """
        # Set up volume constraint with tolerance
        constraints = [(volume_constraint_fn, 1e-8)]
        
        super().__init__(
            algorithm=algorithm,
            n_vars=n_vars,
            objective_fn=objective_fn,
            constraints=constraints,
            save_callback=save_callback,
            memory_cleanup_freq=memory_cleanup_freq
        )
        
        # Set typical bounds for topology optimization (0 to 1)
        self.set_lower_bounds(onp.zeros(n_vars))
        self.set_upper_bounds(onp.ones(n_vars))
        
        # Set typical stopping criteria
        self.set_maxeval(50)  # Maximum number of evaluations
        self.set_xtol_rel(1e-6)  # Relative tolerance on design variables
        self.set_ftol_rel(1e-9)  # Relative tolerance on objective
        
    def set_gcmma_options(self, 
                         max_eval: int = 50,
                         x_tol: float = 1e-6,
                         f_tol: float = 1e-9):
        """Set GCMMA algorithm specific options.
        
        Args:
            max_eval (int): Maximum number of function evaluations (CPU-intensive).
            x_tol (float): Relative tolerance on design variables.
            f_tol (float): Relative tolerance on objective function.
            
        Performance Notes:
            - Higher max_eval values increase CPU computation time significantly
            - Tighter tolerances (smaller x_tol, f_tol) require more iterations
            - For large problems (>10k variables), consider looser tolerances initially
        """
        self.set_maxeval(max_eval)
        self.set_xtol_rel(x_tol)
        self.set_ftol_rel(f_tol)