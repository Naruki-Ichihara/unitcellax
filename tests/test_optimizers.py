"""
Tests for unitcellax optimizers module.

This module tests the JAX-integrated optimization classes including
memory management, gradient computation, and algorithm convergence.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from unittest.mock import Mock, patch
import nlopt

from unitcellax.optimizers import JAXOptimizer, NLoptJAXOptimizer, GCMMAOptimizer


class TestJAXOptimizer:
    """Test the abstract JAXOptimizer base class."""
    
    def test_jax_to_numpy_conversion(self):
        """Test JAX array to numpy conversion."""
        
        class ConcreteOptimizer(JAXOptimizer):
            def optimize(self, initial_guess):
                return initial_guess, 0.0
        
        optimizer = ConcreteOptimizer()
        
        # Test with JAX array
        jax_array = jnp.array([1.0, 2.0, 3.0])
        numpy_result = optimizer._jax_to_numpy(jax_array)
        
        assert isinstance(numpy_result, np.ndarray)
        np.testing.assert_array_equal(numpy_result, np.array([1.0, 2.0, 3.0]))
    
    def test_memory_cleanup_frequency(self):
        """Test that memory cleanup occurs at specified frequency."""
        
        class ConcreteOptimizer(JAXOptimizer):
            def optimize(self, initial_guess):
                return initial_guess, 0.0
        
        optimizer = ConcreteOptimizer(memory_cleanup_freq=3)
        
        # Mock JAX clear_caches and gc.collect
        with patch('jax.clear_caches') as mock_clear, \
             patch('gc.collect') as mock_gc:
            
            # Call cleanup multiple times
            for i in range(5):
                optimizer._periodic_cleanup()
            
            # Should trigger cleanup at iterations 3 and 6 (but we only go to 5)
            assert mock_clear.call_count == 1  # Called once at iteration 3
            assert mock_gc.call_count == 1


class TestNLoptJAXOptimizer:
    """Test the NLopt JAX optimizer wrapper."""
    
    def test_simple_quadratic_optimization(self):
        """Test optimization of a simple quadratic function."""
        
        def objective(x):
            """Simple quadratic: f(x) = (x[0]-1)^2 + (x[1]-2)^2"""
            return (x[0] - 1.0)**2 + (x[1] - 2.0)**2
        
        optimizer = NLoptJAXOptimizer(
            algorithm=nlopt.LD_LBFGS,
            n_vars=2,
            objective_fn=objective,
            memory_cleanup_freq=10  # Less frequent for short test
        )
        
        # Set bounds and stopping criteria
        optimizer.set_lower_bounds(np.array([-5.0, -5.0]))
        optimizer.set_upper_bounds(np.array([5.0, 5.0]))
        optimizer.set_maxeval(100)
        optimizer.set_ftol_rel(1e-8)
        
        # Initial guess
        x0 = np.array([0.0, 0.0])
        
        # Optimize
        x_opt, f_opt = optimizer.optimize(x0)
        
        # Check convergence to expected minimum
        assert abs(x_opt[0] - 1.0) < 1e-3
        assert abs(x_opt[1] - 2.0) < 1e-3
        assert abs(f_opt) < 1e-6
    
    def test_constrained_optimization(self):
        """Test optimization with constraints."""
        
        def objective(x):
            """Minimize x[0]^2 + x[1]^2"""
            return x[0]**2 + x[1]**2
        
        def constraint(x):
            """Constraint: x[0] + x[1] - 1 >= 0"""
            # NLopt expects constraints in the form g(x) >= 0
            # So we flip the sign from the original <= constraint
            return 1.0 - x[0] - x[1]
        
        optimizer = NLoptJAXOptimizer(
            algorithm=nlopt.LD_MMA,
            n_vars=2,
            objective_fn=objective,
            constraints=[(constraint, 1e-8)],
            memory_cleanup_freq=10
        )
        
        # Set bounds and stopping criteria
        optimizer.set_lower_bounds(np.array([0.0, 0.0]))
        optimizer.set_upper_bounds(np.array([2.0, 2.0]))
        optimizer.set_maxeval(100)
        optimizer.set_ftol_rel(1e-6)
        
        # Initial guess
        x0 = np.array([0.8, 0.8])
        
        # Optimize
        x_opt, f_opt = optimizer.optimize(x0)
        
        # Check that constraint is satisfied
        constraint_value = constraint(x_opt)
        assert constraint_value <= 1e-3  # Constraint satisfied (g(x) >= 0 means constraint(x) <= 0)
        
        # Check that the sum is approximately 1 (on the constraint boundary)
        sum_value = x_opt[0] + x_opt[1]
        assert abs(sum_value - 1.0) < 1e-2  # Should be close to 1
        
        # Check that solution minimizes objective on constraint
        # The optimal solution should have x[0] ≈ x[1] ≈ 0.5
        assert abs(x_opt[0] - x_opt[1]) < 0.1  # Should be approximately equal
    
    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        
        def objective(x):
            return x[0]**2 + 2*x[1]**2
        
        # Track gradient computations
        gradient_history = []
        
        def mock_objective_wrapper(x, grad):
            J, dJ = jax.value_and_grad(objective)(x)
            if grad.size > 0:
                gradient_history.append((np.array(x), np.array(dJ)))
                grad[:] = dJ
            return float(J)
        
        optimizer = NLoptJAXOptimizer(
            algorithm=nlopt.LD_LBFGS,
            n_vars=2,
            objective_fn=objective,
            memory_cleanup_freq=10
        )
        
        # Replace the wrapper with our mock
        optimizer._objective_wrapper = mock_objective_wrapper
        optimizer.set_min_objective(mock_objective_wrapper)
        
        optimizer.set_lower_bounds(np.array([-2.0, -2.0]))
        optimizer.set_upper_bounds(np.array([2.0, 2.0]))
        optimizer.set_maxeval(5)  # Just a few iterations to check gradient
        
        x0 = np.array([1.0, 1.0])
        x_opt, f_opt = optimizer.optimize(x0)
        
        # Check that gradients were computed
        assert len(gradient_history) > 0, "No gradients were computed"
        
        # Verify gradient computation at initial point
        # The first gradient computation should be at or near x0
        first_x, first_grad = gradient_history[0]
        expected_grad_at_x0 = np.array([2.0 * x0[0], 4.0 * x0[1]])  # [2.0, 4.0]
        
        # Check if first evaluation was at initial point
        if np.allclose(first_x, x0, rtol=1e-6):
            np.testing.assert_allclose(first_grad, expected_grad_at_x0, rtol=1e-6)
        
        # Verify gradient computation is correct for any point in history
        for x, grad in gradient_history:
            expected_grad = np.array([2.0 * x[0], 4.0 * x[1]])
            np.testing.assert_allclose(grad, expected_grad, rtol=1e-6)


class TestGCMMAOptimizer:
    """Test the GCMMA topology optimizer."""
    
    def test_initialization(self):
        """Test proper initialization of GCMMA optimizer."""
        
        def objective(x):
            return np.sum(x**2)
        
        def volume_constraint(x):
            return np.mean(x) - 0.5
        
        optimizer = GCMMAOptimizer(
            n_vars=10,
            objective_fn=objective,
            volume_constraint_fn=volume_constraint,
            volume_fraction=0.5
        )
        
        # Check that bounds are set correctly
        assert optimizer.get_lower_bounds() is not None
        assert optimizer.get_upper_bounds() is not None
        assert len(optimizer.get_lower_bounds()) == 10
        assert len(optimizer.get_upper_bounds()) == 10
        
        # Check bounds values
        np.testing.assert_array_equal(optimizer.get_lower_bounds(), np.zeros(10))
        np.testing.assert_array_equal(optimizer.get_upper_bounds(), np.ones(10))
    
    def test_gcmma_options(self):
        """Test setting GCMMA-specific options."""
        
        def objective(x):
            return np.sum(x**2)
        
        def volume_constraint(x):
            return np.mean(x) - 0.5
        
        optimizer = GCMMAOptimizer(
            n_vars=5,
            objective_fn=objective,
            volume_constraint_fn=volume_constraint
        )
        
        # Set custom options
        optimizer.set_gcmma_options(max_eval=100, x_tol=1e-5, f_tol=1e-8)
        
        # Note: NLopt doesn't provide getters for these values,
        # so we can't directly test them. This mainly tests that
        # the method doesn't raise exceptions.
        assert True  # If we get here, no exceptions were raised
    
    def test_topology_optimization_problem(self):
        """Test a simple constrained optimization problem with GCMMAOptimizer."""
        
        def simple_objective(x):
            """Simple quadratic objective with bias towards higher values."""
            # Minimize sum of squares but penalize values close to zero
            # This encourages non-zero solutions
            return np.sum((x - 0.7)**2)
        
        def volume_constraint(x):
            """Volume constraint: mean density - target <= 0"""
            return np.mean(x) - 0.4  # Target 40% volume fraction
        
        optimizer = GCMMAOptimizer(
            n_vars=3,  # Smaller problem for stability
            objective_fn=simple_objective,
            volume_constraint_fn=volume_constraint,
            volume_fraction=0.4,
            memory_cleanup_freq=10
        )
        
        # Use reasonable settings
        optimizer.set_gcmma_options(max_eval=50, x_tol=1e-3, f_tol=1e-5)
        
        # Initial guess: uniform at volume fraction
        x0 = np.array([0.4, 0.4, 0.4])
        
        # Optimize
        x_opt, f_opt = optimizer.optimize(x0)
        
        # Check basic properties that should always hold
        # 1. Densities should be within bounds
        assert np.all(x_opt >= -1e-3), f"Min density: {np.min(x_opt)}"
        assert np.all(x_opt <= 1.0 + 1e-3), f"Max density: {np.max(x_opt)}"
        
        # 2. Volume constraint should be approximately satisfied
        # (MMA may have some tolerance for constraint violations)
        mean_density = np.mean(x_opt)
        constraint_violation = abs(mean_density - 0.4)
        assert constraint_violation < 0.1, f"Volume constraint violation: {constraint_violation}"
        
        # 3. The optimization should not have completely failed
        # (finite objective value)
        assert np.isfinite(f_opt), f"Objective value is not finite: {f_opt}"
        
        # 4. At least the optimization ran successfully
        assert len(x_opt) == 3, "Solution should have correct dimensions"


class TestPerformanceFeatures:
    """Test performance-related features of the optimizers."""
    
    def test_memory_management(self):
        """Test that memory management features work correctly."""
        
        def objective(x):
            # Create some JAX operations that might accumulate memory
            temp = jnp.exp(x)
            temp = jnp.sin(temp)
            return jnp.sum(temp**2)
        
        optimizer = NLoptJAXOptimizer(
            algorithm=nlopt.LD_LBFGS,
            n_vars=3,
            objective_fn=objective,
            memory_cleanup_freq=2  # Frequent cleanup for testing
        )
        
        # Patch the _periodic_cleanup method directly
        original_cleanup = optimizer._periodic_cleanup
        cleanup_count = {'count': 0}
        
        def mock_cleanup():
            cleanup_count['count'] += 1
            original_cleanup()
        
        optimizer._periodic_cleanup = mock_cleanup
        
        optimizer.set_lower_bounds(np.array([-1.0, -1.0, -1.0]))
        optimizer.set_upper_bounds(np.array([1.0, 1.0, 1.0]))
        optimizer.set_maxeval(10)  # Run for a few iterations
        optimizer.set_ftol_rel(1e-4)  # Relax tolerance for faster convergence
        
        x0 = np.array([0.5, 0.5, 0.5])
        optimizer.optimize(x0)
        
        # Check that cleanup was called
        # With memory_cleanup_freq=2, it should be called at least once
        assert cleanup_count['count'] >= 1, f"Cleanup called {cleanup_count['count']} times"
    
    @pytest.mark.skip(reason="Requires actual performance measurement")
    def test_cpu_performance_scaling(self):
        """Test CPU performance scaling with problem size."""
        # This would be a more comprehensive test measuring actual
        # performance scaling, but requires longer execution times
        pass


if __name__ == "__main__":
    pytest.main([__file__])