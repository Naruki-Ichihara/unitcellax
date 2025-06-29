"""Simple unit tests for Helmholtz filter implementation.

This module provides basic tests for the Helmholtz filter that focus on
testing the core logic without complex mesh setup.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, MagicMock, patch
from unitcellax.filters import Helmholtz, HelmholtzFilter


class TestHelmholtzProblem:
    """Test suite for Helmholtz problem class."""
    
    def test_helmholtz_initialization(self):
        """Test Helmholtz problem initialization."""
        # Create Helmholtz instance without full initialization to test custom_init
        helmholtz = Helmholtz.__new__(Helmholtz)
        helmholtz.vec = 1
        
        # Test scalar field initialization
        radius = 0.1
        helmholtz.custom_init(radius)
        
        assert helmholtz.radius == radius
        assert helmholtz.num_components == 1
        
    def test_helmholtz_vector_initialization(self):
        """Test Helmholtz problem initialization with vector field."""
        # Create Helmholtz instance without full initialization to test custom_init
        helmholtz = Helmholtz.__new__(Helmholtz)
        helmholtz.vec = 3
        
        # Test vector field initialization
        radius = 0.15
        helmholtz.custom_init(radius)
        
        assert helmholtz.radius == radius
        assert helmholtz.num_components == 3
        
    def test_diffusion_map(self):
        """Test diffusion tensor mapping function."""
        # Create Helmholtz instance without full initialization
        helmholtz = Helmholtz.__new__(Helmholtz)
        radius = 0.2
        helmholtz.radius = radius
        
        # Get diffusion function
        diffusion_fn = helmholtz.get_tensor_map()
        
        # Test with sample gradient
        u_grad = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        design_var = jnp.array([0.5, 0.5])  # Not used in linear diffusion
        
        result = diffusion_fn(u_grad, design_var)
        expected = radius**2 * u_grad
        
        assert jnp.allclose(result, expected)
        
    def test_mass_map(self):
        """Test mass term mapping function."""
        # Create Helmholtz instance without full initialization
        helmholtz = Helmholtz.__new__(Helmholtz)
        
        # Get mass term function
        mass_fn = helmholtz.get_mass_map()
        
        # Test with sample values
        u = jnp.array([0.8, 0.9, 0.7])
        x = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # Not used
        design_var = jnp.array([0.5, 0.6, 0.4])
        
        result = mass_fn(u, x, design_var)
        expected = u - design_var
        
        assert jnp.allclose(result, expected)
        
    def test_set_params_scalar(self):
        """Test set_params for scalar field."""
        # Create mock finite element
        mock_fe = Mock()
        mock_fe.num_total_nodes = 10
        mock_fe.cells = jnp.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
        mock_fe.shape_vals = jnp.array([[0.25, 0.5, 0.25], 
                                        [0.33, 0.34, 0.33]])
        
        # Create Helmholtz instance with mocked FE
        helmholtz = Helmholtz.__new__(Helmholtz)
        helmholtz.num_components = 1
        helmholtz.fes = [mock_fe]
        
        # Test parameters
        params = jnp.arange(10) * 0.1
        
        # Call set_params
        helmholtz.set_params(params)
        
        assert helmholtz.full_params is not None
        assert len(helmholtz.internal_vars) == 1
        assert helmholtz.internal_vars[0].shape == (3, 2)  # (num_cells, num_quads)
        
    def test_set_params_vector(self):
        """Test set_params for vector field."""
        # Create mock finite element
        mock_fe = Mock()
        mock_fe.num_total_nodes = 5
        mock_fe.cells = jnp.array([[0, 1, 2], [1, 2, 3]])
        mock_fe.shape_vals = jnp.array([[0.33, 0.33, 0.34]])
        
        # Create Helmholtz instance with mocked FE
        helmholtz = Helmholtz.__new__(Helmholtz)
        helmholtz.num_components = 2  # 2-component vector
        helmholtz.fes = [mock_fe]
        
        # Test parameters (5 nodes * 2 components)
        params = jnp.arange(10) * 0.1
        
        # Call set_params
        helmholtz.set_params(params)
        
        assert helmholtz.full_params is not None
        assert len(helmholtz.internal_vars) == 1
        assert helmholtz.internal_vars[0].shape == (2, 1, 2)  # (num_cells, num_quads, num_components)
        
    def test_set_params_validation(self):
        """Test parameter validation in set_params."""
        # Create mock finite element
        mock_fe = Mock()
        mock_fe.num_total_nodes = 10
        
        # Create Helmholtz instance
        helmholtz = Helmholtz.__new__(Helmholtz)
        helmholtz.num_components = 1
        helmholtz.fes = [mock_fe]
        
        # Test with wrong size
        wrong_params = jnp.zeros(5)  # Should be 10
        
        with pytest.raises(ValueError, match="Number of parameters"):
            helmholtz.set_params(wrong_params)
            

class TestHelmholtzFilterUnit:
    """Unit tests for HelmholtzFilter class."""
    
    def test_filter_initialization(self):
        """Test HelmholtzFilter initialization."""
        # Create mock unit cell
        mock_unitcell = Mock()
        mock_unitcell.num_dims = 2
        mock_unitcell.ele_type = 'QUAD4'
        mock_unitcell.mesh = Mock()
        
        # Test initialization
        vec = 1
        radius = 0.1
        
        # Mock the problem creation and ad_wrapper
        with patch('unitcellax.filters.Helmholtz') as mock_helmholtz_class:
            with patch('unitcellax.filters.ad_wrapper') as mock_ad_wrapper:
                mock_problem = Mock()
                mock_problem.fe = Mock()
                mock_helmholtz_class.return_value = mock_problem
                
                mock_fwd_pred = Mock()
                mock_ad_wrapper.return_value = mock_fwd_pred
                
                # Create filter
                helmholtz_filter = HelmholtzFilter(
                    unitcell=mock_unitcell,
                    vec=vec,
                    radius=radius
                )
                
                assert helmholtz_filter.vec == vec
                assert helmholtz_filter.num_components == 1
                assert helmholtz_filter.fwd_pred == mock_fwd_pred
                assert helmholtz_filter.fe == mock_problem.fe
                
    def test_filter_vector_field(self):
        """Test HelmholtzFilter with vector field."""
        # Create mock unit cell
        mock_unitcell = Mock()
        mock_unitcell.num_dims = 3
        mock_unitcell.ele_type = 'HEX8'
        mock_unitcell.mesh = Mock()
        
        # Test initialization with vector field
        vec = 3
        radius = 0.05
        
        # Mock the problem creation and ad_wrapper
        with patch('unitcellax.filters.Helmholtz') as mock_helmholtz_class:
            with patch('unitcellax.filters.ad_wrapper') as mock_ad_wrapper:
                mock_problem = Mock()
                mock_problem.fe = Mock()
                mock_helmholtz_class.return_value = mock_problem
                
                mock_fwd_pred = Mock()
                mock_ad_wrapper.return_value = mock_fwd_pred
                
                # Create filter
                helmholtz_filter = HelmholtzFilter(
                    unitcell=mock_unitcell,
                    vec=vec,
                    radius=radius
                )
                
                assert helmholtz_filter.vec == vec
                assert helmholtz_filter.num_components == 3
                
    def test_filtered_scalar(self):
        """Test filtered method for scalar field."""
        # Create mock unit cell
        mock_unitcell = Mock()
        mock_unitcell.num_dims = 2
        mock_unitcell.ele_type = 'QUAD4'
        mock_unitcell.mesh = Mock()
        
        # Mock the problem creation and ad_wrapper
        with patch('unitcellax.filters.Helmholtz'):
            with patch('unitcellax.filters.ad_wrapper') as mock_ad_wrapper:
                # Mock forward prediction
                mock_fwd_pred = MagicMock()
                test_output = jnp.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
                mock_fwd_pred.return_value = [test_output]
                mock_ad_wrapper.return_value = mock_fwd_pred
                
                # Create filter
                helmholtz_filter = HelmholtzFilter(
                    unitcell=mock_unitcell,
                    vec=1,
                    radius=0.1
                )
                
                # Test filtering
                params = jnp.array([0.0, 0.5, 1.0, 0.5, 0.0])
                result = helmholtz_filter.filtered(params)
                
                # For scalar field, output should be squeezed
                assert result.shape == (5,)
                assert jnp.allclose(result, test_output.squeeze())
                
    def test_filtered_vector(self):
        """Test filtered method for vector field."""
        # Create mock unit cell
        mock_unitcell = Mock()
        mock_unitcell.num_dims = 2
        mock_unitcell.ele_type = 'QUAD4'
        mock_unitcell.mesh = Mock()
        
        # Mock the problem creation and ad_wrapper
        with patch('unitcellax.filters.Helmholtz'):
            with patch('unitcellax.filters.ad_wrapper') as mock_ad_wrapper:
                # Mock forward prediction for 2-component vector
                mock_fwd_pred = MagicMock()
                test_output = jnp.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
                mock_fwd_pred.return_value = [test_output]
                mock_ad_wrapper.return_value = mock_fwd_pred
                
                # Create filter
                helmholtz_filter = HelmholtzFilter(
                    unitcell=mock_unitcell,
                    vec=2,
                    radius=0.1
                )
                helmholtz_filter.num_components = 2  # Ensure this is set
                
                # Test filtering
                params = jnp.array([0.0, 0.5, 1.0, 1.0, 0.5, 0.0])  # 3 nodes * 2 components
                result = helmholtz_filter.filtered(params)
                
                # For vector field, output should be reshaped to 1D
                assert result.shape == (6,)
                assert jnp.allclose(result, test_output.reshape(-1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])