"""Helmholtz filter implementation for design variable-based topology optimization.

This module provides Helmholtz filtering capabilities for smoothing design
variables in topology optimization problems, preventing checkerboard patterns
and ensuring mesh-independent designs.
"""

import jax.numpy as np
from typing import Callable, List, Optional, Tuple, Union
from unitcellax.fem.problem import Problem
from unitcellax.pbc import prolongation_matrix
from unitcellax.fem.solver import ad_wrapper
from unitcellax.fem import logger

class Helmholtz(Problem):
    """Helmholtz equation problem for design variable filtering.
    
    Implements the Helmholtz PDE filter: -r²∇²x̃ + x̃ = x
    where x is the original design variable field and x̃ is the filtered design variable.
    Supports both scalar and vector-valued design variable fields.
    
    Attributes:
        radius (float): Filter radius controlling the smoothing length scale.
        num_components (int): Number of components in the design variable field (1 for scalar).
    """
    
    def custom_init(self, radius: float) -> None:
        """Initialize Helmholtz filter with specified radius.
        
        Args:
            radius (float): Filter radius parameter controlling smoothing.
        """
        self.radius = radius
        # Determine number of components from vec parameter
        self.num_components = self.vec[0] if isinstance(self.vec, list) else self.vec
        
    def get_tensor_map(self) -> Callable:
        """Get the diffusion tensor mapping for the Helmholtz equation.
        
        Returns:
            Callable: Function mapping gradient and design variable to diffusion term.
        """
        def diffusion(u_grad: np.ndarray, design_variable: np.ndarray) -> np.ndarray:
            """Compute diffusion term r²∇u.
            
            For vector fields, applies the same diffusion to each component independently.
            
            Args:
                u_grad (np.ndarray): Gradient of solution field. Shape depends on field type:
                    - Scalar: (num_quads, num_dims)
                    - Vector: (num_quads, num_components, num_dims)
                design_variable (np.ndarray): Design variable field (unused in linear diffusion).
                
            Returns:
                np.ndarray: Scaled gradient r²∇u with same shape as u_grad.
            """
            return self.radius**2 * u_grad
        return diffusion
        
    def get_mass_map(self) -> Callable:
        """Get the mass term mapping for the Helmholtz equation.
        
        Returns:
            Callable: Function mapping solution, position, and design variable to mass term.
        """
        def mass_term(u: np.ndarray, x: np.ndarray, design_variable: np.ndarray) -> np.ndarray:
            """Compute mass term u - x (design variable).
            
            For vector fields, computes the difference for each component independently.
            
            Args:
                u (np.ndarray): Solution field (filtered design variable). Shape:
                    - Scalar: (num_quads,)
                    - Vector: (num_quads, num_components)
                x (np.ndarray): Spatial position (unused).
                design_variable (np.ndarray): Original design variable field with same shape as u.
                
            Returns:
                np.ndarray: Mass term u - x for Helmholtz equation with same shape as inputs.
            """
            return u - design_variable
        return mass_term
    def set_params(self, params: np.ndarray) -> None:
        """Set design variable parameters and interpolate to quadrature points.
        
        Handles both scalar and vector-valued design variable fields.
        
        Args:
            params (np.ndarray): Nodal design variable values. Shape:
                - Scalar field: (num_nodes,)
                - Vector field: (num_nodes * num_components,) in component-major order
            
        Raises:
            ValueError: If number of parameters doesn't match expected size.
        """
        expected_size = self.fes[0].num_total_nodes * self.num_components
        if len(params) != expected_size:
            raise ValueError(f"Number of parameters {len(params)} does not match expected size {expected_size} "
                           f"(num_nodes={self.fes[0].num_total_nodes}, num_components={self.num_components}).")

        self.full_params = params

        if self.num_components == 1:
            # Scalar field - original implementation
            design_variable_cells = params[self.fes[0].cells]  # (num_cells, num_nodes)
            shape_vals = self.fes[0].shape_vals   # (num_quads, num_nodes)
            design_variable_quads = np.einsum("qn,cn->cq", shape_vals, design_variable_cells)  # (num_cells, num_quads)
        else:
            # Vector field - reshape and interpolate each component
            params_reshaped = params.reshape(self.num_components, -1)  # (num_components, num_nodes)
            design_variable_quads_list = []
            
            for comp in range(self.num_components):
                design_variable_cells_comp = params_reshaped[comp][self.fes[0].cells]  # (num_cells, num_nodes)
                shape_vals = self.fes[0].shape_vals   # (num_quads, num_nodes)
                design_variable_quads_comp = np.einsum("qn,cn->cq", shape_vals, design_variable_cells_comp)  # (num_cells, num_quads)
                design_variable_quads_list.append(design_variable_quads_comp)
            
            # Stack components: (num_cells, num_quads, num_components)
            design_variable_quads = np.stack(design_variable_quads_list, axis=-1)

        self.internal_vars = [design_variable_quads]

class HelmholtzFilter:
    """Helmholtz filter for topology optimization.
    
    Applies Helmholtz PDE filtering to design variables to ensure smooth,
    mesh-independent designs in topology optimization. The filter solves:
    -r²∇²x̃ + x̃ = x
    
    Supports both scalar and vector-valued design variable fields. For vector fields,
    each component is filtered independently using the same filter radius.
    
    Attributes:
        vec (Union[int, List[int]]): Solution vector specification (number of components).
        fwd_pred (Callable): Forward prediction function with automatic differentiation.
        fe: Finite element object from the Helmholtz problem.
        num_components (int): Number of components in the design variable field.
    """
    
    def __init__(self, unitcell, vec: Union[int, List[int]], radius: float = 0.05, 
                 prolongation_matrix: Optional[np.ndarray] = None) -> None:
        """Initialize Helmholtz filter.
        
        Args:
            unitcell: Unit cell object containing mesh and element information.
            vec (Union[int, List[int]]): Number of components in the design variable field.
                Use 1 for scalar fields, or higher values for vector fields.
            radius (float): Filter radius controlling smoothing length scale. Default is 0.05.
            prolongation_matrix (Optional[np.ndarray]): Optional prolongation matrix for 
                periodic boundary conditions. Default is None.
        """
        dim = unitcell.num_dims
        self.vec = vec
        self.num_components = vec if isinstance(vec, int) else vec[0]
        ele_type = unitcell.ele_type
        if prolongation_matrix is not None:
            P = prolongation_matrix
        else:
            P = None
        problem = Helmholtz(unitcell.mesh, vec=self.vec, dim=dim, 
                                 ele_type=ele_type, prolongation_matrix=P, 
                                 additional_info=[radius])
        self.fwd_pred = ad_wrapper(problem, solver_options={'jax_solver': {}}, 
                                   adjoint_solver_options={'jax_solver': {}})
        self.fe = problem.fe
        
    def filtered(self, params: np.ndarray) -> np.ndarray:
        """Apply Helmholtz filter to design variable parameters.
        
        For vector fields, each component is filtered independently with the same radius.
        
        Args:
            params (np.ndarray): Nodal design variable values to be filtered. Shape:
                - Scalar field: (num_nodes,)
                - Vector field: (num_nodes * num_components,) in component-major order
            
        Returns:
            np.ndarray: Filtered design variable values with same shape as input.
        """
        logger.debug(f"Applying Helmholtz filter to {self.num_components} component(s)")
        sol = self.fwd_pred(params)[0]
        
        # For scalar fields, squeeze the extra dimension
        if self.num_components == 1:
            sol = np.squeeze(sol)
        else:
            # For vector fields, reshape back to original format
            sol = sol.reshape(-1)
            
        return sol