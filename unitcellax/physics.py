"""Built-in physics implementation for linear elasticity problems.

This module provides a ready-to-use physics implementation for linear elastic
solid mechanics. The LinearElasticity class inherits from the Problem base class
and implements the necessary methods for weak form computation.

Key Physics Implementation:
    LinearElasticity: Linear elastic solid mechanics with support for
                     topology optimization and heterogeneous materials

Example:
    >>> from unitcellax.physics import LinearElasticity
    >>> from unitcellax.fem.mesh import box_mesh
    >>> 
    >>> # Create mesh and setup elasticity problem
    >>> mesh = box_mesh(10, 10, 10, 1.0, 1.0, 1.0)
    >>> problem = LinearElasticity(
    ...     mesh=mesh,
    ...     E=70e3,  # Young's modulus
    ...     nu=0.3   # Poisson's ratio
    ... )
"""

import jax.numpy as np
from typing import Callable, Optional, Union, List, Any
from unitcellax.fem.problem import Problem, DirichletBC


class LinearElasticity(Problem):
    """Linear elasticity problem for solid mechanics.
    
    Solves the linear elasticity equations:
        -∇·σ = f  in Ω
        σ = λ tr(ε) I + 2μ ε
        ε = 0.5 * (∇u + ∇u^T)
    
    where σ is stress, ε is strain, u is displacement, and λ, μ are Lamé parameters.
    
    Args:
        mesh: Computational mesh
        E (float): Young's modulus
        nu (float): Poisson's ratio
        E_min (float): Minimum Young's modulus for topology optimization
        density_field (Optional[np.ndarray]): Material density field for topology optimization
        penalty (float): SIMP penalty exponent for topology optimization
        body_force (Optional[Callable]): Body force function f(x) -> force vector
        **kwargs: Additional arguments passed to Problem constructor
        
    Example:
        >>> # Simple cantilever beam
        >>> mesh = box_mesh(20, 10, 1, 2.0, 1.0, 0.1)
        >>> bcs = [DirichletBC(lambda x: x[0] < 1e-6, 0, lambda x: 0.0),  # Fix left edge
        ...        DirichletBC(lambda x: x[0] < 1e-6, 1, lambda x: 0.0)]
        >>> problem = LinearElasticity(mesh=mesh, E=210e9, nu=0.3, 
        ...                           vec=2, dim=2, dirichlet_bcs=bcs)
    """
    
    def __init__(
        self,
        mesh,
        E: float,
        nu: float,
        E_min: float = 1e-8,
        density_field: Optional[np.ndarray] = None,
        penalty: float = 3.0,
        body_force: Optional[Callable] = None,
        **kwargs
    ):
        self.E = E
        self.nu = nu
        self.E_min = E_min
        self.penalty = penalty
        self.body_force = body_force
        self.density_field = density_field
        
        # Compute Lamé parameters
        self.mu = E / (2.0 * (1.0 + nu))
        self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        
        # Default to 3D if not specified
        if 'vec' not in kwargs:
            kwargs['vec'] = 3
        if 'dim' not in kwargs:
            kwargs['dim'] = 3
            
        super().__init__(mesh=mesh, **kwargs)
        
    def get_tensor_map(self) -> Callable:
        """Get the stress-strain relationship for linear elasticity.
        
        Returns:
            Callable: Function mapping strain to stress tensor
        """
        def stress(u_grad, rho):
            # Apply SIMP penalty if density field is provided
            if self.density_field is not None:
                E_effective = self.E_min + (self.E - self.E_min) * rho ** self.penalty
                mu = E_effective / (2.0 * (1.0 + self.nu))
                lmbda = E_effective * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
            else:
                mu = self.mu
                lmbda = self.lmbda
                
            # Compute strain tensor: ε = 0.5 * (∇u + ∇u^T)
            epsilon = 0.5 * (u_grad + u_grad.T)
            
            # Compute stress tensor: σ = λ tr(ε) I + 2μ ε
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
            
            return sigma
            
        return stress
        
    def get_mass_map(self) -> Optional[Callable]:
        """Get the body force term if specified.
        
        Returns:
            Optional[Callable]: Body force function or None
        """
        if self.body_force is not None:
            def mass_map(u, x):
                return -self.body_force(x)
            return mass_map
        return None
        
    def set_params(self, params: np.ndarray) -> None:
        """Update material density field for topology optimization.
        
        Args:
            params (np.ndarray): Material density values at nodes
        """
        if len(params) != self.fes[0].num_total_nodes:
            raise ValueError(
                f"Number of parameters {len(params)} does not match "
                f"number of nodes {self.fes[0].num_total_nodes}."
            )
            
        self.full_params = params
        self.density_field = params
        
        # Interpolate density to quadrature points
        rho_cells = params[self.fes[0].cells]  # (num_cells, num_nodes)
        shape_vals = self.fes[0].shape_vals     # (num_quads, num_nodes)
        rho_quads = np.einsum("qn,cn->cq", shape_vals, rho_cells)  # (num_cells, num_quads)
        
        self.internal_vars = [rho_quads]
        
    def compliance(self, u_sol: np.ndarray, rho: Optional[np.ndarray] = None) -> float:
        """Compute structural compliance (strain energy).
        
        Compliance is the strain energy of the structure and serves as a common
        objective function in topology optimization (minimize compliance = maximize stiffness).
        
        Args:
            u_sol (np.ndarray): Displacement solution field
            rho (Optional[np.ndarray]): Density field. If None, uses current density_field
            
        Returns:
            float: Compliance value (strain energy)
            
        Example:
            >>> # After solving the forward problem
            >>> u_solution = solver.solve(problem)
            >>> compliance_value = problem.compliance(u_solution)
            >>> print(f"Structural compliance: {compliance_value:.6e}")
        """
        import jax
        
        fe = self.fe
        u_cells = u_sol[fe.cells]
        shape_grads = fe.shape_grads
        JxW = fe.JxW
        
        # Compute displacement gradients at quadrature points
        u_grad = np.einsum("cqni,cnj->cqij", shape_grads, u_cells)
        strain = 0.5 * (u_grad + np.transpose(u_grad, axes=(0, 1, 3, 2)))
        
        # Use provided density or current density field
        if rho is not None:
            # Interpolate density to quadrature points
            rho_cells = rho[fe.cells]
            shape_vals = fe.shape_vals
            rho_quads = np.einsum("qv,cv->cq", shape_vals, rho_cells)
        else:
            # Use current internal density field if available
            if hasattr(self, 'internal_vars') and self.internal_vars:
                rho_quads = self.internal_vars[0]  # Already interpolated to quad points
            else:
                # No density field - use uniform material
                rho_quads = np.ones((fe.cells.shape[0], fe.shape_vals.shape[0]))
        
        # Compute stress using the tensor map
        stress_fn = self.get_tensor_map()
        stress_fn_vmap = jax.vmap(jax.vmap(stress_fn, in_axes=(0, 0)), in_axes=(0, 0))
        sigma = stress_fn_vmap(u_grad, rho_quads)
        
        # Compute strain energy density and integrate
        energy_density = np.einsum("cqij,cqij->cq", sigma, strain)
        compliance = np.sum(energy_density * JxW)
        
        return compliance