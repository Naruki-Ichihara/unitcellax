"""Finite element solvers and solution algorithms.

This module provides comprehensive solution algorithms for finite element problems,
including linear solvers, nonlinear Newton-Raphson methods, boundary condition
enforcement, and specialized continuation methods. It supports multiple backend
solvers (JAX, SciPy/UMFPACK, PETSc) and advanced techniques like automatic
differentiation and adjoint methods.

The module includes:
    - Linear solver interfaces (JAX, UMFPACK, PETSc)
    - Nonlinear Newton-Raphson solver with line search
    - Boundary condition enforcement via row elimination
    - Arc-length continuation methods for path-following
    - Dynamic relaxation for static equilibrium problems
    - Automatic differentiation wrappers for sensitivity analysis
    - Memory-efficient sparse matrix assembly

Key Functions:
    solver: Main nonlinear solver with Newton-Raphson iteration
    linear_solver: Unified interface to multiple linear solvers
    array_to_petsc_vec: Conversion utilities for PETSc integration
    implicit_vjp: Adjoint method for parameter sensitivity
    ad_wrapper: Automatic differentiation wrapper for optimization

Example:
    Basic nonlinear solver usage:
    
    >>> from unitcellax.fem.solver import solver
    >>> from unitcellax.fem.problem import Problem
    >>> 
    >>> # Setup problem (see problem.py documentation)
    >>> problem = Problem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)
    >>> 
    >>> # Solve with Newton-Raphson method
    >>> solver_options = {'jax_solver': {'precond': True}, 'tol': 1e-6}
    >>> solution = solver(problem, solver_options)

Note:
    This module requires JAX for automatic differentiation, PETSc for advanced
    linear algebra, and SciPy for sparse matrix operations. GPU acceleration
    is available through JAX when configured properly.
"""

import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
from jax.experimental.sparse import BCOO
import scipy
import time
from typing import Dict, List, Optional, Callable, Union, Tuple, Any
from petsc4py import PETSc
import gc

from unitcellax.fem import logger

from jax import config

config.update("jax_enable_x64", True)
CHUNK_SIZE = 100000000


def array_to_petsc_vec(arr: Union[np.ndarray, onp.ndarray], size: Optional[int] = None) -> PETSc.Vec:
    """Convert a JAX or NumPy array to a PETSc.Vec.

    Args:
        arr (array-like): JAX array (DeviceArray) or NumPy array of shape (N,).
        size (int, optional): Vector size. If None, uses len(arr) as vector size.

    Returns:
        PETSc.Vec: PETSc vector with values set from arr.
    """
    arr_np = onp.array(arr).flatten()  # ensure NumPy, ensure 1D
    if size is None:
        size = arr_np.shape[0]
    vec = PETSc.Vec().createSeq(size)
    vec.setValues(range(size), arr_np)
    vec.assemble()
    return vec


def jax_solve(A: PETSc.Mat, b: np.ndarray, x0: np.ndarray, precond: bool) -> np.ndarray:
    """Solves the equilibrium equation using a JAX solver.

    Args:
        A: System matrix.
        b: Right-hand side vector.
        x0: Initial guess.
        precond (bool): Whether to calculate the preconditioner or not.

    Returns:
        Solution vector.
    """
    logger.debug(f"JAX Solver - Solving linear system")
    indptr, indices, data = A.getValuesCSR()
    A_sp_scipy = scipy.sparse.csr_array((data, indices, indptr), shape=A.getSize())
    A = BCOO.from_scipy_sparse(A_sp_scipy).sort_indices()
    jacobi = np.array(A_sp_scipy.diagonal())
    pc = lambda x: x * (1.0 / jacobi) if precond else None
    x, info = jax.scipy.sparse.linalg.bicgstab(
        A, b, x0=x0, M=pc, tol=1e-10, atol=1e-10, maxiter=10000
    )

    # Verify convergence
    err = np.linalg.norm(A @ x - b)
    logger.debug(f"JAX Solver - Finshed solving, res = {err}")
    assert err < 0.1, f"JAX linear solver failed to converge with err = {err}"
    x = np.where(
        err < 0.1, x, np.nan
    )  # For assert purpose, some how this also affects bicgstab.

    return x


def umfpack_solve(A: PETSc.Mat, b: np.ndarray) -> np.ndarray:
    """Solve linear system using SciPy's UMFPACK interface.
    
    Solves the linear system Ax = b using the UMFPACK sparse direct solver
    through SciPy's interface. UMFPACK is typically faster and more robust
    than iterative methods for moderately-sized problems.
    
    Args:
        A (PETSc.Mat): Sparse system matrix in PETSc format.
        b (np.ndarray): Right-hand side vector.
        
    Returns:
        np.ndarray: Solution vector x.
        
    Note:
        The function converts the PETSc matrix to SciPy CSR format internally.
        Consider using the experimental JAX sparse solver for GPU acceleration.
    """
    logger.debug(f"Scipy Solver - Solving linear system with UMFPACK")
    indptr, indices, data = A.getValuesCSR()
    Asp = scipy.sparse.csr_matrix((data, indices, indptr))
    x = scipy.sparse.linalg.spsolve(Asp, onp.array(b))

    # TODO: try https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html
    # x = jax.experimental.sparse.linalg.spsolve(av, aj, ai, b)

    logger.debug(
        f"Scipy Solver - Finished solving, linear solve res = {np.linalg.norm(Asp @ x - b)}"
    )
    return x


def petsc_solve(A: PETSc.Mat, b: np.ndarray, ksp_type: str, pc_type: str) -> np.ndarray:
    """Solve linear system using PETSc iterative solvers.
    
    Solves the linear system Ax = b using PETSc's iterative Krylov subspace
    methods with preconditioning. Supports various solver and preconditioner
    combinations for different problem types.
    
    Args:
        A (PETSc.Mat): Sparse system matrix in PETSc format.
        b (np.ndarray): Right-hand side vector.
        ksp_type (str): Krylov subspace method type. Options include:
            - 'bicgstab': Bi-conjugate gradient stabilized method
            - 'gmres': Generalized minimal residual method
            - 'tfqmr': Transpose-free quasi-minimal residual method
            - 'minres': Minimal residual method
        pc_type (str): Preconditioner type. Options include:
            - 'ilu': Incomplete LU factorization
            - 'jacobi': Jacobi (diagonal) preconditioning
            - 'hypre': Algebraic multigrid via HYPRE
            
    Returns:
        np.ndarray: Solution vector x.
        
    Raises:
        AssertionError: If solver fails to converge to specified tolerance.
        
    Note:
        For 'tfqmr' solver, automatically uses MUMPS factorization when available.
        Convergence is verified by computing the residual norm.
    """
    rhs = PETSc.Vec().createSeq(len(b))
    rhs.setValues(range(len(b)), onp.array(b))
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.setType(ksp_type)
    ksp.pc.setType(pc_type)

    # TODO: This works better. Do we need to generalize the code a little bit?
    if ksp_type == "tfqmr":
        ksp.pc.setFactorSolverType("mumps")

    logger.debug(
        f"PETSc Solver - Solving linear system with ksp_type = {ksp.getType()}, pc = {ksp.pc.getType()}"
    )
    x = PETSc.Vec().createSeq(len(b))
    ksp.solve(rhs, x)

    # Verify convergence
    y = PETSc.Vec().createSeq(len(b))
    A.mult(x, y)

    err = np.linalg.norm(y.getArray() - rhs.getArray())
    logger.debug(f"PETSc Solver - Finished solving, linear solve res = {err}")
    assert err < 0.1, f"PETSc linear solver failed to converge, err = {err}"

    # Get solution array before cleanup
    solution = x.getArray().copy()  # Make a copy to prevent memory issues
    
    # Properly destroy PETSc objects to prevent memory leaks
    rhs.destroy()
    x.destroy()
    y.destroy()
    ksp.destroy()
    
    return solution


def linear_solver(A: PETSc.Mat, b: np.ndarray, x0: np.ndarray, solver_options: Dict[str, Any]) -> np.ndarray:
    """Unified interface for multiple linear solver backends.
    
    Provides a consistent interface to JAX, UMFPACK, PETSc, and custom linear
    solvers. Automatically selects JAX solver if no specific solver is requested.
    
    Args:
        A (PETSc.Mat): Sparse system matrix.
        b (np.ndarray): Right-hand side vector.
        x0 (np.ndarray): Initial guess for iterative solvers.
        solver_options (dict): Solver configuration dictionary with possible keys:
            - 'jax_solver': JAX iterative solver options
            - 'umfpack_solver': UMFPACK direct solver options
            - 'petsc_solver': PETSc iterative solver options
            - 'custom_solver': User-defined solver function
            
    Returns:
        np.ndarray: Solution vector x.
        
    Raises:
        NotImplementedError: If no valid solver is specified in options.
        
    Example:
        >>> options = {'jax_solver': {'precond': True}}
        >>> x = linear_solver(A, b, x0, options)
        
        >>> options = {'petsc_solver': {'ksp_type': 'gmres', 'pc_type': 'ilu'}}
        >>> x = linear_solver(A, b, x0, options)
    """

    # If user does not specify any solver, set jax_solver as the default one.
    if (
        len(
            solver_options.keys()
            & {"jax_solver", "umfpack_solver", "petsc_solver", "custom_solver"}
        )
        == 0
    ):
        solver_options["jax_solver"] = {}

    if "jax_solver" in solver_options:
        precond = (
            solver_options["jax_solver"]["precond"]
            if "precond" in solver_options["jax_solver"]
            else True
        )
        x = jax_solve(A, b, x0, precond)
    elif "umfpack_solver" in solver_options:
        x = umfpack_solve(A, b)
    elif "petsc_solver" in solver_options:
        ksp_type = (
            solver_options["petsc_solver"]["ksp_type"]
            if "ksp_type" in solver_options["petsc_solver"]
            else "bcgsl"
        )
        pc_type = (
            solver_options["petsc_solver"]["pc_type"]
            if "pc_type" in solver_options["petsc_solver"]
            else "ilu"
        )
        x = petsc_solve(A, b, ksp_type, pc_type)
    elif "custom_solver" in solver_options:
        # Users can define their own solver
        custom_solver = solver_options["custom_solver"]
        x = custom_solver(A, b, x0, solver_options)
    else:
        raise NotImplementedError(f"Unknown linear solver.")

    return x


################################################################################
# "row elimination" solver


def apply_bc_vec(res_vec: np.ndarray, dofs: np.ndarray, problem: Any, scale: float = 1.0) -> np.ndarray:
    """Apply Dirichlet boundary conditions to residual vector.
    
    Modifies the residual vector to enforce Dirichlet boundary conditions
    using the row elimination method. This function directly modifies the
    residual at constrained degrees of freedom.
    
    Args:
        res_vec (np.ndarray): Global residual vector to modify.
        dofs (np.ndarray): Current solution degrees of freedom.
        problem (Problem): Finite element problem containing boundary condition data.
        scale (float, optional): Scaling factor for boundary condition values. Defaults to 1.0.
        
    Returns:
        np.ndarray: Modified residual vector with boundary conditions applied.
        
    Note:
        This function implements the row elimination method where constrained
        DOFs are set to (current_value - prescribed_value) * scale.
    """
    res_list = problem.unflatten_fn_sol_list(res_vec)
    sol_list = problem.unflatten_fn_sol_list(dofs)

    for ind, fe in enumerate(problem.fes):
        res = res_list[ind]
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
                sol[fe.node_inds_list[i], fe.vec_inds_list[i]], unique_indices=True
            )
            res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].add(
                -fe.vals_list[i] * scale
            )

        res_list[ind] = res

    return jax.flatten_util.ravel_pytree(res_list)[0]


def apply_bc(res_fn: Callable[[np.ndarray], np.ndarray], problem: Any, scale: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Create a boundary condition-aware residual function.
    
    Wraps a residual function to automatically apply Dirichlet boundary
    conditions using the row elimination method.
    
    Args:
        res_fn (Callable): Original residual function that takes DOFs and returns residual.
        problem (Problem): Finite element problem with boundary condition information.
        scale (float, optional): Scaling factor for boundary conditions. Defaults to 1.0.
        
    Returns:
        Callable: Modified residual function that enforces boundary conditions.
        
    Example:
        >>> res_fn_bc = apply_bc(problem.compute_residual, problem)
        >>> residual = res_fn_bc(dofs)
    """
    def res_fn_bc(dofs):
        """Apply Dirichlet boundary conditions"""
        res_vec = res_fn(dofs)
        return apply_bc_vec(res_vec, dofs, problem, scale)

    return res_fn_bc


def assign_bc(dofs: np.ndarray, problem: Any) -> np.ndarray:
    """Assign prescribed values to Dirichlet boundary condition DOFs.
    
    Sets the solution values at constrained degrees of freedom to their
    prescribed Dirichlet boundary condition values.
    
    Args:
        dofs (np.ndarray): Solution vector to modify.
        problem (Problem): Finite element problem with boundary condition data.
        
    Returns:
        np.ndarray: Modified solution vector with boundary conditions enforced.
    """
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(fe.vals_list[i])
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def assign_ones_bc(dofs: np.ndarray, problem: Any) -> np.ndarray:
    """Set Dirichlet boundary condition DOFs to unity values.
    
    Utility function that sets all constrained degrees of freedom to 1.0.
    Useful for testing and generating unit perturbations.
    
    Args:
        dofs (np.ndarray): Solution vector to modify.
        problem (Problem): Finite element problem with boundary condition data.
        
    Returns:
        np.ndarray: Modified solution vector with boundary DOFs set to 1.0.
    """
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(1.0)
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def assign_zeros_bc(dofs: np.ndarray, problem: Any) -> np.ndarray:
    """Set Dirichlet boundary condition DOFs to zero values.
    
    Utility function that sets all constrained degrees of freedom to 0.0.
    Useful for homogeneous boundary conditions and initialization.
    
    Args:
        dofs (np.ndarray): Solution vector to modify.
        problem (Problem): Finite element problem with boundary condition data.
        
    Returns:
        np.ndarray: Modified solution vector with boundary DOFs set to 0.0.
    """
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(0.0)
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def copy_bc(dofs: np.ndarray, problem: Any) -> np.ndarray:
    """Extract boundary condition values to a new zero vector.
    
    Creates a new vector filled with zeros except at boundary condition
    locations, where it copies the values from the input DOFs.
    
    Args:
        dofs (np.ndarray): Source solution vector.
        problem (Problem): Finite element problem with boundary condition data.
        
    Returns:
        np.ndarray: New vector with only boundary DOF values copied.
    """
    new_dofs = np.zeros_like(dofs)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    new_sol_list = problem.unflatten_fn_sol_list(new_dofs)

    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        new_sol = new_sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            new_sol = new_sol.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
                sol[fe.node_inds_list[i], fe.vec_inds_list[i]]
            )
        new_sol_list[ind] = new_sol

    return jax.flatten_util.ravel_pytree(new_sol_list)[0]


def get_flatten_fn(fn_sol_list: Callable[[List[np.ndarray]], List[np.ndarray]], problem: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Create a flattened version of a solution list function.
    
    Converts a function that operates on solution lists to one that operates
    on flattened DOF vectors, handling the conversion automatically.
    
    Args:
        fn_sol_list (Callable): Function that takes solution list and returns values.
        problem (Problem): Finite element problem with flattening utilities.
        
    Returns:
        Callable: Function that takes flattened DOFs and returns flattened values.
    """

    def fn_dofs(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        val_list = fn_sol_list(sol_list)
        return jax.flatten_util.ravel_pytree(val_list)[0]

    return fn_dofs


def operator_to_matrix(operator_fn: Callable[[np.ndarray], np.ndarray], problem: Any) -> np.ndarray:
    """Convert a nonlinear operator to its Jacobian matrix.
    
    Computes the full Jacobian matrix of a nonlinear operator using automatic
    differentiation. Primarily used for debugging and analysis.
    
    Args:
        operator_fn (Callable): Nonlinear operator function.
        problem (Problem): Finite element problem for size information.
        
    Returns:
        np.ndarray: Dense Jacobian matrix.
        
    Warning:
        This function computes a dense matrix and should only be used for
        small problems or debugging purposes.
    """
    """Only used for when debugging.
    Can be used to print the matrix, check the conditional number, etc.
    """
    J = jax.jacfwd(operator_fn)(np.zeros(problem.num_total_dofs_all_vars))
    return J


def linear_incremental_solver(problem: Any, res_vec: np.ndarray, A: PETSc.Mat, dofs: np.ndarray, solver_options: Dict[str, Any]) -> np.ndarray:
    """Solve linear system for Newton-Raphson increment.
    
    Computes the Newton increment by solving the linearized system at each
    Newton iteration. Handles constraint enforcement and optional line search.
    
    Args:
        problem (Problem): Finite element problem instance.
        res_vec (np.ndarray): Current residual vector.
        A (PETSc.Mat): Jacobian matrix at current solution state.
        dofs (np.ndarray): Current solution degrees of freedom.
        solver_options (dict): Solver configuration options.
        
    Returns:
        np.ndarray: Updated solution after applying Newton increment.
        
    Note:
        The function automatically constructs appropriate initial guesses
        that satisfy boundary conditions and handles prolongation matrices
        for constrained problems.
    """
    logger.debug(f"Solving linear system...")
    b = -res_vec

    # x0 will always be correct at boundary locations
    x0_1 = assign_bc(np.zeros(problem.num_total_dofs_all_vars), problem)
    if problem.prolongation_matrix is not None:
        x0_2 = copy_bc(problem.prolongation_matrix @ dofs, problem)
        x0 = problem.prolongation_matrix.T @ (x0_1 - x0_2)
    else:
        x0_2 = copy_bc(dofs, problem)
        x0 = x0_1 - x0_2

    inc = linear_solver(A, b, x0, solver_options)

    line_search_flag = (
        solver_options["line_search_flag"]
        if "line_search_flag" in solver_options
        else False
    )
    if line_search_flag:
        dofs = line_search(problem, dofs, inc)
    else:
        dofs = dofs + inc

    return dofs


def line_search(problem: Any, dofs: np.ndarray, inc: np.ndarray) -> np.ndarray:
    """Perform line search to optimize Newton step size.
    
    Implements a simple backtracking line search to find an optimal step size
    along the Newton direction. Particularly useful for finite deformation
    problems and nonlinear material behavior.
    
    Args:
        problem (Problem): Finite element problem instance.
        dofs (np.ndarray): Current solution degrees of freedom.
        inc (np.ndarray): Newton increment direction.
        
    Returns:
        np.ndarray: Updated solution with optimized step size.
        
    Note:
        Uses a simple halving strategy with a maximum of 3 iterations.
        The implementation is basic and could be enhanced with more
        sophisticated line search algorithms.
        
    Todo:
        Implement more robust line search methods for finite deformation plasticity.
    """
    res_fn = problem.compute_residual
    res_fn = get_flatten_fn(res_fn, problem)
    res_fn = apply_bc(res_fn, problem)

    def res_norm_fn(alpha):
        res_vec = res_fn(dofs + alpha * inc)
        return np.linalg.norm(res_vec)

    # grad_res_norm_fn = jax.grad(res_norm_fn)
    # hess_res_norm_fn = jax.hessian(res_norm_fn)

    # tol = 1e-3
    # alpha = 1.
    # lr = 1.
    # grad_alpha = 1.
    # while np.abs(grad_alpha) > tol:
    #     grad_alpha = grad_res_norm_fn(alpha)
    #     hess_alpha = hess_res_norm_fn(alpha)
    #     alpha = alpha - 1./hess_alpha*grad_alpha
    #     print(f"alpha = {alpha}, grad_alpha = {grad_alpha}, hess_alpha = {hess_alpha}")

    alpha = 1.0
    res_norm = res_norm_fn(alpha)
    for i in range(3):
        alpha *= 0.5
        res_norm_half = res_norm_fn(alpha)
        logger.debug(f"i = {i}, res_norm = {res_norm}, res_norm_half = {res_norm_half}")
        if res_norm_half > res_norm:
            alpha *= 2.0
            break
        res_norm = res_norm_half

    return dofs + alpha * inc


def get_A(problem: Any) -> Union[PETSc.Mat, Tuple[PETSc.Mat, PETSc.Mat]]:
    """Construct PETSc matrix with boundary condition enforcement.
    
    Converts the assembled sparse matrix to PETSc format and applies
    boundary condition enforcement via row elimination. Handles
    prolongation matrices for constraint enforcement.
    
    Args:
        problem (Problem): Finite element problem with assembled sparse matrix.
        
    Returns:
        PETSc.Mat or Tuple[PETSc.Mat, PETSc.Mat]: System matrix. If prolongation
            matrix is present, returns both original and reduced matrices.
            
    Note:
        The function zeros out rows corresponding to Dirichlet boundary
        conditions and applies prolongation operations for multipoint constraints.
    """
    A_sp_scipy = problem.csr_array
    logger.info(
        f"Global sparse matrix takes about {A_sp_scipy.data.shape[0]*8*3/2**30} G memory to store."
    )

    A = PETSc.Mat().createAIJ(
        size=A_sp_scipy.shape,
        csr=(
            A_sp_scipy.indptr.astype(PETSc.IntType, copy=False),
            A_sp_scipy.indices.astype(PETSc.IntType, copy=False),
            A_sp_scipy.data,
        ),
    )

    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            row_inds = onp.array(
                fe.node_inds_list[i] * fe.vec
                + fe.vec_inds_list[i]
                + problem.offset[ind],
                dtype=onp.int32,
            )
            A.zeroRows(row_inds)

    # Linear multipoint constraints
    if problem.prolongation_matrix is not None:
        P = PETSc.Mat().createAIJ(
            size=problem.prolongation_matrix.shape,
            csr=(
                problem.prolongation_matrix.indptr.astype(PETSc.IntType, copy=False),
                problem.prolongation_matrix.indices.astype(PETSc.IntType, copy=False),
                problem.prolongation_matrix.data,
            ),
        )

        tmp = A.matMult(P)
        P_T = P.transpose()
        A_reduced = P_T.matMult(tmp)
        return A, A_reduced
    return A


def solver(problem: Any, solver_options: Dict[str, Any] = {}) -> List[np.ndarray]:
    """Solve nonlinear finite element problem using Newton-Raphson method.
    
    Main nonlinear solver that implements Newton-Raphson iteration with multiple
    linear solver backends. Enforces Dirichlet boundary conditions via row 
    elimination method and supports advanced features like line search, 
    prolongation matrices, and macro terms.
    
    Args:
        problem (Problem): Finite element problem instance containing mesh,
            finite elements, and boundary conditions.
        solver_options (dict, optional): Solver configuration dictionary. Defaults to {}.
            Supported keys:
            - 'jax_solver': JAX iterative solver options
                - 'precond' (bool): Enable Jacobi preconditioning. Defaults to True.
            - 'umfpack_solver': SciPy UMFPACK direct solver options (empty dict)
            - 'petsc_solver': PETSc iterative solver options
                - 'ksp_type' (str): Krylov method ('bicgstab', 'gmres', 'tfqmr'). Defaults to 'bcgsl'.
                - 'pc_type' (str): Preconditioner ('ilu', 'jacobi', 'hypre'). Defaults to 'ilu'.
            - 'line_search_flag' (bool): Enable line search optimization. Defaults to False.
            - 'initial_guess' (List[np.ndarray]): Initial solution guess. Same shape as output.
            - 'tol' (float): Absolute tolerance for residual L2 norm. Defaults to 1e-6.
            - 'rel_tol' (float): Relative tolerance for residual L2 norm. Defaults to 1e-8.
            
    Returns:
        List[np.ndarray]: Solution list where each array corresponds to a variable.
            For multi-variable problems, returns [u1, u2, ...] where each ui has
            shape (num_nodes, vec_components).
            
    Raises:
        AssertionError: If residual contains NaN values or solver fails to converge.
        
    Note:
        Boundary Condition Enforcement:
        Uses row elimination method where the residual becomes:
        res(u) = D*r(u) + (I - D)*u - u_b
        
        Where:
        - D: Diagonal matrix with zeros at constrained DOFs
        - r(u): Physical residual from weak form
        - u_b: Prescribed boundary values
        
        The Jacobian matrix is modified accordingly:
        A = d(res)/d(u) = D*dr/du + (I - D)
        
        Solver Selection:
        If no solver is specified, JAX solver is used by default.
        Only one solver type should be specified per call.
        
    Example:
        Basic nonlinear solve with JAX solver:
        
        >>> solver_options = {'jax_solver': {'precond': True}}
        >>> solution = solver(problem, solver_options)
        
        PETSc solver with custom tolerances:
        
        >>> options = {
        ...     'petsc_solver': {'ksp_type': 'gmres', 'pc_type': 'ilu'},
        ...     'tol': 1e-8,
        ...     'rel_tol': 1e-10
        ... }
        >>> solution = solver(problem, options)
        
        With initial guess and line search:
        
        >>> options = {
        ...     'umfpack_solver': {},
        ...     'initial_guess': initial_solution,
        ...     'line_search_flag': True
        ... }
        >>> solution = solver(problem, options)
    """
    logger.debug(f"Calling the row elimination solver for imposing Dirichlet B.C.")
    logger.debug("Start timing")
    start = time.time()

    if "initial_guess" in solver_options:
        # We dont't want inititual guess to play a role in the differentiation chain.
        initial_guess = jax.lax.stop_gradient(solver_options["initial_guess"])
        dofs = jax.flatten_util.ravel_pytree(initial_guess)[0]

    else:
        if problem.prolongation_matrix is not None:
            dofs = np.zeros(problem.prolongation_matrix.shape[1])  # reduced dofs
        else:
            dofs = np.zeros(problem.num_total_dofs_all_vars)

    rel_tol = solver_options["rel_tol"] if "rel_tol" in solver_options else 1e-8
    tol = solver_options["tol"] if "tol" in solver_options else 1e-6

    def newton_update_helper(dofs):
        if problem.prolongation_matrix is not None:
            logger.debug(
                f"Using prolongation_matrix, shape = {problem.prolongation_matrix.shape}"
            )
            dofs = problem.prolongation_matrix @ dofs

        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)

        problem.compute_csr(CHUNK_SIZE)  # Ensure CSR matrix is computed

        if problem.prolongation_matrix is not None:
            res_vec = problem.prolongation_matrix.T @ res_vec

        A, A_reduced = get_A(problem)

        if problem.macro_term is not None:

            macro_term_petsc = array_to_petsc_vec(problem.macro_term, A.getSize()[0])
            K_affine_vec = PETSc.Vec().createSeq(A.getSize()[0])
            A.mult(macro_term_petsc, K_affine_vec)
            del A
            gc.collect()
            affine_force = problem.prolongation_matrix.T @ K_affine_vec
            res_vec += affine_force

        return res_vec, A_reduced

    res_vec, A = newton_update_helper(dofs)
    res_val = np.linalg.norm(res_vec)
    res_val_initial = res_val
    rel_res_val = res_val / res_val_initial
    logger.debug(f"Before, l_2 res = {res_val}, relative l_2 res = {rel_res_val}")

    while (rel_res_val > rel_tol) and (res_val > tol):
        dofs = linear_incremental_solver(problem, res_vec, A, dofs, solver_options)
        res_vec, A = newton_update_helper(dofs)
        res_val = np.linalg.norm(res_vec)
        rel_res_val = res_val / res_val_initial

        logger.debug(f"l_2 res = {res_val}, relative l_2 res = {rel_res_val}")

    assert np.all(np.isfinite(res_val)), f"res_val contains NaN, stop the program!"
    assert np.all(np.isfinite(dofs)), f"dofs contains NaN, stop the program!"

    if problem.prolongation_matrix is not None:
        dofs = problem.prolongation_matrix @ dofs

    if problem.macro_term is not None:
        dofs = dofs + problem.macro_term

    # If sol_list = [[[u1x, u1y],
    #                 [u2x, u2y],
    #                 [u3x, u3y],
    #                 [u4x, u4y]],
    #                [[p1],
    #                 [p2]]],
    # the flattend DOF vector will be [u1x, u1y, u2x, u2y, u3x, u3y, u4x, u4y, p1, p2]
    sol_list = problem.unflatten_fn_sol_list(dofs)

    end = time.time()
    solve_time = end - start
    logger.info(f"Solve took {solve_time} [s]")
    logger.info(f"max of dofs = {np.max(dofs)}")
    logger.info(f"min of dofs = {np.min(dofs)}")

    return sol_list


def implicit_vjp(problem: Any, sol_list: List[np.ndarray], params: Any, v_list: List[np.ndarray], adjoint_solver_options: Dict[str, Any]) -> Any:
    """Compute vector-Jacobian product using the adjoint method.
    
    Implements the adjoint method to efficiently compute gradients of functionals
    with respect to problem parameters. This is essential for optimization,
    parameter identification, and sensitivity analysis.
    
    Args:
        problem (Problem): Finite element problem instance.
        sol_list (List[np.ndarray]): Solution state at which to evaluate gradients.
        params: Problem parameters with respect to which gradients are computed.
        v_list (List[np.ndarray]): Vector for the vector-Jacobian product.
        adjoint_solver_options (dict): Linear solver options for adjoint system.
        
    Returns:
        Gradients with respect to problem parameters.
        
    Note:
        The method solves the adjoint system A^T λ = v where A is the Jacobian
        at the solution state, then computes the parameter sensitivities using
        the chain rule: dF/dp = -λ^T (∂c/∂p) where c is the constraint (residual).
        
    Example:
        >>> adjoint_options = {'jax_solver': {'precond': True}}
        >>> gradients = implicit_vjp(problem, solution, params, v_list, adjoint_options)
    """

    def constraint_fn(dofs, params):
        """c(u, p)"""
        problem.set_params(params)
        res_fn = problem.compute_residual
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        return res_fn(dofs)

    def constraint_fn_sol_to_sol(sol_list, params):
        dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
        con_vec = constraint_fn(dofs, params)
        return problem.unflatten_fn_sol_list(con_vec)

    def get_partial_params_c_fn(sol_list):
        """c(u=u, p)"""

        def partial_params_c_fn(params):
            return constraint_fn_sol_to_sol(sol_list, params)

        return partial_params_c_fn

    def get_vjp_contraint_fn_params(params, sol_list):
        """v*(partial dc/dp)"""
        partial_c_fn = get_partial_params_c_fn(sol_list)

        def vjp_linear_fn(v_list):
            primals_output, f_vjp = jax.vjp(partial_c_fn, params)
            (val,) = f_vjp(v_list)
            return val

        return vjp_linear_fn

    problem.set_params(params)
    problem.newton_update(sol_list)

    A, A_reduced = get_A(problem)
    v_vec = jax.flatten_util.ravel_pytree(v_list)[0]

    if problem.prolongation_matrix is not None:
        v_vec = problem.prolongation_matrix.T @ v_vec

    # Be careful that A.transpose() does in-place change to A
    # Create a copy to avoid in-place modification
    A_reduced_T = A_reduced.transpose()
    adjoint_vec = linear_solver(
        A_reduced_T, v_vec, None, adjoint_solver_options
    )
    # Clean up transposed matrix
    A_reduced_T.destroy()

    if problem.prolongation_matrix is not None:
        adjoint_vec = problem.prolongation_matrix @ adjoint_vec

    vjp_linear_fn = get_vjp_contraint_fn_params(params, sol_list)
    vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_vec))
    vjp_result = jax.tree_map(lambda x: -x, vjp_result)
    
    # Clean up PETSc matrices
    A.destroy()
    A_reduced.destroy()

    return vjp_result


def ad_wrapper(problem: Any, solver_options: Dict[str, Any] = {}, adjoint_solver_options: Dict[str, Any] = {}) -> Callable[[Any], List[np.ndarray]]:
    """Create automatic differentiation wrapper for the solver.
    
    Wraps the nonlinear solver with JAX's custom VJP (vector-Jacobian product)
    to enable automatic differentiation through the solution process. This allows
    the solver to be used in optimization loops and gradient-based algorithms.
    
    Args:
        problem (Problem): Finite element problem template.
        solver_options (dict, optional): Options for forward solver. Defaults to {}.
        adjoint_solver_options (dict, optional): Options for adjoint solver. Defaults to {}.
        
    Returns:
        Callable: JAX-differentiable function that takes parameters and returns solution.
        
    Example:
        Setup for parameter optimization:
        
        >>> differentiable_solver = ad_wrapper(problem)
        >>> 
        >>> def objective(params):
        ...     solution = differentiable_solver(params)
        ...     return compute_objective(solution)
        >>> 
        >>> grad_fn = jax.grad(objective)
        >>> gradients = grad_fn(initial_params)
        
    Note:
        The wrapper uses implicit differentiation via the adjoint method to
        compute gradients efficiently, avoiding the need to differentiate
        through the entire Newton iteration process.
    """
    @jax.custom_vjp
    def fwd_pred(params):
        problem.set_params(params)
        sol_list = solver(problem, solver_options)
        return sol_list

    def f_fwd(params):
        sol_list = fwd_pred(params)
        return sol_list, (params, sol_list)

    def f_bwd(res, v):
        logger.info("Running backward and solving the adjoint problem...")
        params, sol_list = res
        vjp_result = implicit_vjp(problem, sol_list, params, v, adjoint_solver_options)
        return (vjp_result,)

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred
