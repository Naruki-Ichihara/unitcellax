"""
Multi-load case topology optimization with sequential solving.

This example demonstrates topology optimization under multiple loading conditions
using sequential solving for each load case with JAX automatic differentiation.

Features:
    - Multiple macroscopic strain tensors (E_macro) handled sequentially
    - Standard solver with JAX autodiff support
    - CPU-optimized GCMMA optimizer with memory management
    - Weighted compliance objective for multi-load cases
    - Helmholtz filtering for mesh-independent solutions

Performance Notes:
    - Sequential solving of load cases for compatibility
    - Uses standard solver (ad_wrapper) with PETSc backend
    - Memory efficient implementation
    - CPU-optimized with automatic memory management
    - Robust and stable for multiple load cases

Technical Implementation:
    - Standard scipy sparse matrices for compatibility
    - Uses ad_wrapper from solver.py with JAX autodiff
    - PETSc-based linear solver for robustness
    - Full JAX autodiff support for optimization
"""

import unitcellax as cell
import os
import jax
import jax.numpy as np
from unitcellax.unitcell import UnitCell
from unitcellax.physics import LinearElasticity
from unitcellax.fem.mesh import box_mesh
from unitcellax.pbc import prolongation_matrix, periodic_bc_3D
from unitcellax.fem.solver import ad_wrapper
from unitcellax.utils import save_as_vtk
from unitcellax.filters import HelmholtzFilter
from unitcellax.optimizers import GCMMAOptimizer
from jax.experimental.sparse import BCOO
import numpy as onp

# Material properties
E = 70e3        # Young's modulus
E_min = 1e-5    # Minimum Young's modulus for topology optimization
nu = 0.3        # Poisson's ratio
vf = 0.25       # Volume fraction constraint

# Mesh and problem parameters
data_dir = os.path.join(os.path.dirname(__file__), 'data_multiload')
os.makedirs(data_dir, exist_ok=True)
N = 10          # Small mesh size for vmap demonstration (memory constraints)
L = 1.0
vec = 3
dim = 3
ele_type = "HEX8"

class Unitcell_3D(UnitCell):
    def mesh_build(self):
        mesh = box_mesh(N, N, N, L, L, L, ele_type)
        return mesh

unitcell = Unitcell_3D()

# Define multiple macroscopic strain tensors for different load cases
E_tensors = np.array([
    # Load case 1: Extension in x-direction
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0]],
    
    # Load case 2: Extension in y-direction  
    [[0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0]],
    
    # Load case 3: Shear in xy-plane
    [[0.0, 0.5, 0.0],
     [0.5, 0.0, 0.0],
     [0.0, 0.0, 0.0]],
    
    # Load case 4: Extension in z-direction
    [[0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]]
])

print(f"Number of load cases: {len(E_tensors)}")

# Apply macro displacements for all load cases
macro_disps = []
for i, E_tensor in enumerate(E_tensors):
    macro_disp = unitcell.apply_macro_displacement(E_tensor)
    macro_disps.append(macro_disp)
    print(f"Load case {i+1}: Applied strain tensor")

macro_disps = np.array(macro_disps)  # Shape: (num_load_cases, num_dofs)

# Periodic boundary conditions (same for all load cases)
P = prolongation_matrix(periodic_bc_3D(unitcell, 3), 
                       unitcell.mesh, vec, 0)
P_rho = prolongation_matrix(periodic_bc_3D(unitcell, 1),
                           unitcell.mesh, 1, 0)
P_rho_jax = BCOO.from_scipy_sparse(P_rho)

# Create base problem template (will be used for all load cases)
base_problem = LinearElasticity(
    mesh=unitcell.mesh,
    E=E,
    nu=nu,
    E_min=E_min,
    penalty=3.0,
    vec=vec,
    dim=dim,
    ele_type=ele_type,
    prolongation_matrix=P,  # Use scipy sparse matrix
    macro_term=macro_disps[0]   # Template macro term
)

print(f"Created base problem for {len(macro_disps)} load cases")

# Helmholtz filter (shared across all load cases)
h_filter = HelmholtzFilter(unitcell, 1, radius=0.05, prolongation_matrix=P_rho)

# Weights for different load cases
load_case_weights = np.array([1.0, 1.0, 1.0, 1.0])
load_case_weights = load_case_weights / np.sum(load_case_weights)
print(f"Load case weights: {load_case_weights}")

fwd_pred = ad_wrapper(base_problem, solver_options={'jax_solver': {}}, adjoint_solver_options={'jax_solver': {}})

def solve_single_load_case_vmap(macro_disp, rho):
    """Solve single load case using standard solver.
    
    Args:
        macro_disp: Macro displacement vector for this load case
        rho: Material density field
        
    Returns:
        float: Compliance for this load case
    """
    base_problem.macro_term = macro_disp  # Update macro term for this load case
    # Solve forward problem
    u_sol = fwd_pred(rho)[0]
    # Compute compliance
    compliance = base_problem.compliance(u_sol, rho)
    return compliance

def solve_all_load_cases(rho):
    """Solve all load cases sequentially."""
    compliances = []
    for macro_disp in macro_disps:
        compliance = solve_single_load_case_vmap(macro_disp, rho)
        compliances.append(compliance)
    return np.array(compliances)

print("Using sequential load case computation with standard solver!")

def J_total(params):
    """Multi-load case objective function: minimize weighted compliance sum.
    
    Computes compliance for all load cases and returns weighted sum.
    """
    rho_reduced = params
    rho = P_rho_jax @ rho_reduced
    rho = h_filter.filtered(rho)
    
    # Solve all load cases sequentially
    compliances = solve_all_load_cases(rho)
    
    # Compute weighted sum of compliances
    weighted_compliance = np.sum(load_case_weights * compliances)
    
    # Minimize compliance (maximize stiffness)
    objective = -weighted_compliance
    return objective

def computeGlobalVolumeConstraint(rho):
    """Volume constraint function for optimization."""
    rho = P_rho_jax @ rho
    rho = h_filter.filtered(rho)
    g = np.mean(rho)/vf - 1.
    return g

def save_visualization(x, epoch):
    """Save visualization callback for optimization."""
    rho = P_rho_jax @ x
    rho = h_filter.filtered(rho)
    rho_np = onp.asarray(rho)  # Convert to numpy for saving
    
    # Save density field
    save_as_vtk(base_problem.fe, os.path.join(data_dir, f'density_epoch_{epoch}.vtu'), 
                point_infos=[('rho', rho_np)])

# Setup optimization with GCMMAOptimizer
rho_ini = np.ones(P_rho.shape[1]) * vf
n_vars = len(rho_ini)

print(f"Number of design variables: {n_vars}")
print(f"Optimization problem setup complete")

# Create GCMMA optimizer for multi-load topology optimization (CPU-optimized)
opt = GCMMAOptimizer(
    n_vars=n_vars,
    objective_fn=J_total,
    volume_constraint_fn=computeGlobalVolumeConstraint,
    volume_fraction=vf,
    save_callback=save_visualization,
    memory_cleanup_freq=3  # More frequent cleanup for multi-load case
)

# Set custom GCMMA stopping criteria (reduced for vmap demonstration)
opt.set_gcmma_options(max_eval=5, x_tol=1e-4, f_tol=1e-7)

# Test objective function before optimization
print("\nTesting multi-load objective function...")
try:
    test_obj = J_total(rho_ini)
    print(f"Initial objective value: {test_obj:.6e}")
except Exception as e:
    print(f"Error in objective function: {e}")
    import traceback
    traceback.print_exc()

# Run optimization
print("\nStarting multi-load GCMMA topology optimization (CPU)...")
x_opt, opt_val = opt.optimize(rho_ini)

print(f"\nOptimization completed!")
print(f"Final design saved to: {data_dir}")
print(f"Number of load cases handled: {len(E_tensors)}")

# Final analysis: compute individual load case compliances
print("\nFinal compliance analysis:")
rho_final = h_filter.filtered(P_rho @ x_opt)
final_compliances = solve_all_load_cases(rho_final)

for i, comp in enumerate(final_compliances):
    print(f"Load case {i+1} compliance: {float(comp):.6e} (weight: {load_case_weights[i]:.3f})")

weighted_final = float(np.sum(load_case_weights * final_compliances))
print(f"Weighted total compliance: {weighted_final:.6e}")