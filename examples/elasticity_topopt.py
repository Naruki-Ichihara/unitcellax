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

data_dir = os.path.join(os.path.dirname(__file__), 'data')
N = 30
L = 1.0
vec = 3
dim = 3
ele_type = "HEX8"

# Mesh and problem parameters


class Unitcell_3D(UnitCell):
    def mesh_build(self):
        mesh = box_mesh(N, N, N, L, L, L, ele_type)
        return mesh
unitcell = Unitcell_3D()

E_tensor = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
])
macro_disp = unitcell.apply_macro_displacement(E_tensor)
P = prolongation_matrix(periodic_bc_3D(unitcell, 3), 
                           unitcell.mesh, vec, 0)
P_rho = prolongation_matrix(periodic_bc_3D(unitcell, 1),
                           unitcell.mesh, 1, 0)
P_rho_jax = BCOO.from_scipy_sparse(P_rho)

# Create physics problem using built-in LinearElasticity
problem = LinearElasticity(
    mesh=unitcell.mesh,
    E=E,
    nu=nu,
    E_min=E_min,
    penalty=3.0,  # SIMP penalty exponent
    vec=vec,
    dim=dim,
    ele_type=ele_type,
    prolongation_matrix=P,
    macro_term=macro_disp
)
fwd_pred = ad_wrapper(problem, solver_options={'jax_solver': {}}, adjoint_solver_options={'jax_solver': {}})
h_filter = HelmholtzFilter(unitcell, 1, radius=0.05, prolongation_matrix=P_rho)

def J_total(params):
    """Objective function: minimize compliance (maximize stiffness)."""
    rho_reduced = params
    rho = P_rho_jax @ rho_reduced
    rho = h_filter.filtered(rho)
    
    # Set material parameters for current density field
    problem.set_params(rho)
    
    # Solve forward problem
    u_sol = fwd_pred(rho)[0]
    
    # Compute compliance using built-in physics method
    comp = problem.compliance(u_sol, rho)
    objective = -comp  # Minimize compliance (maximize stiffness)
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
    save_as_vtk(problem.fe, os.path.join(data_dir, f'epoch_{epoch}.vtu'), point_infos=[('rho', rho_np)])

# Setup optimization with TopologyOptimizer
rho_ini = np.ones(P_rho.shape[1]) * vf
n_vars = len(rho_ini)

# Create GCMMA optimizer for topology optimization (CPU-optimized)
opt = GCMMAOptimizer(
    n_vars=n_vars,
    objective_fn=J_total,
    volume_constraint_fn=computeGlobalVolumeConstraint,
    volume_fraction=vf,
    save_callback=save_visualization,
    memory_cleanup_freq=5
)

# Set custom GCMMA stopping criteria
opt.set_gcmma_options(max_eval=20, x_tol=1e-6, f_tol=1e-9)

# Run optimization
print("Starting GCMMA topology optimization (CPU)...")
x_opt, opt_val = opt.optimize(rho_ini)