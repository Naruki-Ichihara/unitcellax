import jax
import jax.numpy as np
import os
from unitcellax.physics import LinearElasticity
from unitcellax.fem.mesh import box_mesh
from unitcellax.pbc import prolongation_matrix, periodic_bc_3D
from unitcellax.fem.solver import ad_wrapper
from unitcellax.graph import fcc
from unitcellax.unitcell import UnitCell
from unitcellax.filters import HelmholtzFilter
from unitcellax.utils import save_as_vtk

data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')

# Material parameters.
E = 70e3
E_min = 1e-8
nu = 0.3

data_dir = os.path.join(os.path.dirname(__file__), 'data')
N = 40
L = 1.0
vec = 3
dim = 3
ele_type = "HEX8"
class Unitcell_3D(UnitCell):
    def mesh_build(self):
        mesh = box_mesh(N, N, N, L, L, L, ele_type)
        return mesh
    
unitcell = Unitcell_3D()

# Perturbation
exx = 0.0
exy = 0.1
exz = 0.0

eyy = 0.0
ezy = 0.0
ezz = 0.0

E_macro = np.array([[exx, exy, exz],
                    [exy, eyy, ezy],
                    [exz, ezy, ezz]])

P = prolongation_matrix(periodic_bc_3D(unitcell, 3), 
                           unitcell.mesh, vec, 0)
P_rho = prolongation_matrix(periodic_bc_3D(unitcell, 1),
                           unitcell.mesh, 1, 0)

macro_disp = unitcell.apply_macro_displacement(E_macro)


# Construct the problem using the physics module
problem = LinearElasticity(unitcell.mesh, 
                           E=E, 
                           nu=nu, 
                           E_min=E_min,
                           penalty=5.0,  # Using penalty=5 to match the original rho**5
                           vec=vec, 
                           dim=dim, 
                           ele_type=ele_type,
                           prolongation_matrix=P, 
                           macro_term=macro_disp)
filter = HelmholtzFilter(unitcell, 1, 0.03, P_rho)

# Solve
fwd_pred = ad_wrapper(problem, solver_options={'petsc_solver': {}}, adjoint_solver_options={'petsc_solver': {}})
rho = jax.vmap(fcc(radius=0.1, scale=L), in_axes=(0,))(unitcell.points)
rho = filter.filtered(rho)
sol_list  = fwd_pred(rho)
u_sol = sol_list[0]
save_as_vtk(problem.fe, vtk_path, point_infos=[('rho', problem.full_params[:]), ('displacement', u_sol)])