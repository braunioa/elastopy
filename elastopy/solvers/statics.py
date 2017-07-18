from .. import boundary
from .. import stress
from ..constructor import constructor
import numpy as np


def solver(model, body_force=None, traction_bc=None, displ_bc=None,
           EPS0=None, t=1):
    """Solver for the elastostatics problem

    Return:

    U: displacement solution of the boundary value problem
    SIG: stresses response of the system under the boundary conditions

    """
    print('Starting statics solver at {:.3f}h '.format(t/3600), end='')
    K = np.zeros((model.num_dof, model.num_dof))
    Pb = np.zeros(model.num_dof)
    Pt = np.zeros(model.num_dof)
    Pe = np.zeros(model.num_dof)

    for eid, type in enumerate(model.TYPE):
        element = constructor(eid, model, EPS0)

        k = element.stiffness_matrix(t)
        pb = element.load_body_vector(body_force, t)
        pt = element.load_traction_vector(traction_bc, t)
        pe = element.load_strain_vector(t)

        K[element.id_m] += k
        Pb[element.id_v] += pb
        Pt[element.id_v] += pt
        Pe[element.id_v] += pe

    P = Pb + Pt + Pe

    Km, Pm = boundary.dirichlet(K, P, model, displ_bc)

    U = np.linalg.solve(Km, Pm)

    if model.xfem:
        SIG = 0
    else:
        SIG = stress.recovery_gauss(model, U, EPS0)

    print('Solution completed!')
    return U, SIG
