from elastopy import boundary
from elastopy import stiffness
from elastopy import load
from elastopy import traction
from elastopy import stress
import numpy as np


def solver(model, material, b_force, trac_bc, displ_bc, EPS0=0, t=1):
    """Solver for the elastostatics problem

    Return:

    U: displacement solution of the boundary value problem
    SIG: stresses response of the system under the boundary conditions

    """
    K = stiffness.K_matrix(model, material, t)

    Pb = load.Pb_vector(model, b_force, t)

    Pt = traction.Pt_vector(model, trac_bc, t)

    Pe = load.Pe_vector(model, material, EPS0, t)

    P = Pb + Pt + Pe

    Km, Pm = boundary.dirichlet(K, P, model, displ_bc)

    U = np.linalg.solve(Km, Pm)

    SIG = stress.recovery(model, material, U, EPS0, t)

    return U, SIG
