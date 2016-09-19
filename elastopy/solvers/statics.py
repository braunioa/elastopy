from elastopy import boundary
from elastopy import stress
from elastopy.constructor import constructor
import numpy as np


def solver(model, material, body_force, traction_bc, displ_bc,
           EPS0=None, t=1):
    """Solver for the elastostatics problem

    Return:

    U: displacement solution of the boundary value problem
    SIG: stresses response of the system under the boundary conditions

    """
    K = np.zeros((model.ndof, model.ndof))
    Pb = np.zeros(model.ndof)
    Pt = np.zeros(model.ndof)
    Pe = np.zeros(model.ndof)

    for eid, type in enumerate(model.TYPE):
        element = constructor(eid, model, material, EPS0)

        k = element.stiffness_matrix()
        pb = element.load_body_vector(body_force)
        pt = element.load_traction_vector(traction_bc)
        pe = element.load_strain_vector()

        K[element.id_m] += k
        Pb[element.id_v] += pb
        Pt[element.id_v] += pt
        Pe[element.id_v] += pe

    P = Pb + Pt + Pe

    Km, Pm = boundary.dirichlet(K, P, model, displ_bc)

    U = np.linalg.solve(Km, Pm)

    SIG = stress.recovery(model, material, U, EPS0)

    return U, SIG
