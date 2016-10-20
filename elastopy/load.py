import numpy as np
from stiffness import c_matrix


def Pb_vector(model, b_force, t=1):
    """Build the load vector due internal body force

    """
    Pb = np.zeros(model.ndof)

    try:
        b_force(1, 1, 1)
    except:
        print("No body force applied!")

    for e, conn in enumerate(model.CONN):
        xyz = model.XYZ[conn]
        dof = model.DOF[e]

        pb = pb_vector(model, xyz, b_force, t)

        id = dof

        Pb[id] += pb

    return Pb


def pb_vector(model, xyz, b_force, t=1):
    """Build the element vector due body forces b_force

    """
    gauss_points = model.chi / np.sqrt(3.0)

    pb = np.zeros(8)
    for gp in gauss_points:
        model.basis_function(gp)
        model.jacobian(xyz)
        dJ = model.detJac
        x1, x2 = model.mapping(xyz)

        pb[0] += model.phi[0]*b_force(x1, x2, t)[0]*dJ
        pb[1] += model.phi[0]*b_force(x1, x2, t)[1]*dJ
        pb[2] += model.phi[1]*b_force(x1, x2, t)[0]*dJ
        pb[3] += model.phi[1]*b_force(x1, x2, t)[1]*dJ
        pb[4] += model.phi[2]*b_force(x1, x2, t)[0]*dJ
        pb[5] += model.phi[2]*b_force(x1, x2, t)[1]*dJ
        pb[6] += model.phi[3]*b_force(x1, x2, t)[0]*dJ
        pb[7] += model.phi[3]*b_force(x1, x2, t)[1]*dJ

    return pb


def Pe_vector(model, material, EPS0, t=1):
    """Build the load vector due initial strain EPS0

    """
    Pe = np.zeros(model.ndof)

    # correct if EPS0 is not specified
    if np.size(EPS0) == 1:
        EPS0 = np.zeros((model.ne, 3))

    for e, conn in enumerate(model.CONN):
        xyz = model.XYZ[conn]
        surf = model.surf_of_ele[e]
        dof = model.DOF[e]

        try:
            E = material.E[surf]
            nu = material.nu[surf]
        except:
            print('Surface {} has no property assigned!'
                  'Default values were used!'.format(surf))
            E = 1.0
            nu = 0.1

        eps0 = EPS0[e]

        pe = pe_vector(model, xyz, E, nu, eps0, t)

        id = dof

        Pe[id] += pe

    return Pe


def pe_vector(model, xyz, E, nu, eps0, t=1):
    """Build the element vector due initial strain

    """
    gauss_points = model.chi / np.sqrt(3.0)

    C = c_matrix(E, nu)

    pe = np.zeros(8)
    for gp in gauss_points:
        model.basis_function(gp)
        model.jacobian(xyz)
        dJ = model.detJac

        dp_xi = model.dphi_xi
        B = np.array([
            [dp_xi[0, 0], 0, dp_xi[0, 1], 0, dp_xi[0, 2], 0,
             dp_xi[0, 3], 0],
            [0, dp_xi[1, 0], 0, dp_xi[1, 1], 0, dp_xi[1, 2], 0,
             dp_xi[1, 3]],
            [dp_xi[1, 0], dp_xi[0, 0], dp_xi[1, 1], dp_xi[0, 1],
             dp_xi[1, 2], dp_xi[0, 2], dp_xi[1, 3], dp_xi[0, 3]]])

        pe += (B.T @ C @ eps0)*dJ

    return pe
