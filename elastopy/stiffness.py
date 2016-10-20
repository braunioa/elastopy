import numpy as np


def K_matrix(model, material, t=1):
    """Build the global stiffness matrix

    """
    K = np.zeros((model.ndof, model.ndof))

    for e, conn in enumerate(model.CONN):
        if 

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

        k = k_matrix(model, xyz, E, nu, t)

        id = np.ix_(dof, dof)

        K[id] += k

    return K


def k_matrix(model, xyz, E, nu, t=1):
    """Build the element stiffness matrix

    """
    gauss_points = model.chi / np.sqrt(3.0)

    C = c_matrix(E, nu, t)

    k = np.zeros((8, 8))

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

        k += (B.T @ C @ B)*dJ

    return k


def c_matrix(E, nu, t=1):
    """Build the element constitutive matrix

    """
    if callable(E) is True:
        E_value = E(t)
    else:
        E_value = E

    C = np.zeros((3, 3))
    C[0, 0] = 1.0
    C[1, 1] = 1.0
    C[1, 0] = nu
    C[0, 1] = nu
    C[2, 2] = (1.0 - nu)/2.0
    C = (E_value/(1.0-nu**2.0))*C

    return C
