"""Compute the stress for each node

"""
import numpy as np
from elastopy.constructor import constructor


def recovery(model, material, U, EPS0):
    """Recovery stress at nodes from displacement

    """
    # initiate the arrays for element and global stress
    sig = np.zeros(3)
    SIG = np.zeros((model.nn, 3))

    for e, conn in enumerate(model.CONN):
        element = constructor(e, model, material, EPS0)
        dof = element.dof
        xyz = element.xyz

        # if there is no initial strain
        if EPS0 is None:
            eps0 = np.zeros(3)
        else:
            eps0 = EPS0[e]

        E = element.E
        nu = element.nu

        C = c_matrix(E, nu)

        u = U[dof]

        # quadrature on the nodes coord in the isodomain
        for n, xez in enumerate(element.XEZ):
            _, dN_ei = element.shape_function(xez)
            dJ, dN_xi, _ = element.jacobian(xyz, dN_ei)

            # number of elements sharing a node
            num_ele_shrg = (model.CONN == conn[n]).sum()

            B = np.array([
               [dN_xi[0, 0], 0, dN_xi[0, 1], 0, dN_xi[0, 2], 0,
                dN_xi[0, 3], 0],
               [0, dN_xi[1, 0], 0, dN_xi[1, 1], 0, dN_xi[1, 2], 0,
                dN_xi[1, 3]],
               [dN_xi[1, 0], dN_xi[0, 0], dN_xi[1, 1], dN_xi[0, 1],
                dN_xi[1, 2], dN_xi[0, 2], dN_xi[1, 3], dN_xi[0, 3]]])

            # sig = [sig_11 sig_22 sig_12] for each n node
            sig = C @ (B @ u - eps0)

            # dof 1 degree of freedom per node
            d = int(dof[2*n]/2)

            # unweighted average of stress at nodes
            SIG[d, :] += sig/num_ele_shrg

    return SIG


def c_matrix(E, nu):
    """Build the element constitutive matrix

    """
    C = np.zeros((3, 3))
    C[0, 0] = 1.0
    C[1, 1] = 1.0
    C[1, 0] = nu
    C[0, 1] = nu
    C[2, 2] = (1.0 - nu)/2.0
    C = (E/(1.0-nu**2.0))*C
    return C


def principal_max(s11, s22, s12):
    """Compute the principal stress max

    """
    sp_max = np.zeros(len(s11))
    for i in range(len(s11)):
        sp_max[i] = (s11[i] + s22[i]) / 2.0 + np.sqrt(
            (s11[i] - s22[i])**2.0 / 2.0 + s12[i]**2.0)
    return sp_max


def principal_min(s11, s22, s12):
    """Compute the principal stress minimum

    """
    sp_min = np.zeros(len(s11))
    for i in range(len(s11)):
        sp_min[i] = (s11[i]+s22[i])/2. - np.sqrt((s11[i] - s22[i])**2./2. +
                                                 s12[i]**2.)
    return sp_min

def recovery_gauss(model, material, U, EPS0):
    """Recovery stress at nodes from displacement

    """
    # initiate the arrays for element and global stress
    SIG = np.zeros((model.nn, 3))
    SIG2 = np.zeros((model.nn, 3))
    # extrapolation matrix
    Q = np.array([[1 + np.sqrt(3)/2, -1/2, 1 - np.sqrt(3)/2, -1/2],
                  [-1/2, 1 + np.sqrt(3)/2, -1/2, 1 - np.sqrt(3)/2],
                  [1 - np.sqrt(3)/2, -1/2, 1 + np.sqrt(3)/2, -1/2],
                  [-1/2, 1 - np.sqrt(3)/2, -1/2, 1 + np.sqrt(3)/2]])

    for e, conn in enumerate(model.CONN):
        element = constructor(e, model, material, EPS0)
        dof = element.dof
        xyz = element.xyz

        # if there is no initial strain
        if EPS0 is None:
            eps0 = np.zeros(3)
        else:
            eps0 = EPS0[e]

        E = element.E
        nu = element.nu
        C = c_matrix(E, nu)

        u = U[dof]

        # quadrature on the nodes coord in the isodomain
        for n, xez in enumerate(element.XEZ/np.sqrt(3)):
            _, dN_ei = element.shape_function(xez)
            dJ, dN_xi, _ = element.jacobian(xyz, dN_ei)

            # number of elements sharing a node
            num_ele_shrg = (model.CONN == conn[n]).sum()

            B = np.array([
               [dN_xi[0, 0], 0, dN_xi[0, 1], 0, dN_xi[0, 2], 0,
                dN_xi[0, 3], 0],
               [0, dN_xi[1, 0], 0, dN_xi[1, 1], 0, dN_xi[1, 2], 0,
                dN_xi[1, 3]],
               [dN_xi[1, 0], dN_xi[0, 0], dN_xi[1, 1], dN_xi[0, 1],
                dN_xi[1, 2], dN_xi[0, 2], dN_xi[1, 3], dN_xi[0, 3]]])

            # sig = [sig_11 sig_22 sig_12] for each n node
            sig_gp = C @ (B @ u - eps0)

            # 1 degree of freedom per node
            # unweighted average of stress at nodes
            SIG[conn[n], :] += sig_gp/num_ele_shrg

        SIG2[conn, 0] = Q @ SIG[conn, 0]
        SIG2[conn, 1] = Q @ SIG[conn, 1]
        SIG2[conn, 2] = Q @ SIG[conn, 2]

    return SIG2
