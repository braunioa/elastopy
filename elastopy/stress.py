import numpy as np
from stiffness import c_matrix

def recovery(model, material, U, EPS0):
    """Recovery stress at nodes from displacement

    """
    sig = np.zeros(3)
    SIG = np.zeros((model.nn, 3))

    # correct if EPS0 is not specified
    if np.size(EPS0) == 1:
        EPS0 = np.zeros((model.ne, 3))

    for e, conn in enumerate(model.CONN):
        surf = model.surf_of_ele[e]
        dof = model.DOF[e]
        xyz = model.XYZ[conn]

        eps0 = EPS0[e]

        try:
            E = material.E[surf]
            nu = material.nu[surf]
        except:
            print('Surface {} has no property assigned!'
                  'Default values were used!'.format(surf))
            E = 1.0
            nu = 0.1

        C = c_matrix(E, nu)

        u = U[dof]

        for n, xez in enumerate(model.chi):
            model.basis_function(xez)
            model.jacobian(xyz)

            # number of elements sharing a node
            num_ele_shrg = (model.CONN == conn[n]).sum()

            dp_xi = model.dphi_xi
            B = np.array([
                [dp_xi[0, 0], 0, dp_xi[0, 1], 0, dp_xi[0, 2], 0,
                 dp_xi[0, 3], 0],
                [0, dp_xi[1, 0], 0, dp_xi[1, 1], 0, dp_xi[1, 2], 0,
                 dp_xi[1, 3]],
                [dp_xi[1, 0], dp_xi[0, 0], dp_xi[1, 1], dp_xi[0, 1],
                 dp_xi[1, 2], dp_xi[0, 2], dp_xi[1, 3], dp_xi[0, 3]]])

            # sig = [sig_11 sig_22 sig_12] for each n node
            sig = C @ (B @ u - eps0)

            # dof 1 degree of freedom per node
            d = int(dof[2*n]/2)

            # unweighted average of stress at nodes
            SIG[d, :] += sig/num_ele_shrg

    return SIG


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
