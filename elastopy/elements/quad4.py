"""Module for quad element with 4 nodes - type 3 in gmsh

"""
import numpy as np


class Quad4(object):
    """Constructor of a 4-node quadrangle (TYPE 3) element

    """
    def __init__(self, element):
        # get element basic attributes
        self.__dict__.update(element.__dict__)

        # Nodal coordinates in the natural domain (isoparametric coordinates)
        self.chi = np.array([[-1.0, -1.0],
                             [1.0, -1.0],
                             [1.0, 1.0],
                             [-1.0, 1.0]])

    def shape_function(self, xez):
        """Create the basis function and its properties.

        Args:
        xez(array) : position in the isoparametric coordinate xi, eta, zeta

        Return:
        N (array): shape functions

        """
        # variables in the natural (iso-parametric) domain
        e1 = xez[0]
        e2 = xez[1]

        # Terms of the shape function
        e1_term = 0.5*(1.0 + self.chi[:, 0] * e1)
        e2_term = 0.5*(1.0 + self.chi[:, 1] * e2)

        # Basis functions
        # N = [ N_1 N_2 N_3 N4 ]
        N = e1_term*e2_term
        self.N = np.array(N)

        # Derivative of the shape functions
        # dN = [ dN1_e1 dN2_e1 ...
        #         dN1_e2 dN2_e2 ... ]
        dN_ei = np.zeros((2, 4))
        dN_ei[0, :] = 0.5 * self.chi[:, 0] * e2_term
        dN_ei[1, :] = 0.5 * self.chi[:, 1] * e1_term

        return N, dN_ei

    def mapping(self, xyz):
        """maps from cartesian to isoparametric.

        """
        x1, x2 = self.N @ xyz
        return x1, x2

    def jacobian(self, xyz, dN_ei):
        """Creates the Jacobian matrix of the mapping between an element

        """
        # Jac = [ x1_e1 x2_e1
        #         x1_e2 x2_e2 ]
        Jac = np.dot(dN_ei, xyz)

        det_jac = ((Jac[0, 0]*Jac[1, 1] -
                   Jac[0, 1]*Jac[1, 0]))

        # jac_inv = [ e1_x1 e2_x1
        #            e1_x2 e2_x2 ]
        jac_inv = np.linalg.inv(Jac)

        # Using Chain rule,
        # N_xi = N_eI * eI_xi (2x8 array)
        dN_xi = np.zeros((2, 4))
        dN_xi[0, :] = (dN_ei[0, :]*jac_inv[0, 0] +
                       dN_ei[1, :]*jac_inv[0, 1])

        dN_xi[1, :] = (dN_ei[0, :]*jac_inv[1, 0] +
                       dN_ei[1, :]*jac_inv[1, 1])

        # Length of the transofmation arch
        # Jacobian for line integral-2.
        arch_length = np.array([
            (Jac[0, 0]**2. + Jac[0, 1]**2.)**(1./2.),
            (Jac[1, 0]**2. + Jac[1, 1]**2.)**(1./2.),
            (Jac[0, 0]**2. + Jac[0, 1]**2.)**(1./2.),
            (Jac[1, 0]**2. + Jac[1, 1]**2.)**(1./2.)
        ])
        return det_jac, dN_xi, arch_length

    def stiffness(self, material, t=1):
        """Build the element stiffness matrix

        """
        try:
            E = material.E[self.surf]
            nu = material.nu[self.surf]
        except:
            raise Exception('Material not assigned for surface!')

        k = np.zeros((8, 8))

        C = self.c_matrix(E, nu, t)

        gauss_points = self.chi / np.sqrt(3.0)

        for gp in gauss_points:
            _, dN_ei = self.shape_function(xez=gp)
            dJ, dN_xi, _ = self.jacobian(self.xyz, dN_ei)

            B = np.array([
                [dN_xi[0, 0], 0, dN_xi[0, 1], 0, dN_xi[0, 2], 0,
                 dN_xi[0, 3], 0],
                [0, dN_xi[1, 0], 0, dN_xi[1, 1], 0, dN_xi[1, 2], 0,
                 dN_xi[1, 3]],
                [dN_xi[1, 0], dN_xi[0, 0], dN_xi[1, 1], dN_xi[0, 1],
                 dN_xi[1, 2], dN_xi[0, 2], dN_xi[1, 3], dN_xi[0, 3]]])

            k += (B.T @ C @ B)*dJ

        return k

    @staticmethod
    def c_matrix(E, nu, t=1):
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

    def body_load(self, b_force, t=1):
        """Build the element vector due body forces b_force

        """
        gauss_points = self.chi / np.sqrt(3.0)

        pb = np.zeros(8)
        for gp in gauss_points:
            N, dN_ei = self.shape_function(xez=gp)
            dJ, dN_xi, _ = self.jacobian(self.xyz, dN_ei)

            x1, x2 = self.mapping(self.xyz)

            pb[0] += N[0]*b_force(x1, x2, t)[0]*dJ
            pb[1] += N[0]*b_force(x1, x2, t)[1]*dJ
            pb[2] += N[1]*b_force(x1, x2, t)[0]*dJ
            pb[3] += N[1]*b_force(x1, x2, t)[1]*dJ
            pb[4] += N[2]*b_force(x1, x2, t)[0]*dJ
            pb[5] += N[2]*b_force(x1, x2, t)[1]*dJ
            pb[6] += N[3]*b_force(x1, x2, t)[0]*dJ
            pb[7] += N[3]*b_force(x1, x2, t)[1]*dJ

        return pb

    def initial_strain_load(self, material, eps0, t=1):
        """Build the element vector due initial strain

        """
        try:
            E = material.E[self.surf]
            nu = material.nu[self.surf]
        except:
            raise Exception('Material not assigned for surface!')

        C = self.c_matrix(E, nu, t)

        gauss_points = self.chi / np.sqrt(3.0)

        pe = np.zeros(8)
        for gp in gauss_points:
            _, dN_ei = self.shape_function(xez=gp)
            dJ, dN_xi, _ = self.jacobian(self.xyz, dN_ei)

            B = np.array([
                [dN_xi[0, 0], 0, dN_xi[0, 1], 0, dN_xi[0, 2], 0,
                 dN_xi[0, 3], 0],
                [0, dN_xi[1, 0], 0, dN_xi[1, 1], 0, dN_xi[1, 2], 0,
                 dN_xi[1, 3]],
                [dN_xi[1, 0], dN_xi[0, 0], dN_xi[1, 1], dN_xi[0, 1],
                 dN_xi[1, 2], dN_xi[0, 2], dN_xi[1, 3], dN_xi[0, 3]]])

            pe += (B.T @ C @ eps0)*dJ

        return pe

    def traction_load(self, side, line, f, t=1):
        """Build element load vector due traction boundary condition

        """
        pt = np.zeros(4)

        gp = np.array([
            [[-1.0/np.sqrt(3), -1.0],
             [1.0/np.sqrt(3), -1.0]],
            [[1.0, -1.0/np.sqrt(3)],
             [1.0, 1.0/np.sqrt(3)]],
            [[-1.0/np.sqrt(3), 1.0],
             [1.0/np.sqrt(3), 1.0]],
            [[-1.0, -1.0/np.sqrt(3)],
             [-1.0, 1/np.sqrt(3)]]])

        for w in range(2):
            N, dN_ei = self.shape_function(xez=gp[side, w])
            _, _, arch_length = self.jacobian(self.xyz, dN_ei)

            dL = arch_length[side]
            x1, x2 = self.mapping(xyz)

            pt += N[:]*f(x1, x2, t)[line]*dL

        return pt
