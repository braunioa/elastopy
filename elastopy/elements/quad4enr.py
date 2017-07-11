"""Module for qmodel, material, EPS0uad element with 4 nodes - type 3 in gmsh

"""
from elastopy.elements.quad4 import Quad4
import numpy as np


class Quad4Enr(Quad4):
    """Constructor of a 4-node quadrangle (TYPE 3) enriched element

    Uses methods of the standard Quad4 element.

    Args:
        eid: element index
        model: object with model parameters
        material: object with material parameters
        eps0: element inital strain array shape [3]

    """
    def __init__(self, eid, model, material, EPS0):
        # initialize Quad4 standard
        super().__init__(eid, model, material, EPS0)

        self.enr_nodes = []
        for n in model.enriched_nodes:
            if n in self.conn:
                self.enr_nodes.append(n)

    def stiffness_matrix(self, t=1):
        """Build the enriched element stiffness matrix

        """
        k = np.zeros((8, 8))

        gauss_points = self.XEZ / np.sqrt(3.0)

        for gp in gauss_points:
            _, dN_ei = self.shape_function(xez=gp)
            dJ, dN_xi, _ = self.jacobian(self.xyz, dN_ei)

            if callable(self.E):
                x1, x2 = self.mapping(self.xyz)
                C = self.c_matrix(t, x1, x2)
            else:
                C = self.c_matrix(t)

            Bj = []
            for j in range(4):
                Bj.append(np.array([[dN_xi[0, j], 0],
                                    [0, dN_xi[1, j]],
                                    [dN_xi[1, j], dN_xi[0, j]]]))
            Bstd = np.block([Bj[i] for i in range(4)])

            Benr = []
            for j in range(4):
                pass

            k += (Bstd.T @ C @ Bstd)*dJ

        return k

    def load_body_vector(self, b_force=None, t=1):
        """Build the element vector due body forces b_force

        """
        gauss_points = self.XEZ / np.sqrt(3.0)

        pb = np.zeros(8)
        if b_force is not None:
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

    def load_strain_vector(self, t=1):
        """Build the element vector due initial strain

        """
        C = self.c_matrix(t)

        gauss_points = self.XEZ / np.sqrt(3.0)

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

            pe += (B.T @ C @ self.eps0)*dJ

        return pe

    def load_traction_vector(self, traction_bc=None, t=1):
        """Build element load vector due traction_bction boundary condition

        """
        gp = np.array([
            [[-1.0/np.sqrt(3), -1.0],
             [1.0/np.sqrt(3), -1.0]],
            [[1.0, -1.0/np.sqrt(3)],
             [1.0, 1.0/np.sqrt(3)]],
            [[-1.0/np.sqrt(3), 1.0],
             [1.0/np.sqrt(3), 1.0]],
            [[-1.0, -1.0/np.sqrt(3)],
             [-1.0, 1/np.sqrt(3)]]])

        pt = np.zeros(8)

        if traction_bc is not None:
            # loop for specified boundary conditions
            for key in traction_bc(1, 1).keys():
                line = key[1]

                for ele_boundary_line, ele_side in zip(self.at_boundary_line,
                                                       self.side_at_boundary):
                    # Check if this element is at the line with traction
                    if line == ele_boundary_line:

                        # perform the integral with GQ
                        for w in range(2):
                            N, dN_ei = self.shape_function(xez=gp[ele_side, w])
                            _, _, arch_length = self.jacobian(self.xyz, dN_ei)

                            dL = arch_length[ele_side]
                            x1, x2 = self.mapping(self.xyz)

                            pt[0] += N[0] * traction_bc(x1, x2, t)[key][0] * dL
                            pt[1] += N[0] * traction_bc(x1, x2, t)[key][1] * dL
                            pt[2] += N[1] * traction_bc(x1, x2, t)[key][0] * dL
                            pt[3] += N[1] * traction_bc(x1, x2, t)[key][1] * dL
                            pt[4] += N[2] * traction_bc(x1, x2, t)[key][0] * dL
                            pt[5] += N[2] * traction_bc(x1, x2, t)[key][1] * dL
                            pt[6] += N[3] * traction_bc(x1, x2, t)[key][0] * dL
                            pt[7] += N[3] * traction_bc(x1, x2, t)[key][1] * dL

                    else:
                        # Catch element that is not at boundary
                        continue

        return pt
