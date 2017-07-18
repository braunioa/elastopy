"""Module for enriched quad element with 4 nodes - type 3 in gmsh

"""
import numpy as np
from .quad4 import Quad4


class Quad4Enr(Quad4):
    """Constructor of a 4-node quadrangle (TYPE 3) enriched element

    Uses methods of the standard Quad4 element.

    Args:
        eid: element index
        model: object with model parameters
        eps0: element inital strain array shape [3]

    Attributes:
        phi (numpy array): nodal value for the signed distance function
            of this element
        num_enr_nodes (int): number of enriched nodes of this element
        num_enr_dof (int): number of enriched degree`s of freedom in this
            element.
    """
    def __init__(self, eid, model, EPS0):
        # initialize Quad4 standard
        super().__init__(eid, model, EPS0)

        self.phi = model.PHI[self.conn]

    def stiffness_matrix(self, t=1):
        """Build the enriched element stiffness matrix

        Note:
            This method overwrites the Quad4 method

        """
        kuu = np.zeros((self.num_std_dof, self.num_std_dof))
        kaa = np.zeros((self.num_enr_dof, self.num_enr_dof))
        kua = np.zeros((self.num_std_dof, self.num_enr_dof))

        gauss_points = self.XEZ / np.sqrt(3.0)

        for i, gp in enumerate(gauss_points):
            N, dN_ei = self.shape_function(xez=gp)
            dJ, dN_xi, _ = self.jacobian(self.xyz, dN_ei)

            if callable(self.E):
                x1, x2 = self.mapping(self.xyz)
                C = self.c_matrix(t, x1, x2)
            elif type(self.E) is list:
                C = self.c_matrix(t, n=i)
            else:
                C = self.c_matrix(t)

            Bj = []
            for j in range(4):
                Bj.append(np.array([[dN_xi[0, j], 0],
                                    [0, dN_xi[1, j]],
                                    [dN_xi[1, j], dN_xi[0, j]]]))
            Bstd = np.block([Bj[i] for i in range(4)])

            Bk = []
            for j, n in enumerate(self.enriched_nodes):
                psi = abs(N @ self.phi) - abs(self.phi[j])
                dpsi_x = np.sign(N @ self.phi)*(dN_xi[0, :] @ self.phi)
                dpsi_y = np.sign(N @ self.phi)*(dN_xi[1, :] @ self.phi)
                Bk.append(np.array([[dN_xi[0, j]*(psi) + N[j]*dpsi_x, 0],
                                    [0, dN_xi[1, j]*(psi) + N[j]*dpsi_y],
                                    [dN_xi[1, j]*(psi) + N[j]*dpsi_y,
                                     dN_xi[0, j]*(psi) + N[j]*dpsi_x]]))
            Benr = np.block([Bk[i] for i in range(len(self.enriched_nodes))])

            kuu += (Bstd.T @ C @ Bstd)*dJ
            kaa += (Benr.T @ C @ Benr)*dJ
            kua += (Bstd.T @ C @ Benr)*dJ

        k = np.block([[kuu, kua],
                      [kua.T, kaa]])

        return k

    def load_body_vector(self, b_force=None, t=1):
        """Build the element vector due body forces b_force

        """
        gauss_points = self.XEZ / np.sqrt(3.0)

        pb_std = np.zeros(self.num_std_dof)
        pb_enr = np.zeros(self.num_enr_dof)

        if b_force is not None:
            for gp in gauss_points:
                N, dN_ei = self.shape_function(xez=gp)
                dJ, dN_xi, _ = self.jacobian(self.xyz, dN_ei)

                x1, x2 = self.mapping(N, self.xyz)

                pb_std[0] += N[0]*b_force(x1, x2, t)[0]*dJ
                pb_std[1] += N[0]*b_force(x1, x2, t)[1]*dJ
                pb_std[2] += N[1]*b_force(x1, x2, t)[0]*dJ
                pb_std[3] += N[1]*b_force(x1, x2, t)[1]*dJ
                pb_std[4] += N[2]*b_force(x1, x2, t)[0]*dJ
                pb_std[5] += N[2]*b_force(x1, x2, t)[1]*dJ
                pb_std[6] += N[3]*b_force(x1, x2, t)[0]*dJ
                pb_std[7] += N[3]*b_force(x1, x2, t)[1]*dJ

        pb = np.block([pb_std, pb_enr])
        return pb

    def load_strain_vector(self, t=1):
        """Build the element vector due initial strain

        """
        gauss_points = self.XEZ / np.sqrt(3.0)

        pe_std = np.zeros(self.num_std_dof)
        pe_enr = np.zeros(self.num_enr_dof)

        for i, gp in enumerate(gauss_points):

            if callable(self.E):
                x1, x2 = self.mapping(N, self.xyz)
                C = self.c_matrix(t, x1, x2)
            elif type(self.E) is list:
                C = self.c_matrix(t, n=i)
            else:
                C = self.c_matrix(t)

            B = np.array([
                [dN_xi[0, 0], 0, dN_xi[0, 1], 0, dN_xi[0, 2], 0,
                 dN_xi[0, 3], 0],
                [0, dN_xi[1, 0], 0, dN_xi[1, 1], 0, dN_xi[1, 2], 0,
                 dN_xi[1, 3]],
                [dN_xi[1, 0], dN_xi[0, 0], dN_xi[1, 1], dN_xi[0, 1],
                 dN_xi[1, 2], dN_xi[0, 2], dN_xi[1, 3], dN_xi[0, 3]]])

            pe_std += (B.T @ C @ self.eps0)*dJ

        pe = np.block([pe_std, pe_enr])
        return pe

    def load_traction_vector(self, traction_bc=None, t=1):
        """Build element load vector due traction_bction boundary condition

        Performs the line integral on boundary with traction vector assigned.

        1. First loop over the boundary lines tag which are the keys of the
           traction_bc dictionary.
        2. Then loop over elements at boundary lines and which side of the
           element is at the boundary line.

        Note:
            This method overwrites the Quad4 method

        Args:
            traction_bc (dict): dictionary with boundary conditions with the
                format: traction_bc = {line_tag: [vector_x, vector_y]}.
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

        pt_std = np.zeros(self.num_std_dof)
        pt_enr = np.zeros(self.num_enr_dof)

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
                            x1, x2 = self.mapping(N, self.xyz)

                            # Nstd matrix with shape function shape (2, 8)
                            N_ = []
                            for j in range(4):
                                N_.append(np.array([[N[j], 0],
                                                    [0, N[j]]]))
                            Nstd = np.block([N_[j] for j in range(4)])

                            # traction_bc(x1, x2, t)[key] is a numpy array shape (2,)
                            pt_std += Nstd.T @ traction_bc(x1, x2, t)[key] * dL

                            # node in this element at this line
                            node = self.local_nodes_in_side[ele_side][w]
                            psi = abs(N @ self.phi) - abs(self.phi[node])

                            N_ = []
                            for j in range(self.num_enr_nodes):
                                N_.append(np.array([[N[j]*psi, 0],
                                                    [0, N[j]*psi]]))

                    else:
                        # Catch element that is not at boundary
                        continue

        pt = np.block([pt_std, pt_enr])
        return pt
