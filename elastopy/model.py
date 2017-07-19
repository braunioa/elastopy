"""Module for building the structure model

This module contains the class that creates the
structure model.
The structure model has all attributes from the mesh object.

"""
import numpy as np
import levelset
from .xfem.zerolevelset import Create


class Build(object):
    """Build the model object

    Note:
        The enrichment attributes are set if the zerolevelset is passed
        as a non None argument.

    Note:
        Updates the number of dofs and the element dofs if the zero
        level set function is set, which means that enrichments will
        be used.
        The enriched dofs are set by continuing counting from the last
        global dof tag.

        Example:
            # standard dofs with global tags
            DOF = [[0, 1, 2, 3, 4, 5, 6, 7],
                   [2, 3, 8, 9, 14, 15, 4, 5],
                   [8, 9, 10, 11, 12, 13, 14, 15]]
            enriched_nodes = [1, 2, 4, 7]
            enriched_DOF = [16, 17, 18, 19, 20, 21, 22, 23]

        where, 16 = 2*(0) + 16 and 21 = 2*(2) + 16 + {1}, the number in
        parenthesis is the index of the node tag in the enriched_nodes list
        and the number in {} is the additional dof for the 2d mechanics
        problem.

    Args:
        mesh (object): object containing mesh data
        zerolevelset (optional): object with zero level set
            definition created with the class:
                elastopy.xfem.zerolevelset.Create()

    Attributes:
        mesh attributes from elastopy.mesh.gmsh.Parse() object
        discontinuity_elements (list): elements cut by the discontinuity
        enriched_elements (list): elements that are enriched
        enriched_nodes (numpy array): enriched nodes
        num_enr_dof (int): number of enriched dofs for whole model
        num_std_dof (int): number of standard dofs for whole model
        xfem (boolean): indicates if xfem is used
        PHI (numpy array): shape (nn, ) signed distance function from the zero
            level set contour to mesh nodes.
        material (obj): aggregation
        zerolevelset (obj): aggregation

    """
    def __init__(self, mesh, material=None, zerolevelset=None,
                 num_quad_points=2):
        # copy attributes from mesh object
        self.__dict__ = mesh.__dict__.copy()

        self.num_quad_points = self.num_ele * [num_quad_points]

        self.xfem = False

        if material is not None:
            # aggregate material object as a model instance
            self.material = material

            if material.case is 'strain':
                self.thickness = 0.01
            else:
                self.thickness = 1

        self.enriched_nodes = np.array([], dtype='int')
        self.enriched_elements = []
        # define if there will be enrichment
        if zerolevelset is not None:
            self.xfem = True
            self.zerolevelset = zerolevelset
            self.discontinuity_elements = []
            # self.PHI shape (nn, ) with signed distance value
            self.PHI = levelset.distance(zerolevelset.mask_ls,
                                         zerolevelset.grid_x,
                                         zerolevelset.grid_y,
                                         self.XYZ)

            for e, conn in enumerate(self.CONN):
                # check if element is enriched or not
                if np.all(self.PHI[conn] < 0) or np.all(
                        self.PHI[conn] > 0):
                    pass
                else:
                    self.discontinuity_elements.append(e)

            for e in self.discontinuity_elements:
                self.enriched_nodes = np.append(self.enriched_nodes,
                                                self.CONN[e])
            self.enriched_nodes = np.unique(self.enriched_nodes)

            # Update global DOF tags
            max_dof_id = np.max(self.DOF) + 1  # +1 to start the count enr dofs
            for e, conn in enumerate(self.CONN):
                # check if any enriched node is in conn
                if np.any(np.in1d(self.enriched_nodes, conn)):
                    self.enriched_elements.append(e)

                    # loop over enriched nodes in element
                    for n in np.intersect1d(self.enriched_nodes, conn):
                        # index of element enriched nodes in enriched nodes in
                        # enriched nodes list
                        ind, = np.where(self.enriched_nodes == n)[0]
                        self.DOF[e].append(ind*2 + max_dof_id)
                        self.DOF[e].append(ind*2 + max_dof_id + 1)

                    # change number of quadrature points
                    self.num_quad_points[e] = 4

            # add 2 new dofs for each enriched node
            self.num_dof += 2*len(self.enriched_nodes)

            # total number of enriched dofs for whole model
            self.num_enr_dof = 2*len(self.enriched_nodes)
            self.num_std_dof = self.num_dof - self.num_enr_dof


if __name__ == '__main__':
    class Mesh():
        pass
    mesh = Mesh()
    mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    mesh.CONN = np.array([[0, 1, 2, 3]])
    mesh.num_ele = 1
    mesh.DOF = [[0, 1, 2, 3, 4, 5, 6, 7]]
    mesh.num_dof = 8

    # center in (1, 1) and radius .2
    def func(x, y):
        # return (x - 1)**2 + (y - 1)**2 - 0.2**2
        return x - .5

    z_ls = Create(func, [0, 1], [0, 1], num_div=3)

    model = Build(mesh, zerolevelset=z_ls)
    # print(model.discontinuity_elements)
    # [0]
    # print(model.enriched_nodes)
    # [ 0  1  2  3]
    # print(model.enriched_elements)
    # [0]
