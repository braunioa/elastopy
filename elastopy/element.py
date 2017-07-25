"""Creates an element object with basic attributes

"""
import numpy as np


class Element(object):
    """Build an Element base clase

    Args:
        eid (int): element index
        model (obj): modle object

    Attributes:
        eid
        type
        conn (numpy array): list with nodes of element
        xyz
        dof
        num_std_dof
        enriched_nodes (numpy array): nodoes in this element (global tag)
            that are enriched.
        num_enr_nodes
        num_std_nodes (int): number of standard nodes
        num_enr_dof
        surf
        id_m
        id_v
        zerolevelset

    """
    def __init__(self, eid, model):
        self.eid = eid
        self.type = model.TYPE[eid]
        self.conn = model.CONN[eid]
        self.xyz = model.XYZ[self.conn]

        self.dof = model.DOF[eid]
        self.num_std_dof = 2*len(self.conn)
        self.num_std_nodes = len(self.conn)

        # depends on zero level set
        self.num_enr_dof = len(self.dof) - self.num_std_dof

        self.zerolevelset = model.zerolevelset

        self.surf = model.surf_of_ele[eid]

        self.id_m = np.ix_(self.dof, self.dof)
        self.id_v = self.dof

        self.num_quad_points = model.num_quad_points[eid]
        self.thickness = model.thickness

        # enriched nodes shape (num zerolvlset, num enr nodes)
        self.enriched_nodes = []
        for zls in model.zerolevelset:
            self.enriched_nodes.append(np.intersect1d(zls.enriched_nodes,
                                                      self.conn))
