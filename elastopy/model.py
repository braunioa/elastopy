"""Module for building the structure model

This module contains the class that creates the
structure model.
The structure model has all attributes from the mesh object.

"""
import numpy as np
import levelset
from elastopy.xfem.zerolevelset import Create


class Build(object):
    """Build the model object

    Args:
        mesh (object): object containing mesh data
        zerolevelset (optional): mask array with -1, 1 or any other
            value which zero level set defines discontinuity interface

    Attributes:
        mesh attributes
        enriched_elements (list): elements that are enriched

    """
    def __init__(self, mesh, zerolevelset=None):
        # copy attributes from mesh object
        self.__dict__ = mesh.__dict__.copy()

        self.enriched_elements = []
        if zerolevelset is not None:
            # phi shape (nn, ) with signed distance value
            phi = levelset.distance(zerolevelset.mask_ls,
                                    zerolevelset.grid_x,
                                    zerolevelset.grid_y,
                                    self.XYZ)

            for e, conn in enumerate(self.CONN):
                # check if element is enriched or not
                if np.all(phi[conn] < 0) or np.all(
                        phi[conn] > 0):
                    pass
                else:
                    self.enriched_elements.append(e)


if __name__ == '__main__':
    class Mesh():
        pass
    mesh = Mesh()
    mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    mesh.CONN = np.array([[0, 1, 2, 3]])

    # center in (1, 1) and radius .2
    def func(x, y):
        return (x - 1)**2 + (y - 1)**2 - .2**2
    z_ls = Create(func, [0, 1], [0, 1], num_div=50)

    model = Build(mesh, zerolevelset=z_ls)
    # print(model.enriched_elements)
    # [0]
