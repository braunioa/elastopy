"""Constructs the correct element object

"""
from elastopy.elements.quad4 import Quad4


def constructor(eid, model, material, EPS0):
    """Function that constructs the correct element

    """
    type = model.TYPE[eid]

    if type == 3:
        return Quad4(eid, model, material, EPS0)
    else:
        raise Exception('Element not implemented yet!')
