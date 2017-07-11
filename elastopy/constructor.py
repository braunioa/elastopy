"""Constructs the correct element object

"""
from elastopy.elements.quad4 import Quad4
from elastopy.elements.quad4enr import Quad4Enr


def constructor(eid, model, material, EPS0):
    """Function that constructs the correct element

    """
    type = model.TYPE[eid]

    if type == 3:
        if eid in model.enriched_elements:
            return Quad4Enr(eid, model, material, EPS0)
        else:
            return Quad4(eid, model, material, EPS0)
    else:
        raise Exception('Element not implemented yet!')
