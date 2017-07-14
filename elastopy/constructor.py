"""Constructs the correct element object

"""
from .elements.quad4 import Quad4
from .elements.quad4enr import Quad4Enr


def constructor(eid, model, EPS0=None):
    """Function that constructs the correct element

    """
    type = model.TYPE[eid]

    if type == 3:
        if eid in model.enriched_elements:
            return Quad4Enr(eid, model, EPS0)
        else:
            return Quad4(eid, model, EPS0)
    else:
        raise Exception('Element not implemented yet!')
