"""Module for building the structure model

This module contains the class that creates the
structure model.
The structure model has all attributes from the mesh object.

"""


class Build(object):
    """Build the model object

    """
    def __init__(self, mesh):
        # copy attributes from mesh object
        self.__dict__ = mesh.__dict__.copy()
