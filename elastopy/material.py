"""Module that contains material properties in a class"""


class Material(object):
    """Instanciate a material object

    Args:
        case (str): which case is the constitutive model

    Attributes:
        mat_dic keywords (dict): dictionary with surface label and
            material paramenter value.

            example:

                Material(E={9: 1000})

          then, the keyword `E` is the attribute self.E with value
          {9: 1000} in which 9 is the surface label and 1000 is the
          E value at that surface

    Note:
        If case is strain, then we use the standard transformations
        in the material parameters, E and nu, in order to avoid
        changing the construction of the constitutive matrix.

    """
    def __init__(self, case='stress', **mat_dic):
        self.__dict__.update(mat_dic)
        self.case = case
        # convert from plane stress to plane strain
        if case is 'strain':
            for surf, E in self.E.items():
                self.E[surf] = self.E[surf]/(1 - self.nu[surf]**2)
            for surf, nu in self.E.items():
                self.nu[surf] = self.nu[surf]/(1 - self.nu[surf])
