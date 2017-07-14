"""creates an object with the zero level set definition"""
import numpy as np


class Create(object):
    """creates the zero level set object with its attributes

    Args:
        function (func(x, y)): function that defines the level set
            the zero level set defines the discontinuity interface
        x_domain (list): domian defined by (x_domain[0], x_domain[1])
        y_domain (list): same as x
        num_div (float, optional): number of division for defining the
            level set, the greater the value more precisa the interface
            will be defined

    Attributes:
        grid_x (numpy array): 2d array shape (num_div, num_div)
            with grid value for x direction
        grid_y (numpy array): same as x for y direction
        mask_ls (numpy array): 2d array shape (num_div, num_div) with
            -1 and 1 where the points between these two values define the
            discontinuity interface.
    """
    def __init__(self, function, x_domain, y_domain, num_div=50):
        self.grid_x, self.grid_y = np.meshgrid(np.linspace(x_domain[0],
                                                           x_domain[1],
                                                           num_div),
                                               np.linspace(y_domain[0],
                                                           y_domain[1],
                                                           num_div))
        ls = function(self.grid_x, self.grid_y)
        self.mask_ls = ls/abs(ls)

        # define material parameters
        self.region = [-1, 1]


if __name__ == '__main__':
    def func(x, y):
        return (x - 2)**2 + (y - 2)**2 - 1.8**2
    z_ls = Create(func, [0, 2], [0, 2], num_div=3)
    # print(z_ls.mask_ls)
    # [[ 1.  1.  1.]
    #  [ 1. -1. -1.]
    #  [ 1. -1. -1.]]
