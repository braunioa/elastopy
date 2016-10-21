import numpy as np
from elastopy.model import Build
from elastopy.mesh import gmsh
from elastopy.material import Material
from elastopy.solvers import statics
from elastopy.postprocess import plotter

mesh_file = 'patch'

mesh = gmsh.Parse(mesh_file)

model = Build(mesh)

material = Material(E={9: 1000}, nu={9: 0.3})


def b_force(x1, x2, t=1):
    return np.array([0.0,
                     0.0])


def trac_bc(x1, x2, t=1):
    return {
        ('line', 3): [-1.0, 0.0],
        ('line', 1): [1.0, 0.0]}


def displ_bc(x1, x2):
    return {('node', 0): [0.0, 0.0],
            ('node', 1): ['free', 0.0]}

U, SIG = statics.solver(model, material, b_force,
                        trac_bc, displ_bc)

plotter.model(model, ele=True, nodes_label=True,
              ele_label=True, edges_label=True)
plotter.model_deformed(model, U, magf=1000, ele=True)


plotter.show()
