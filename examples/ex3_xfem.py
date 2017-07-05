import numpy as np
from elastopy import gmsh, Build, Material, statics, plotter

mesh_file = 'xfem'
mesh = gmsh.Parse(mesh_file)
model = Build(mesh)
material = Material(E={0: 2e6}, nu={0: 0.3}, case='strain')


def b_force(x1, x2, t=1):
    return np.array([0.0,
                     0.0])


def trac_bc(x1, x2, t=1):
    return {
        ('line', 3): [-1, 0],
        ('line', 1): [1, 0]}


def displ_bc(x1, x2):
    return {('node', 0): [0, 0],
            ('node', 1): ['free', 0]}

U, SIG = statics.solver(model, material, b_force,
                        trac_bc, displ_bc)

plotter.model(model, ele=True, nodes_label=True,
              ele_label=True, edges_label=True)
plotter.model_deformed(model, U, magf=100, ele=True)
plotter.stresses
plotter.show()
