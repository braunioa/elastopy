import numpy as np
from elastopy import gmsh, Build, Material, statics, plotter

mesh_file = 'patch'

mesh = gmsh.Parse(mesh_file)

model = Build(mesh)

material = Material(E={9: 1000}, nu={9: 0.3})


def body_forces(x1, x2, t=1):
    return np.array([0.0, 0.0])


def traction_imposed(x1, x2, t=1):
    return {}


def displacement_imposed(x1, x2):
    return {
        ('nodes', 0, 3, 7): [0.0, 0.0],
        ('nodes', 4, 6): [0.5, 0.0],
        ('nodes', 1, 5, 2): [1.0, 0.0]}

U, SIG = statics.solver(model, material, body_forces,
                        traction_imposed, displacement_imposed)

plotter.model(model, ele=True, nodes_label=True, edges_label=True,
              ele_label=True)
plotter.model_deformed(model, U, magf=0.1, ele=True)
print(SIG[:, 0])

plotter.show()
