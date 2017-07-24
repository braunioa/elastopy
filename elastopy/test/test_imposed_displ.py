import numpy as np
import pytest
from elastopy import gmsh, Build, Material, statics


def test_imposed_displ():
    mesh_file = 'examples/patch'

    mesh = gmsh.Parse(mesh_file)

    material = Material(E={9: 1000}, nu={9: 0.3})

    model = Build(mesh, material)

    def body_forces(x1, x2, t=1):
        return np.array([0.0, 0.0])

    def traction_imposed(x1, x2, t=1):
        return {}

    def displacement_imposed(x1, x2):
        return {
            ('nodes', 0, 3, 7): [0.0, 0.0],
            ('nodes', 4, 6): [0.5, 0.0],
            ('nodes', 1, 5, 2): [1.0, 0.0]}

    U, SIG = statics.solver(model, body_forces,
                            traction_imposed, displacement_imposed)

    assert pytest.approx(U[16], 1) == .4

