import numpy as np
import pytest
from elastopy import gmsh, Build, Material
from elastopy.constructor import constructor


def simulation():
    mesh_file = 'examples/patch'
    mesh = gmsh.Parse(mesh_file)
    material = Material(E={9: 1000}, nu={9: 0.3})
    model = Build(mesh, material)
    EPS0 = None
    t = 1

    def body_force(x1, x2, t=1):
        return np.array([0.0, 0.0])

    def traction_bc(x1, x2, t=1):
        return {('line', 3): [-1, 0], ('line', 1): [1, 0]}

    K = np.zeros((model.num_dof, model.num_dof))
    Pb = np.zeros(model.num_dof)
    Pt = np.zeros(model.num_dof)
    Pe = np.zeros(model.num_dof)

    for eid, type in enumerate(model.TYPE):
        element = constructor(eid, model, EPS0)

        k = element.stiffness_matrix(t)
        pb = element.load_body_vector(body_force, t)
        pt = element.load_traction_vector(traction_bc, t)
        pe = element.load_strain_vector(t)

        K[element.id_m] += k
        Pb[element.id_v] += pb
        Pt[element.id_v] += pt
        Pe[element.id_v] += pe

    return K, Pt


def test_tractionbc():
    _, Pt = simulation()
    assert (Pt == np.array([
        -0.25, 0., 0.25, 0., 0.25, 0., -0.25, 0., 0., 0., 0.5, 0., 0., 0.,
        -0.5, 0., 0., 0.
    ])).all()


def test_stiffness():
    K, _ = simulation()
    assert pytest.approx(np.linalg.norm(K), 4) == 5230.11249876
