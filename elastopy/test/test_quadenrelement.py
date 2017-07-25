import numpy as np
from elastopy import gmsh, Build, Material
from elastopy.constructor import constructor
from elastopy import xfem
from elastopy.xfem.zerolevelset import Create

np.set_printoptions(precision=3, suppress=True)


def test_1zls():
    def b_force(x1, x2, t=1):
        return np.array([0.0, 0.0])

    def trac_bc(x1, x2, t=1):
        return {('line', 1): [2e7, 0]}  # kg/m

    def displ_bc(x1, x2):
        return {('node', 0): [0, 0], ('node', 3): [0, 0]}

    EPS0 = None
    func = lambda x, y: x - 0.3
    zls = xfem.zerolevelset.Create(func, [0, .6], [0, .2], num_div=100)
    mesh = gmsh.Parse('examples/xfem_bimaterial')
    material = Material(
        E=[{
            zls.region['reinforcement']: 2e11,
            zls.region['matrix']: 1e11
        }],
        nu=[{
            zls.region['reinforcement']: .3,
            zls.region['matrix']: .3
        }],
        case='strain')  # kg/m2
    model = Build(mesh, material=material, zerolevelset=[zls], thickness=.01)
    K = np.zeros((model.num_dof, model.num_dof))
    k = {}
    for eid, type in enumerate(model.TYPE):
        element = constructor(eid, model, EPS0)
        k[eid] = element.stiffness_matrix()
        K[element.id_m] += k[eid]

    kstdele0 = np.array(
        [[1.154, 0.481, -0.769, 0.096, -0.577, -0.481, 0.192, -0.096],
         [0.481, 1.154, -0.096, 0.192, -0.481, -0.577, 0.096, -0.769],
         [-0.769, -0.096, 1.154, -0.481, 0.192, 0.096, -0.577, 0.481],
         [0.096, 0.192, -0.481, 1.154, -0.096, -0.769, 0.481, -0.577],
         [-0.577, -0.481, 0.192, -0.096, 1.154, 0.481, -0.769, 0.096],
         [-0.481, -0.577, 0.096, -0.769, 0.481, 1.154, -0.096, 0.192],
         [0.192, 0.096, -0.577, 0.481, -0.769, -0.096, 1.154, -0.481],
         [-0.096, -0.769, 0.481, -0.577, 0.096, 0.192, -0.481, 1.154]])
    assert np.allclose(k[0][:8, :8] / 1e9, kstdele0, rtol=1e-2, atol=1e-2)
    kenrele0 = np.array(
        [[129.915, 0., 49.573, -0.], [-0., 70.085, -0., -18.803],
         [49.573, -0., 129.915, 0.], [0., -18.803, 0., 70.085]])
    assert np.allclose(k[0][8:, 8:]/1e9*3600, kenrele0, rtol=1e-2, atol=1e-2)


def test_2zls():
    """Plate with 2 zls, left with E=2e11 and right E=5e11
    2 elements
    check how two or more level sets are dealt
    """

    class Mesh():
        pass

    mesh = Mesh()
    mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]])
    mesh.CONN = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
    mesh.num_ele = 2
    mesh.DOF = [[0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 8, 9, 10, 11, 4, 5]]
    mesh.num_dof = 12
    mesh.TYPE = [3, 3]
    mesh.surf_of_ele = [0, 0]
    mesh.bound_ele = np.array([[0, 0, 0], [0, 2, 2], [0, 3, 3], [1, 0, 0],
                               [1, 1, 1], [1, 2, 2]])
    mesh.nodes_in_bound_line = np.array([[0, 0, 1], [0, 1, 4], [1, 4, 5],
                                         [2, 5, 2], [2, 2, 3], [3, 3, 0]])
    fun1 = lambda x, y: x - .5
    fun2 = lambda x, y: (-x + 1.5)
    zls = [
        Create(
            fun1, [0, 2], [0, 1], num_div=10), Create(
                fun2, [0, 2], [0, 1], num_div=10)
    ]
    material = Material(
        E=[{
            zls[0].region['reinforcement']: 2e11,
            zls[0].region['matrix']: 1e11
        }, {
            zls[1].region['reinforcement']: 5e11,
            zls[1].region['matrix']: 1e11
        }],
        nu=[{
            zls[0].region['reinforcement']: .3,
            zls[0].region['matrix']: .3
        }, {
            zls[1].region['reinforcement']: .3,
            zls[1].region['matrix']: .3
        }],
        case='strain')  # kg/m2
    model = Build(mesh, material=material, zerolevelset=zls, thickness=.01)
    assert set(model.zerolevelset[0].enriched_nodes) == set([0, 1, 2, 3])
    assert set(model.zerolevelset[1].enriched_nodes) == set([1, 2, 4, 5])

    EPS0 = None
    K = np.zeros((model.num_dof, model.num_dof))
    k = {}
    element = {}
    for eid, type in enumerate(model.TYPE):
        element[eid] = constructor(eid, model, EPS0)

        k[eid] = element[eid].stiffness_matrix()
        K[element[eid].id_m] += k[eid]
    # 3x8 standard 3x8 due one level set and 3x4 due the other
    assert np.shape(k[0]) == (20, 20)
    assert np.allclose(element[0].E, np.array([219780219780.2198,
                                               109890109890.1099,
                                               109890109890.1099,
                                               219780219780.2198]),
                       rtol=1e-2, atol=1e-2)
    assert np.allclose(element[1].E, np.array([109890109890.1099,
                                               549450549450.54944,
                                               549450549450.54944,
                                               109890109890.1099]),
                       rtol=1e-2, atol=1e-2)


test_2zls()
