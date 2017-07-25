import numpy as np
from elastopy.xfem.zerolevelset import Create
from elastopy.model import Build


def test_1zerolevelset():
    class Mesh():
        pass
    mesh = Mesh()
    mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    mesh.CONN = np.array([[0, 1, 2, 3]])
    mesh.num_ele = 1
    mesh.DOF = [[0, 1, 2, 3, 4, 5, 6, 7]]
    mesh.num_dof = 8

    def func(x, y):
        return x - .5
    zls = Create(func, [0, 1], [0, 1], num_div=3)

    model = Build(mesh, zerolevelset=[zls])
    assert (model.zerolevelset[0].phi == [-.5, .5, .5, -.5]).all()
    assert model.zerolevelset[0].enriched_nodes == [0, 1, 2, 3]
    assert model.num_enr_dof == 8


def test_2zerolevelset():
    class Mesh():
        pass
    mesh = Mesh()
    mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                         [2, 0], [2, 1],
                         [2, 2], [1, 2],
                         [0, 2]])
    mesh.CONN = np.array([[0, 1, 2, 3], [1, 4, 5, 2], [2, 5, 6, 7],
                          [3, 2, 7, 8]])
    mesh.num_ele = 4
    mesh.DOF = [[0, 1, 2, 3, 4, 5, 6, 7],
                [2, 3, 8, 9, 4, 5, 10, 11],
                [4, 5, 10, 11, 12, 13, 14, 15],
                [6, 7, 4, 5, 14, 15, 16, 17]]
    mesh.num_dof = 18

    def func(x, y):
        return (x-1)**2 + (y-1)**2 - .5**2
    zls = Create(func, [0, 2], [0, 2], num_div=10)
    model = Build(mesh, zerolevelset=[zls])

    assert (model.zerolevelset[0].enriched_nodes ==
            [0, 1, 2, 3, 4, 5, 6, 7, 8])
    print(model.zerolevelset[0].enriched_elements)
    assert model.zerolevelset[0].enriched_elements == [0, 1, 2, 3]
    assert model.enriched_elements == [0, 1, 2, 3]


def test_3zerolevelset():
    class Mesh():
        pass
    mesh = Mesh()
    mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                         [2, 0], [2, 1]])
    mesh.CONN = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
    mesh.num_ele = 2
    mesh.DOF = [[0, 1, 2, 3, 4, 5, 6, 7],
                [2, 3, 8, 9, 10, 11, 4, 5]]
    mesh.num_dof = 12

    fun1 = lambda x, y: x - .5
    fun2 = lambda x, y: (-x + 1.5)
    zls = [Create(fun1, [0, 2], [0, 1], num_div=10),
           Create(fun2, [0, 2], [0, 1], num_div=10)]

    model = Build(mesh, zerolevelset=zls)
    assert set(model.zerolevelset[0].enriched_nodes) == set([0, 1, 2, 3])
    assert set(model.zerolevelset[1].enriched_nodes) == set([1, 2, 4, 5])
    assert model.zerolevelset[0].discontinuity_elements == [0]
    assert model.zerolevelset[1].discontinuity_elements == [1]

    # includes dof for each element and for each zero level set
    # the order is defined by the zero level set order
    # and by the enriched nodes order
    assert model.DOF == [[0, 1, 2, 3, 4, 5, 6, 7,
                          12, 13, 14, 15, 16, 17, 18, 19,
                          20, 21, 22, 23],
                         [2, 3, 8, 9, 10, 11, 4, 5,
                          14, 15, 16, 17,
                          20, 21, 22, 23, 24, 25, 26, 27]]
    # numbering order defined by enriched nodes
    assert model.zerolevelset[0].enriched_dof[0] == [12, 13, 14, 15, 16, 17,
                                                     18, 19]
    assert model.zerolevelset[0].enriched_dof[1] == [14, 15, 16, 17]
    assert model.zerolevelset[1].enriched_dof[0] == [20, 21, 22, 23]
    assert model.num_enr_dof == 2*8
    assert model.enriched_elements == [0, 1]


def test_4zerolevelset():
    class Mesh():
        pass
    mesh = Mesh()
    mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                         [2, 0], [2, 1],
                         [3, 0], [3, 1],
                         [3, 2], [2, 2], [1, 2], [0, 2]])
    mesh.CONN = np.array([[0, 1, 2, 3], [1, 4, 5, 2],
                          [4, 6, 7, 5], [5, 7, 8, 9],
                          [2, 5, 9, 10], [3, 2, 10, 11]])
    mesh.num_ele = 6
    mesh.DOF = [[0, 1, 2, 3, 4, 5, 6, 7],
                [2, 3, 8, 9, 10, 11, 4, 5],
                [8, 9, 12, 13, 14, 15, 10, 11],
                [10, 11, 14, 15, 16, 17, 18, 19],
                [4, 5, 10, 11, 18, 19, 20, 21],
                [9, 10, 4, 5, 20, 21, 22, 23]]

    mesh.num_dof = 22

    fun = lambda x, y: (x-3)**2 + (y-2)**2 - .3**2
    zls = [Create(fun, [0, 3], [0, 2], num_div=15)]

    model = Build(mesh, zerolevelset=zls)
    assert model.zerolevelset[0].discontinuity_elements == [3]
    assert set(model.zerolevelset[0].enriched_nodes) == set([8, 9, 5, 7])
    assert set(model.zerolevelset[0].enriched_elements) == set([1, 2, 3, 4])
    # dof numbering according to enriched nodes order
    assert model.zerolevelset[0].enriched_dof[3] == [28, 29, 30, 31,
                                                     24, 25, 26, 27]
    assert model.enriched_elements == [1, 2, 3, 4]
    assert set(model.enriched_nodes) == set([5, 7, 8, 9])
