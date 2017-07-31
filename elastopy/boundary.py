"""Apply the boundary conditions to matrices and vectors"""


def dirichlet(K, F, model, displ_bc=None):
    """Apply Dirichlet BC.

    Note:

        How its done:

        1. Loop over the lines where Dirichlet boundary conditions are
        applied. This is specified as key argument of the dictionary. The
        boundaries are defined on the gmsh nad tagged with a number.

        2. loop over all the nodes at the boundaries.

        3. If those nodes, identified as follows::

            nodes_in_bound_line = [line node1 node2]

            are in the line where dirichlet BC were specified, change the
            stiffness matrix and the B vector based on this node index.

    The assignment can be done in two ways:

    1. with a dictionary specifying the boundary line:

        ['line', line]:[dx, dy]

    which means: the boundary line with tag line has a displ_bc dx on
    the x-direction and dy on the y-direction.

    2. with a dictionary with a specific node:

        ['node', node]:[0, 1]

    this way: the string 'node'identifies the type of boundary displ_bc
    that is been assigned; the number 0 identifies a free to move degree of
    freedom and 1 a restrained one. The first entry is the x-direction and
    the second y-direction.

    Args:
        K (2nd order array): Stiffness matrix.
        B (1st order array): Vector with the load and traction.
        temperature (function): Function with 4 components.

    Returns:
        K (2nd order array), B (1st order array): Modified stiffness matrix
        and vector.

    """
    if displ_bc is not None:
        for line in displ_bc(1, 1).keys():
            if isinstance(line, tuple) and line[0] == 'line':
                for n in range(len(model.nodes_in_bound_line[:, 0])):
                    if line[1] == model.nodes_in_bound_line[n, 0]:
                        rx = displ_bc(1, 1)[line][0]
                        ry = displ_bc(1, 1)[line][1]

                        # nodes indexes on the element boundary line

                        n1 = model.nodes_in_bound_line[n, 1]
                        n2 = model.nodes_in_bound_line[n, 2]

                        d1 = displ_bc(
                            model.XYZ[n1, 0],
                            model.XYZ[n1, 1])

                        d2 = displ_bc(
                            model.XYZ[n2, 0],
                            model.XYZ[n2, 1])

                        if rx != 'free' and rx is not None:
                            K[2 * n1, :] = 0.0
                            K[2 * n2, :] = 0.0
                            K[2 * n1, 2 * n1] = 1.0
                            K[2 * n2, 2 * n2] = 1.0
                            F[2 * n1] = d1[line][0]
                            F[2 * n2] = d2[line][0]

                        if ry != 'free' and ry is not None:
                            K[2 * n1 + 1, :] = 0.0
                            K[2 * n2 + 1, :] = 0.0
                            K[2 * n1 + 1, 2 * n1 + 1] = 1.0
                            K[2 * n2 + 1, 2 * n2 + 1] = 1.0
                            F[2 * n1 + 1] = d1[line][1]
                            F[2 * n2 + 1] = d2[line][1]
            elif isinstance(line, int):
                for n in range(len(model.nodes_in_bound_line[:, 0])):
                    if line == model.nodes_in_bound_line[n, 0]:
                        rx = displ_bc(1, 1)[line][0]
                        ry = displ_bc(1, 1)[line][1]

                        # nodes indexes on the element boundary line

                        n1 = model.nodes_in_bound_line[n, 1]
                        n2 = model.nodes_in_bound_line[n, 2]

                        d1 = displ_bc(
                            model.XYZ[n1, 0],
                            model.XYZ[n1, 1])

                        d2 = displ_bc(
                            model.XYZ[n2, 0],
                            model.XYZ[n2, 1])

                        if rx != 'free' and rx is not None:
                            K[2 * n1, :] = 0.0
                            K[2 * n2, :] = 0.0
                            K[2 * n1, 2 * n1] = 1.0
                            K[2 * n2, 2 * n2] = 1.0
                            F[2 * n1] = d1[line][0]
                            F[2 * n2] = d2[line][0]

                        if ry != 'free' and ry is not None:
                            K[2 * n1 + 1, :] = 0.0
                            K[2 * n2 + 1, :] = 0.0
                            K[2 * n1 + 1, 2 * n1 + 1] = 1.0
                            K[2 * n2 + 1, 2 * n2 + 1] = 1.0
                            F[2 * n1 + 1] = d1[line][1]
                            F[2 * n2 + 1] = d2[line][1]

            if isinstance(line, tuple) and (line[0] == 'nodes' or line[0] == 'node'):

                rx = displ_bc(1, 1)[line][0]
                ry = displ_bc(1, 1)[line][1]

                for n in line[1:]:
                    if rx != 'free':
                        K[2 * n, :] = 0.0
                        K[2 * n, 2 * n] = 1.0
                        F[2 * n] = rx

                    if ry != 'free':
                        K[2 * n + 1, :] = 0.0
                        K[2 * n + 1, 2 * n + 1] = 1.0
                        F[2 * n + 1] = ry

            # modify Linear system matrix and vector for imposed 0.0 displ_bc
            if isinstance(line, tuple) and line[0] == 'support':

                for n in line[1:]:
                    rx = displ_bc(1, 1)[line][0]
                    ry = displ_bc(1, 1)[line][1]
                    if rx != 'free' and rx != 0:
                        K[2 * n, :] = 0.0
                        K[2 * n, 2 * n] = 1.0
                        F[2 * n] = 0.0

                    if ry != 'free' and rx != 0:
                        K[2 * n + 1, :] = 0.0
                        K[2 * n + 1, 2 * n + 1] = 1.0
                        F[2 * n + 1] = 0.0

    return K, F
