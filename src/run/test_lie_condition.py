import itertools
import numpy as np
from sage.all import Graph
from sage.all import graphs  # pyright: ignore  (this is sage.graphs ...)

from rust_backend.paulis import Pauli, PauliSum
from hamiltonian import Hamiltonian
from models.fendley import Fendley
from models.weights import ConstantWeight
from krylov import Generators
import graph_helper

import matplotlib.pyplot as plt
import itertools


def run():
    mat = 2 * get_mat(1, 1, 1, 1)
    print(mat)
    delta = det(mat)
    print(delta)
    return

    a_values = np.linspace(-10, 10, 20)
    b_values = np.linspace(-10, 10, 20)
    c_values = np.linspace(-10, 10, 20)
    d = 1.0

    # remove zero from a_values, b_values, c_values
    a_values = a_values[a_values != 0]
    b_values = b_values[b_values != 0]
    c_values = c_values[c_values != 0]

    y = []
    max = -float("inf")
    maxvals = (0, 0, 0)

    for a, b, c in itertools.product(a_values, b_values, c_values):
        print(a, b, c)
        mat = get_mat(a, b, c, d)
        delta = det(mat)
        y.append(delta)
        if delta > max:
            max = delta
            maxvals = (a, b, c)
        if delta == 0:
            print("Lie condition not satisfied for a =", a, "b =", b, "c =", c)
            break
        # if np.isclose(delta, 0):
        #     print(
        #         "Lie condition approximately not satisfied for a =",
        #         a,
        #         "b =",
        #         b,
        #         "c =",
        #         c,
        #         delta,
        #     )
        #     break

    print(max, maxvals)

    fig = plt.figure()
    # plot a 3d scatter plot of a, b, c, and y
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        [a for a, b, c in itertools.product(a_values, b_values, c_values)],
        [b for a, b, c in itertools.product(a_values, b_values, c_values)],
        [c for a, b, c in itertools.product(a_values, b_values, c_values)],
        c=y,
        cmap="viridis",
    )
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("c")
    cbar = plt.colorbar(ax.collections[0], ax=ax, pad=0.1)
    cbar.set_label("determinant")
    plt.savefig("output/lie_condition.pdf")


def get_mat(a, b, c, d):
    fendley = Fendley(2, ConstantWeight(a), ConstantWeight(b), ConstantWeight(c))

    graph = fendley.hamiltonian.get_frustration_graph()
    # simplicial_mode = fendley.example_simplicial_modes["IIYII"]
    simplicial_mode = fendley.example_simplicial_modes["ZZZZZ"]
    # simplicial_mode = fendley.example_simplicial_modes["IZZZZ"]
    # simplicial_mode = fendley.example_simplicial_modes["IIZZZ"]
    # simplicial_mode = fendley.example_simplicial_modes["IIIIX"]
    generators = Generators(
        (d, simplicial_mode[0]),
        fendley.hamiltonian,
    )
    generators.test_path_decompositions(graph)
    return generators.init_gammas(do_eigval_zero_check=False)


def det(mat):
    c0 = mat[0][0]
    c2 = mat[1][1]
    c4 = mat[2][2]
    c6 = mat[3][3]
    return c2**2 * (c0 * c4 - c2**2) - c0**2 * (c2 * c6 - c4**2)
