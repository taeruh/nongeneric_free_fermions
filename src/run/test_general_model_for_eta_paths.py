import itertools
import numpy as np
from sage.all import Graph
from sage.all import graphs  # pyright: ignore  (this is sage.graphs ...)

from rust_backend.paulis import Pauli, PauliSum
from hamiltonian import Hamiltonian

from models.fukai import Fukai
from models.fendley import Fendley
from models.weights import ConstantWeight, RandomWeight
from krylov_but_dont_know_whether_paths import Generators
import graph_helper


def run():

    # fendley = Fendley(
    #     4,
    #     RandomWeight(-1, 1),
    #     RandomWeight(-1, 1),
    #     RandomWeight(-1, 1),
    # )
    # graph = fendley.hamiltonian.get_frustration_graph()
    # simplicial_mode = (Pauli.from_indices(fendley.n, [0, 1], [1], 3), 2)
    # generators = Generators(
    #     (1.0, simplicial_mode[0]),
    #     fendley.hamiltonian,
    # )
    # generators.test_eta_path_decompositions(graph)
    # return

    fukai = Fukai(
        4,
        ConstantWeight(1.1),
        ConstantWeight(1.2),
        ConstantWeight(1.3),
        ConstantWeight(0),
    )
    fukai_graph = fukai.hamiltonian.get_frustration_graph()
    fukai_labeled_graph: Graph = fukai_graph.relabel(
        lambda x: f"({x}; {fukai.labels[x]}, {fukai.hamiltonian.weights[x]:.2f})",
        inplace=False,
    )
    fukai_labeled_graph.plot().save_image("output/fukai_graph.png")  # pyright: ignore
    simplicial_mode = (Pauli.from_indices(fukai.n, [0, 1], [1], 3), 2)
    generators = Generators(
        (1.0, simplicial_mode[0]),
        fukai.hamiltonian,
        max_search_eta_index=6
    )
    generators.test_eta_path_decompositions(fukai_graph)
