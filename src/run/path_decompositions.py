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


def run():
    fendley = Fendley(2, ConstantWeight(1), ConstantWeight(1), ConstantWeight(1))
    graph = fendley.hamiltonian.get_frustration_graph()
    # simplicial_mode = fendley.example_simplicial_modes["ZZZZZ"]
    # simplicial_mode = fendley.example_simplicial_modes["IZZZZ"]
    simplicial_mode = fendley.example_simplicial_modes["IIZZZ"]
    generators = Generators(
        (1.0, simplicial_mode[0]),
        fendley.hamiltonian,
    )
    generators.test_path_decompositions(graph)

    generators.init_gammas(do_eigval_zero_check=False)
    generators.init_gamma_bilinears()
    generators.init_eta_bilinears()

    # generators.init_eta_currents()

    for vec in generators.eta_vectors:
        for i, w in enumerate(vec):
            if w != 0:
                op = generators.eta_vector_to_pauli_map[i]
                path = generators.eta_vector_to_path_map[i]
                print(f"{w:5.1f} {op.to_string()} {path}")
        print()

    generators.init_eta_path_bilinears([i for i in range(simplicial_mode[1])], graph)
    # graph.plot().save_image("output/fendley_graph.pdf")  # pyright: ignore

    for i in range(generators.num_generators):
        for j in range(i + 1, generators.num_generators):
            print(i, j)
            print(
                [
                    (f"{w}", path, graph_helper.is_induced_path(graph, path[0]))
                    for (w, path) in generators.eta_path_bilinears[(i, j)]
                ]
            )
            print()

    return

    # print(
    #     [
    #         (w, op.to_string())
    #         for (w, op) in generators.gamma_bilinears[(0, 2)].to_py_list()
    #     ]
    # )
    # print()
    # print(
    #     [
    #         (w, op.to_string())
    #         for (w, op) in generators.eta_bilinears[(0, 2)].to_py_list()
    #     ]
    # )
    # print()

    # for current in generators.eta_currents:
    #     print([(w, p.to_string()) for (w, p) in current.to_py_list()])
    #     print()


    path_ops = []
    paths = []
    # for path in graph.all_paths_iterator(simple=True):
    for path in itertools.chain(
        [[vertex] for vertex in graph.vertices()], graph.all_paths_iterator(simple=True)
    ):
        if len(path) > 1 and path[0] == path[-1]:  # skip cycles
            continue
        is_induced = True
        for i in range(len(path)):
            for j in range(i + 2, len(path)):
                if graph.has_edge(path[i], path[j]):
                    is_induced = False
                    break
            if not is_induced:
                break
        if not is_induced:
            continue
        op = Pauli.identity(fendley.hamiltonian.n)
        for vertex in path:
            op = op.multiply_as_paulis(fendley.hamiltonian.operators[vertex])
        already_in = False
        for path_op in path_ops:
            if op.is_proportional_to(path_op):
                already_in = True
                break
        if not already_in:
            paths.append(path)
            hermitian_phase = op.get_hermitian_phase()
            op.add_to_phase((-hermitian_phase) % 4)  # make them hermitian
            path_ops.append(op)

    projections = []
    for path_op in path_ops:
        projections.append(generators.bilinear_gamma_projection(path_op))

    # allowed_lengths = set([2, 4])
    allowed_lengths = set([2, 2])
    # allowed_lengths = set([2])
    restricted_path_indices = []
    for i, path in enumerate(paths):
        if len(path) in allowed_lengths:
            restricted_path_indices.append(i)

    print(f"len(restricted_path_indices)={len(restricted_path_indices)}")

    weight = 1.0
    for choice in range(
        1, 2 ** (len(restricted_path_indices))
    ):  # don't need empty 0 choice
        indices = []
        for i, index in enumerate(restricted_path_indices):
            if (choice >> i) & 1:
                indices.append((weight, index))
        path_sum = PauliSum([(weight, path_ops[i]) for weight, i in indices])
        path_sum.remove_zero_weights()
        norm = 0.0
        for w, _ in path_sum.to_py_list():
            norm += w**2
        norm = np.sqrt(norm)
        path_projection = dict()
        for weight, i in indices:
            for a, b, w in projections[i]:
                if (a, b) not in path_projection:
                    path_projection[(a, b)] = 0.0
                path_projection[(a, b)] += w * weight
        reconstructed_norm = 0.0
        for w in path_projection.values():
            reconstructed_norm += w**2
        reconstructed_norm = np.sqrt(reconstructed_norm)
        # if abs(norm-reconstructed_norm) < 0.2:
        #     print(f"norms close...: {norm} vs {reconstructed_norm}")
        if np.isclose(norm, reconstructed_norm):
            path_projection = list(path_projection.items())
            print(f"choice: {bin(choice)}")
            print(
                f"path sum: {[(w, p.to_string()) for (w, p) in path_sum.to_py_list()]}"
            )
            print(f"path sum norm: {norm}")
            print(f"reconstructed norm: {reconstructed_norm}")
            print(path_projection)
            print([paths[i] for _, i in indices])
            (a, b), w = path_projection[0]
            reconstructed = generators.gamma_bilinears[(a, b)].copy()
            reconstructed.multiply_with_float(w)
            for (a, b), w in path_projection[1:]:
                op = generators.gamma_bilinears[(a, b)].copy()
                op.multiply_with_float(w)
                reconstructed = reconstructed.add(op)
            reconstructed.remove_zero_weights()
            # print(
            #     f"reconstructed: {[(w, p.to_string()) for (w, p) in reconstructed.to_py_list()]}"
            # )
            print()
