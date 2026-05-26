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
    fendley = Fendley(3, ConstantWeight(1), ConstantWeight(1), ConstantWeight(1))
    graph = fendley.hamiltonian.get_frustration_graph()
    simplicial_mode = fendley.example_simplicial_modes["IIYII"]
    # simplicial_mode = fendley.example_simplicial_modes["ZZZZZ"]
    # simplicial_mode = fendley.example_simplicial_modes["IZZZZ"]
    # simplicial_mode = fendley.example_simplicial_modes["IIZZZ"]
    # simplicial_mode = fendley.example_simplicial_modes["IIIIX"]
    generators = Generators(
        (1.0, simplicial_mode[0]),
        fendley.hamiltonian,
    )
    generators.test_path_decompositions(graph)

    generators.init_gammas(do_eigval_zero_check=False)
    generators.init_gamma_bilinears()
    generators.init_eta_bilinears()

    generators.init_eta_path_bilinears([i for i in range(simplicial_mode[1])], graph)
    # graph.plot().save_image("output/fendley_graph.pdf")  # pyright: ignore

    incorrect_paths = dict()
    len_incorrect_paths = 0
    # length 0 is the identity, which we generally allow
    allowed_lengths = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    # allowed_lengths = set([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

    def incorrect_filter(path) -> bool:
        if len(path) not in allowed_lengths:
            return False
        is_induced, is_path = graph_helper.is_induced_path(graph, path)
        if not (is_induced and is_path):
            return False
        return True

    for i in range(generators.num_generators):
        for j in range(i + 1, generators.num_generators):
            for _, (path, phase) in generators.eta_path_bilinears[(i, j)]:
                if not incorrect_filter(path):
                    tuple_path = tuple(path)  # pyright: ignore
                    if tuple_path not in incorrect_paths:
                        incorrect_paths[tuple_path] = (len_incorrect_paths, phase)
                        len_incorrect_paths += 1

    dim = len(incorrect_paths)
    all_okay_bilinears = []
    labels = []
    vectors = []
    for i in range(generators.num_generators):
        for j in range(i + 1, generators.num_generators):
            is_all_correct = True
            label = (i, j)
            vector = [int(0.0) for _ in range(dim)]
            for weight, (path, phase) in generators.eta_path_bilinears[(i, j)]:
                tuple_path = tuple(path)  # pyright: ignore
                value = incorrect_paths.get(tuple_path, None)
                if value is not None:
                    is_all_correct = False
                    index, fixed_phase = value
                    phase_diff = (phase - fixed_phase) % 4
                    assert phase_diff in [0, 2]
                    vector[index] = int(weight * (-1) ** (phase_diff // 2))
            if not is_all_correct:
                labels.append(label)
                vectors.append(vector)
            else:
                all_okay_bilinears.append(label)

            # print(label, vector)

    for i, (label, vector) in enumerate(zip(labels, vectors)):
        print(f"x{i}", label)
        print(vector)
        print()

    from sympy import Matrix, symbols, linsolve

    num_cols = len(vectors)
    system = Matrix(vectors)
    system = system.transpose()
    vars = symbols(f"x0:{num_cols}")
    solution = linsolve((system, Matrix.zeros(system.rows, 1)), vars)
    print(f"solution: {solution}")

    print(all_okay_bilinears)
    print(len(generators.etas))

    return

    b2 = generators.eta_bilinears[labels[2]]
    b0 = generators.eta_bilinears[labels[0]]

    b0.multiply_with_float(3.0)

    s = b2.add(b0)
    print([(w, p.to_string()) for (w, p) in s.to_py_list()])

    p2 = generators.eta_path_bilinears[labels[2]]
    p0 = generators.eta_path_bilinears[labels[0]]

    p0 = [(w * 3.0, (path, phase)) for w, (path, phase) in p0]

    s = generators.add_path_sums(p2, p0)
    print([(float(w), (p[0], p[1])) for (w, p) in s])
    print([(float(w), generators.path_to_operator(p).to_string()) for (w, p) in s])

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
    allowed_lengths = set([3, 3])
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
