import numpy as np
from sage.all import Graph
from sage.all import graphs  # pyright: ignore  (this is sage.graphs ...)

from rust_backend.paulis import Pauli, PauliSum
from hamiltonian import Hamiltonian
from models.fendley import Fendley
from models.weights import ConstantWeight
from krylov import Generators



def run():
    fendley = Fendley(2, ConstantWeight(1), ConstantWeight(1), ConstantWeight(1))
    graph = fendley.hamiltonian.get_frustration_graph()
    simplicial_mode = fendley.example_simplicial_modes["IZZZZ"][0]
    generators = Generators((1.0, simplicial_mode), fendley.hamiltonian)
    generators.init_gammas(do_eigval_zero_check=False)
    generators.init_gamma_bilinears()

    # graph.plot().save_image("output/fendley_graph.png")  # pyright: ignore

    path_ops = []
    for path in graph.all_paths_iterator(simple=True):
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
            hermitian_phase = op.get_hermitian_phase()
            op.add_to_phase((-hermitian_phase) % 4)  # make them hermitian
            path_ops.append(op)

    count = 0
    for path in path_ops:
        count += 1
        # if not count == 6:
        #     continue
        projection = generators.bilinear_gamma_projection(path)
        # print(projection)
        if len(projection) == 0:
            print("no overlap")
            print()
            continue
        start = 0
        overlap = 0
        a, b, w = projection[start]
        overlap += w**2
        reconstructed = generators.gamma_bilinears[(a, b)].copy()
        reconstructed.multiply_with_float(w)
        for a, b, w in projection[start+1:]:
            overlap += w**2
            op = generators.gamma_bilinears[(a, b)].copy()
            op.multiply_with_float(w)
            reconstructed = reconstructed.add(op)
        overlap = np.sqrt(overlap)
        reconstructed.remove_zero_weights()
        alt_overlap = 0
        for w, _ in reconstructed.to_py_list():
            alt_overlap += w**2
        alt_overlap = np.sqrt(alt_overlap)
        print(f"overlap: {overlap}, {alt_overlap}")
        # print([(w, p.to_string()) for (w, p) in reconstructed.to_py_list()])
        print()
        # if count == 6:
        #     break
        if np.isclose(overlap, 1.0):
            print("PERFECT OVERLAP")
            print(path.to_string())
            print(projection)
            print([(w, p.to_string()) for (w, p) in reconstructed.to_py_list()])
            # break
