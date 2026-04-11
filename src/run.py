import numpy as np
from sage.all import Graph

from models.fendley import Fendley
from models.fukai import Fukai
from models.weights import ConstantWeight, RandomWeight
from krylov import Generators


def run():
    num_triangles = 2
    fendley = Fendley(num_triangles)
    fukai = Fukai(num_triangles, beta_3=ConstantWeight(1), beta_5=ConstantWeight(1))

    graph = fukai.hamiltonian.get_frustration_graph()
    labeled_graph: Graph = graph.relabel(
        lambda x: f"({fukai.labels[x]}, {fukai.hamiltonian.weights[x]:.2f})",
        inplace=False,
    )
    labeled_graph.plot().save_image("output/fukai_graph.png")  # pyright: ignore

    generators = Generators(fendley.example_simplicial_mode, fendley.hamiltonian)
    print(f"Number of generators: {generators.num_generators}")

    print([(w, op.to_string()) for op, w in zip(fukai.fukai_ops, fukai.fukai_weights)])

    reconstructed = []
    # for fukai_op in fukai.fukai_ops:
    for fukai_op in fukai.fukai_ops[:1]:
        projection = generators.bilinear_gamma_projection(fukai_op)

        inner_product = 0

        for i, j, coeff in projection:
            inner_product += coeff**2
            print(coeff)
            part = generators.gamma_bilinears[(i, j)]
            for w, op in part:
                already_in = False
                for i, (rw, rop) in enumerate(reconstructed):
                    if op.is_proportional_to(rop):
                        phase = op.phase_difference(rop)
                        reconstructed[i] = (rw + coeff * w * (1j) ** phase, rop)
                        already_in = True
                        break
                if not already_in:
                    reconstructed.append((coeff * w, op))

        print(inner_product)
        print()

    to_remove = []
    for i, (coeff, op) in enumerate(reconstructed):
        if np.isclose(coeff, 0):
            to_remove.append(i)
    for i in reversed(to_remove):
        del reconstructed[i]

    print([f"{coeff:.4f} * {op.to_string()}" for coeff, op in reconstructed])
