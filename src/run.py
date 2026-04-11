from sage.all import Graph

from models.fendley import Fendley
from models.fukai import Fukai
from models.weights import ConstantWeight, RandomWeight
from krylov import Generators


def run():
    num_triangles = 3
    fendley = Fendley(num_triangles)
    fukai = Fukai(num_triangles, beta_3=ConstantWeight(1), beta_5=ConstantWeight(1))

    graph = fendley.hamiltonian.get_frustration_graph()
    labeled_graph: Graph = graph.relabel(
        lambda x: f"({fendley.labels[x]}, {fendley.hamiltonian.weights[x]:.2f})",
        inplace=False,
    )

    labeled_graph.plot().save_image("output/fendley_graph.png")  # pyright: ignore
    # for op in model.hamiltonian.operators:
    #     print(op.to_string())
    # print(model.example_simplicial_mode.to_string())

    generators = Generators(fendley.example_simplicial_mode, fendley.hamiltonian)
    print(f"Number of generators: {generators.num_generators}")

    x = fukai.fukai_ops[1]
    y = generators.gamma_projection(x)
    print(x.to_string())
    print(y)

    foundit = False
    for eta in generators.etas:
        for _, op in eta:
            # print(op.to_string())
            if x.is_proportional_to(op):
                foundit = True
                break

    if foundit:
        print("FOUND IT")
    else:
        print("DID NOT FIND IT")
