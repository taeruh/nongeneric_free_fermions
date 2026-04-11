from sage.all import Graph

from models.fendley import Fendley
from models.fukai import Fukai
from models.weights import ConstantWeight, RandomWeight
from krylov import Generators


def run():
    model = Fendley(2)
    # model = Fukai(4, beta_3=ConstantWeight(3), beta_5=ConstantWeight(1))


    graph = model.hamiltonian.get_frustration_graph()
    labeled_graph: Graph = graph.relabel(
        lambda x: f"({model.labels[x]}, {model.hamiltonian.weights[x]:.2f})",
        inplace=False,
    )

    labeled_graph.plot().save_image("output/fendley_graph.png")  # pyright: ignore
    # for op in model.hamiltonian.operators:
    #     print(op.to_string())
    # print(model.example_simplicial_mode.to_string())

    generators = Generators(model.example_simplicial_mode, model.hamiltonian)
    print(f"Number of generators: {generators.num_generators}")
