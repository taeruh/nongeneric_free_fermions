from sage.all import Graph

from models.fendley import Fendley
from models.fukai import Fukai

from models.weights import ConstantWeight, RandomWeight


def run():
    # model = Fendley(3)
    model = Fukai(4, beta_3=ConstantWeight(0), beta_5=ConstantWeight(1))

    graph = model.hamiltonian.get_frustration_graph()
    labeled_graph: Graph = graph.relabel(
        lambda x: f"({model.labels[x]}, {model.hamiltonian.weights[x]:.2f})",
        inplace=False,
    )

    labeled_graph.plot().save_image("output/fendley_graph.png")  # pyright: ignore
