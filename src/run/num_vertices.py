import os
import json
import numpy as np
from scipy import optimize
from sage.all import graphs  # pyright: ignore  (this is sage.graphs ...)
import matplotlib.pyplot as plt

from models.fendley import Fendley
from models.weights import ConstantWeight, RandomWeight
from krylov import Generators

# observations:
# - while the coefficients in the etas and eta_currents obviously depend on alpha,
#   beta, gamma and currents_alpha, the operators themselves seem to be independent
#   of those parameters
# - changing the simplicial mode changes the operators of course, however, the graph
#   and the coefficients seem to stay the same between different modes from the same
#   connectivity to the graph (i.e., connect to one vertex, connect to two vertices,
#   connect to three vertices)


def run():
    low_num_triangles = 1
    up_num_triangles = 10

    # weight = 1
    # seed = 3
    # seed = None
    alpha = ConstantWeight(np.float64(1))
    beta = ConstantWeight(np.float64(1))
    gamma = ConstantWeight(np.float64(1))
    currents_alpha = ConstantWeight(np.float64(1))
    # simplicial_mode_choices = ["IIZZZ", "IZZZZ", "ZZZZZ"]
    simplicial_mode_choices = ["IIZZZ"]

    load_data = False
    # load_data = True

    os.makedirs("output/currents", exist_ok=True)
    file_identifier = (
        f"num_vertices_{alpha}_{beta}_{gamma}_{currents_alpha}"
        + "-".join(simplicial_mode_choices)
    )
    data_file = f"output/currents/data_{file_identifier}.json"
    plot_file = f"output/currents/plot_{file_identifier}.pdf"

    if load_data:
        with open(data_file, "rb") as f:
            data = json.load(f)
            all_num_vertices = data["all_num_vertices"]
    else:
        all_num_vertices = []
        for simplicial_mode_choice in simplicial_mode_choices:
            get_generators = lambda tolerance: Generators(
                simplicial_mode,
                fendley.hamiltonian,
                eta_normalisation_factor=np.float64(
                    len(fendley.hamiltonian.operators)
                    / fendley.hamiltonian.pauli_l1_norm
                    / (np.sqrt(np.sqrt(2.7)))
                ),
                renormalise=True,
                orthogonal_tolerance=tolerance,
                max_search=expected_rank - 1,
            )

            num_vertices = []
            for num_triangles in range(low_num_triangles, up_num_triangles + 1):
                expected_rank = 2 * num_triangles + 1
                print(f"Processing num_triangles={num_triangles}...")
                fendley = Fendley(num_triangles, alpha, beta, gamma)
                simplicial_mode, mode_neighbours = fendley.example_simplicial_modes[
                    simplicial_mode_choice
                ]
                simplicial_mode = (1.0, simplicial_mode)
                if mode_neighbours == 3:
                    # in this case it is one generator less, probably, since the graph,
                    # without the simplicial clique
                    # TODO: proof that? or is it wrong and I have a bug?
                    expected_rank = expected_rank - 1
                generators = get_generators(
                    1e-12,
                )
                if not generators.gram_schmidt_terminated:
                    norms = generators.gram_schmidt_process.norms
                    average_norm = np.mean(norms)
                    # assert norms[-1] < norms[-2] * 1e-3
                    assert norms[-1] < norms[-2]
                    # assert norms[-1] < average_norm * 1e-3
                    assert norms[-1] < average_norm
                if generators.num_generators != expected_rank:
                    generators = get_generators(
                        1e-24,
                    )
                    if generators.num_generators != expected_rank:
                        with open(
                            f"output/currents/intermediate_{simplicial_mode_choice}"
                            + f"_{num_triangles}_{file_identifier}.json"
                            "w"
                        ) as f:
                            json.dump(
                                {
                                    "num_vertices": num_vertices,
                                    "all_num_vertices": all_num_vertices,
                                },
                                f,
                            )
                        raise ValueError(
                            f"Unexpected number of generators: ",
                            f"{generators.num_generators} (expected {expected_rank})",
                        )
                print("got generators")
                generators.init_eta_currents()
                print("got currents")
                fendley.extend_with_currents(
                    generators.eta_currents,
                    [currents_alpha() for _ in generators.eta_currents],
                )
                print("extended with currents")
                graph = fendley.hamiltonian.get_frustration_graph()
                print("got graph")
                num_vertices.append(graph.num_verts())
                print(f"num_vertices={num_vertices[-1]}")
            all_num_vertices.append(num_vertices)
        with open(data_file, "w") as f:
            json.dump(
                {
                    "all_num_vertices": all_num_vertices,
                },
                f,
            )

    num_axes = len(simplicial_mode_choices)
    fig = plt.figure(figsize=(10, 5 * num_axes))
    gs = fig.add_gridspec(num_axes, 1)
    axes = []
    # fitting_functions = [p2, p3, p4, p5, p6, p7, p8, p9, p10, exp_2, exp]
    fitting_functions = [p7, p8, p9,  exp]
    colormap = plt.get_cmap("plasma")
    colors = [
        colormap(i / len(fitting_functions)) for i in range(len(fitting_functions))
    ]
    x = [i for i in range(low_num_triangles, up_num_triangles + 1)]
    for i, (y, label) in enumerate(zip(all_num_vertices, simplicial_mode_choices)):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)
        ax.plot(x, y, label="data", color="black")
        ax.set_ylabel(f"Number of vertices with {label} simplicial mode")
        ax.set_xticks(x)
        ax.set_yscale("log")

        for i, fn in enumerate(fitting_functions):
            try:
                popt, _ = optimize.curve_fit(fn, x, y)
                ax.plot(
                    x,
                    fn(np.array(x), *popt),
                    label=f"{fn.__name__} fit",
                    linestyle="dashed",
                    color=colors[i],
                )
            except RuntimeError:
                print(f"Could not fit {fn.__name__} for {label} simplicial mode")

    ax = axes[0]
    ax.set_title(file_identifier)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    ax = axes[num_axes - 1]
    ax.set_xlabel("Number of triangles")

    # plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.06, left=0.08, right=0.95)
    plt.savefig(plot_file)


def p2(x, a, b):
    return a * x**2 + b


def p3(x, a, b):
    return a * x**3 + b


def p4(x, a, b):
    return a * x**4 + b


def p5(x, a, b):
    return a * x**5 + b


def p6(x, a, b):
    return a * x**6 + b


def p7(x, a, b):
    return a * x**7 + b


def p8(x, a, b):
    return a * x**8 + b


def p9(x, a, b):
    return a * x**9 + b


def p10(x, a, b):
    return a * x**10 + b


def exp_2(x, a, b):
    return a * np.exp(x ** (1 / 2)) + b


def exp(x, a, b):
    return a * np.exp(x) + b
