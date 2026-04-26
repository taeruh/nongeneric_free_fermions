import os
import json
import numpy as np
from sage.all import Graph
from sage.all import graphs  # pyright: ignore  (this is sage.graphs ...)
from sage.graphs.independent_sets import IndependentSets
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from rust_backend.paulis import Pauli, PauliSum
from hamiltonian import Hamiltonian
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

def currents_plot():

    low_num_triangles = 1
    up_num_triangles = 4

    weight = 1
    # seed = 3
    seed = None
    alpha = ConstantWeight(np.float64(1))
    beta = ConstantWeight(np.float64(1))
    gamma = ConstantWeight(np.float64(1))
    currents_alpha = ConstantWeight(np.float64(1))
    simplicial_mode_choice = "IIIIX"

    load_data = False
    # load_data = True

    os.makedirs("output/currents", exist_ok=True)
    file_identifier = (
        f"{alpha}_{beta}_{gamma}_{currents_alpha}_{simplicial_mode_choice}"
    )
    data_file = f"output/currents/data_{file_identifier}.json"
    plot_file = f"output/currents/plot_{file_identifier}.pdf"

    get_generators = lambda tolerance: Generators(
        simplicial_mode,
        fendley.hamiltonian,
        eta_normalisation_factor=np.float64(
            len(fendley.hamiltonian.operators)
            / fendley.hamiltonian.pauli_l1_norm
            / (np.sqrt(np.sqrt(2.7)))
        ),
        orthogonal_tolerance=tolerance,
        max_search=expected_rank - 1,
    )

    if load_data:
        with open(data_file, "rb") as f:
            data = json.load(f)
            num_vertices = data["num_vertices"]
            num_claws = data["num_claws"]
    else:
        num_vertices = []
        num_claws = []  # up to permutation
        for num_triangles in range(low_num_triangles, up_num_triangles + 1):
            expected_rank = 2 * num_triangles + 1
            print(f"Processing num_triangles={num_triangles}...")
            fendley = Fendley(num_triangles, alpha, beta, gamma)
            simplicial_mode, mode_neighbours = fendley.example_simplicial_modes[
                simplicial_mode_choice
            ]
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
                assert norms[-1] < norms[-2] * 1e-3
                assert norms[-1] < average_norm * 1e-3
            if generators.num_generators != expected_rank:
                generators = get_generators(
                    1e-24,
                )
                if generators.num_generators != expected_rank:
                    with open(
                        "output/currents/intermediate_"
                        + f"{num_triangles}_{file_identifier}.json"
                        "w"
                    ) as f:
                        json.dump(
                            {
                                "num_vertices": num_vertices,
                                "num_claws": num_claws,
                            },
                            f,
                        )
                    raise ValueError(
                        f"Unexpected number of generators: {generators.num_generators} ",
                        f"(expected {expected_rank})",
                    )
            print(f"Number of generators: {generators.num_generators}")
            generators.init_eta_currents()
            fendley.extend_with_currents(
                generators.eta_currents,
                [currents_alpha() for _ in generators.eta_currents],
            )
            print("extended_model")
            graph = fendley.hamiltonian.get_frustration_graph()
            # TODO: do we care about the identity vertex, i.e., include it in the graph and
            # the number of vertices or should I remove it from the hamiltonian?
            num_vertices.append(fendley.hamiltonian.num_ops)
            print(f"Number of vertices in the frustration graph: {num_vertices[-1]}")
            # sagemaths SubgraphSearch does go over all subgraphs in the graph isomorphism
            # class of that subgraph (sadly there is no option to return the single unique
            # up to graph isomorphism subgraph), importantly the graph isomorphism class
            # is in general not the same as the relabeling class, it is smaller, since it
            # preserves the edges; therefore, for each claw we get 6 versions of it, which
            # correspond to the 6 permutations of the 3 outer vertices
            # PERF: this here is the bottleneck; I could do a lot of stuff in krylov.py in
            # Rust, but it doesn't really matter, as this here is going to take way longer
            if num_triangles < 6:
                num_non_unique_num_claws = graph.subgraph_search_count(
                    graphs.ClawGraph(), induced=True
                )
            else:
                # great, if the graph is too big, subgraph_search_count returns a negative
                # number ..., they probably only use a 32 bit integer and then it
                # overflows ...
                num_non_unique_num_claws = 0
                for _ in graph.subgraph_search_iterator(
                    graphs.ClawGraph(), induced=True, return_graphs=False
                ):
                    num_non_unique_num_claws += 1
            num_claws.append(int(num_non_unique_num_claws / 6))
            print(f"Number of claws in the frustration graph: {num_claws[-1]}")
        with open(data_file, "w") as f:
            json.dump(
                {
                    "num_vertices": num_vertices,
                    "num_claws": num_claws,
                },
                f,
            )

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 1)
    axes = []
    x = [i for i in range(low_num_triangles, up_num_triangles + 1)]
    for i, (y, label) in enumerate(
        zip([num_vertices, num_claws], ["Number of vertices", "Number of claws"])
    ):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)
        ax.plot(x, y)
        ax.set_ylabel(label)
        ax.set_xticks(x)
    ax = axes[0]
    ax.set_title(file_identifier)
    ax = axes[1]
    ax.set_xlabel("Number of triangles")
    ax.set_yscale("log")
    # plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.06, left=0.08, right=0.95)
    plt.savefig(plot_file)
