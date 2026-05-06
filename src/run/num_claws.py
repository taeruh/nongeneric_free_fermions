import os
import json
import numpy as np
from sage.all import graphs  # pyright: ignore  (this is sage.graphs ...)
import matplotlib.pyplot as plt

from models.fendley import Fendley
from models.weights import ConstantWeight, RandomWeight
from krylov import Generators


def run():
    low_num_triangles = 1
    up_num_triangles = 5

    # weight = 1
    # seed = 3
    # seed = None
    alpha = ConstantWeight(np.float64(1))
    beta = ConstantWeight(np.float64(1))
    gamma = ConstantWeight(np.float64(1))
    currents_alpha = ConstantWeight(np.float64(1))
    simplicial_mode_choices = ["IIZZZ", "IZZZZ", "ZZZZZ"]

    load_data = False
    # load_data = True

    os.makedirs("output/currents", exist_ok=True)
    file_identifier = (
        f"num_claws_{alpha}_{beta}_{gamma}_{currents_alpha}"
        + "-".join(simplicial_mode_choices)
    )
    data_file = f"output/currents/data_{file_identifier}.json"
    plot_file = f"output/currents/plot_{file_identifier}.pdf"

    if load_data:
        with open(data_file, "rb") as f:
            data = json.load(f)
            all_num_claws = data["all_num_claws"]
    else:
        all_num_claws = []
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
                max_search_eta_index=expected_rank-1,
            )

            num_claws = []  # up to permutation
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
                    assert norms[-1] < norms[-2] * 1e-3
                    assert norms[-1] < average_norm * 1e-3
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
                                    "num_claws": num_claws,
                                    "all_num_claws": all_num_claws,
                                },
                                f,
                            )
                        raise ValueError(
                            f"Unexpected number of generators: ",
                            f"{generators.num_generators} (expected {expected_rank})",
                        )
                generators.init_eta_currents()
                fendley.extend_with_currents(
                    generators.eta_currents,
                    [currents_alpha() for _ in generators.eta_currents],
                )
                graph = fendley.hamiltonian.get_frustration_graph()
                # sagemaths SubgraphSearch does go over all subgraphs in the graph
                # isomorphism class of that subgraph (sadly there is no option to return
                # the single unique up to graph isomorphism subgraph), importantly the
                # graph isomorphism class is in general not the same as the relabeling
                # class, it is smaller, since it preserves the edges; therefore, for each
                # claw we get 6 versions of it, which correspond to the 6 permutations of
                # the 3 outer vertices
                # PERF: this here is the bottleneck; I could do a lot of stuff in
                # krylov.py in Rust, but it doesn't really matter, as this here is going
                # to take way longer
                if num_triangles < 6:
                    num_non_unique_num_claws = graph.subgraph_search_count(
                        graphs.ClawGraph(), induced=True
                    )
                else:
                    # great, if the graph is too big, subgraph_search_count returns a
                    # negative number ..., they probably only use a 32 bit integer and
                    # then it overflows ...
                    num_non_unique_num_claws = 0
                    for _ in graph.subgraph_search_iterator(
                        graphs.ClawGraph(), induced=True, return_graphs=False
                    ):
                        num_non_unique_num_claws += 1
                num_claws.append(int(num_non_unique_num_claws / 6))
            all_num_claws.append(num_claws)
        with open(data_file, "w") as f:
            json.dump(
                {
                    "all_num_claws": all_num_claws,
                },
                f,
            )

    num_axes = len(simplicial_mode_choices)
    fig = plt.figure(figsize=(10, 5 * num_axes))
    gs = fig.add_gridspec(num_axes, 1)
    axes = []
    x = [i for i in range(low_num_triangles, up_num_triangles + 1)]
    for i, (y, label) in enumerate(zip(all_num_claws, simplicial_mode_choices)):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)
        ax.plot(x, y)
        ax.set_ylabel(f"Number of claws with {label} simplicial mode")
        ax.set_xticks(x)
        ax.set_yscale("log")
    ax = axes[0]
    ax.set_title(file_identifier)
    ax = axes[num_axes - 1]
    ax.set_xlabel("Number of triangles")
    # plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.06, left=0.08, right=0.95)
    plt.savefig(plot_file)
