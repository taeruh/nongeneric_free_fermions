import json
import numpy as np
from sage.all import graphs  # pyright: ignore  (this is sage.graphs ...)
import matplotlib.pyplot as plt

from models.fendley import Fendley
from models.weights import ConstantWeight
from krylov import Generators
from . import utils


def run():
    low_num_triangles = 1
    up_num_triangles = 6

    alpha = ConstantWeight(np.float64(1))
    beta = ConstantWeight(np.float64(1))
    gamma = ConstantWeight(np.float64(1))
    currents_alpha = ConstantWeight(np.float64(1))
    simplicial_mode_choices = ["IIZZZ", "IZZZZ", "ZZZZZ"]

    # do_calculation = True
    do_calculation = False
    do_plot = True
    # do_plot = False

    file_identifier = (
        f"num_claws_{alpha}_{beta}_{gamma}_{currents_alpha}_{low_num_triangles}_{up_num_triangles}_"
        + "-".join(simplicial_mode_choices)
    )
    data_file = f"output/final/data/{file_identifier}.json"

    if do_calculation:
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
                max_search_eta_index=expected_rank - 1,
            )

            num_claws = []
            for num_triangles in range(low_num_triangles, up_num_triangles + 1):
                expected_rank = 2 * num_triangles + 1
                print(f"Processing num_triangles={num_triangles}...")
                fendley = Fendley(num_triangles, alpha, beta, gamma)
                simplicial_mode, mode_neighbours = fendley.example_simplicial_modes[
                    simplicial_mode_choice
                ]
                simplicial_mode = (1.0, simplicial_mode)
                if mode_neighbours == 3:
                    expected_rank = expected_rank - 1
                generators = get_generators(
                    1e-12,
                )
                assert generators.num_generators == expected_rank
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
    else:
        with open(data_file, "rb") as f:
            data = json.load(f)
            all_num_claws = data["all_num_claws"]

    if do_plot:
        utils.paper_setup()
        fig = plt.figure(figsize=utils.set_size(height_in_width=0.5))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        x = [i for i in range(low_num_triangles, up_num_triangles + 1)]
        labels = [
            r"Z_1 \cdots Z_3",
            r"Z_1 \cdots Z_4",
            r"Z_1 \cdots Z_5",
        ]
        for y, label in zip(
            all_num_claws,
            labels,
        ):
            ax.plot(x, y, label=rf"$\chi = {label}$")
        ax.set_ylabel(r"Number of claws $N_C$")
        ax.set_xticks(x)
        ax.set_yscale("log")
        ax.set_xlabel(r"Independence number $\alpha(G)$")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper left")
        plt.subplots_adjust(top=0.97, bottom=0.20, left=0.13, right=0.97)
        plt.savefig("output/final/number_of_claws.pdf")
