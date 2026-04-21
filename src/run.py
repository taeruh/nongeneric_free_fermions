import os
import json
import numpy as np
from sage.all import Graph
from sage.all import graphs  # pyright: ignore  (this is sage.graphs ...)
from sage.graphs.independent_sets import IndependentSets
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import paulis
from paulis import Pauli
# from rust_backend.paulis import Pauli, PauliSum
from hamiltonian import Hamiltonian
from models.fendley import Fendley
from models.fukai import Fukai
from models.weights import ConstantWeight, RandomWeight
from krylov import Generators
import house_of_graphs
import phase_diagram

# observations:
# - while the coefficients in the etas and eta_currents obviously depend on alpha,
#   beta, gamma and currents_alpha, the operators themselves seem to be independent
#   of those parameters
# - changing the simplicial mode changes the operators of course, however, the graph
#   and the coefficients seem to stay the same between different modes from the same
#   connectivity to the graph (i.e., connect to one vertex, connect to two vertices,
#   connect to three vertices)


def get_phase_diagram():
    def calc_gap(alpha, beta, gamma, extend: bool) -> float:
        print(alpha, beta, gamma)
        fendley = Fendley(
            2, ConstantWeight(alpha), ConstantWeight(beta), ConstantWeight(gamma)
        )

        # simplicial_mode = fendley.example_simplicial_modes["IIIIX"][0]
        # choose a mode that is connected to all three vertices, so that it is (more)
        # symmetric (e.g., otherwise if alpha is 0 and we only connect to the first
        # vertex, this vertex doesn't actually exist...) -> can use triangle_grid_inner to
        # avoid any zero values
        simplicial_mode = fendley.example_simplicial_modes["IIYII"][0]

        eta_normalisation_factor_makes_sense = np.float64(
            len(fendley.hamiltonian.operators) / fendley.hamiltonian.pauli_l1_norm
        )
        # this _extra factor improves the stability of the gram schmidt process, currently
        # we factor it out again afterwards when we extend the model with the currents (so
        # that the normalisation factor looks nicer)
        # TODO: check that we actually do that (because I might decide against doing that
        # to also keep the currents more normalised)
        eta_normalisation_factor_extra = np.float64(1 / (np.sqrt(np.sqrt(2.7))))
        eta_normalisation_factor = (
            eta_normalisation_factor_makes_sense * eta_normalisation_factor_extra
        )

        generators = Generators(
            simplicial_mode,
            fendley.hamiltonian,
            eta_normalisation_factor=eta_normalisation_factor,
        )

        generators.init_gammas(False)
        # generators.init_gammas()
        print("got gammas")
        generators.init_gamma_bilinears()
        print("got gamma bilinears")

        total_coeffs = dict()
        for w, op in zip(fendley.hamiltonian.weights, fendley.hamiltonian.operators):
            coeff = generators.bilinear_gamma_projection(op)
            for j, k, c in coeff:
                assert c.imag == 0
                total_coeffs[(j, k)] = total_coeffs.get((j, k), 0) + c.real * w
        print("got hamiltonian coeffs")

        if extend:
            currents_coeffs = generators.eta_currents_bilinear_gamma_projection()

            # get rid of the _extra factor, so that the normalisation is effectively just
            # _makes_sense
            alphas = [
                1 / (eta_normalisation_factor_extra**l)
                for l in range(generators.num_generators)
                # 1.  for _ in range(generators.num_generators)
            ]

            # don't actually need to do that since we only care about the coefficients
            # generators.init_eta_currents()
            # fendley.extend_with_currents(generators.eta_currents, alphas)

            for alpha, coeffs in zip(alphas, currents_coeffs):
                for i, j, c in coeffs:
                    total_coeffs[(i, j)] = total_coeffs.get((i, j), 0) + alpha * c

        for c in total_coeffs.values():
            assert c.imag == 0

        h = np.zeros((generators.num_generators, generators.num_generators))
        for (i, j), coeff in total_coeffs.items():
            h[i, j] = coeff / 2
            h[j, i] = -coeff / 2
        print("got h")

        lm, _ = phase_diagram.skew_diagonalise(h)
        print("got skew diagonalised h")
        phase_diagram.smoothen_lamda(lm)
        lm_pairs = phase_diagram.get_lamda_pairs(lm)
        print("got lamda pairs")
        all_values = phase_diagram.get_lamda_eigenvalues(lm_pairs)
        print("got lamda eigenvalues")
        min_value = all_values[0]
        gap = phase_diagram.get_gap(all_values)

        # print(all_values)
        print(min_value)
        print(gap)

        return gap

    factor = 3
    points = phase_diagram.triangle_grid(15, factor)
    # points = phase_diagram.triangle_grid_inner(10, factor)
    values = [calc_gap(alpha, beta, gamma, True) for alpha, beta, gamma in points]

    x, y = phase_diagram.points_to_plot_coordinates(points)

    triang = tri.Triangulation(x, y)

    fig, ax = plt.subplots(figsize=(7, 6))

    tpc = ax.tripcolor(triang, values, shading="gouraud")

    # Triangle outline
    vertices_bary = np.array(
        [
            [factor, 0, 0],
            [0, factor, 0],
            [0, 0, factor],
            [factor, 0, 0],  # close the loop
        ]
    )
    vx, vy = phase_diagram.points_to_plot_coordinates(vertices_bary)
    ax.plot(vx, vy, lw=1)

    def place_label(ax, alpha, beta, gamma, text, **kwargs):
        x, y = phase_diagram.points_to_plot_coordinates(
            np.array([[alpha, beta, gamma]])
        )
        ax.text(x[0], y[0], text, **kwargs)

    place_label(ax, factor, 0, 0, rf"$\alpha={factor}$", ha="right", va="top")
    place_label(ax, 0, factor, 0, rf"$\beta={factor}$", ha="left", va="top")
    place_label(ax, 0, 0, factor, rf"$\gamma={factor}$", ha="center", va="bottom")

    place_label(ax, factor / 2, factor / 2, 0, r"$\gamma=0$", ha="center", va="top")
    place_label(ax, factor / 2, 0, factor / 2, r"$\beta=0$", rotation=60, ha="right")
    place_label(ax, 0, factor / 2, factor / 2, r"$\alpha=0$", rotation=-60, ha="left")

    cbar = plt.colorbar(tpc, ax=ax)
    cbar.set_label("Function value")

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("output/phase_diagram.pdf")


def test_t():
    fendley = Fendley(2, ConstantWeight(1), ConstantWeight(1), ConstantWeight(1))

    simplicial_mode = fendley.example_simplicial_modes["IIYII"][0]
    generators = Generators(simplicial_mode, fendley.hamiltonian)
    generators.init_eta_currents()
    fendley.extend_with_currents(
        generators.eta_currents,
        [np.float64(1.0) for _ in generators.eta_currents],
    )

    graph = fendley.hamiltonian.get_frustration_graph()
    independent_sets = IndependentSets(graph)
    alpha = 0
    for independent_set in independent_sets:
        length = len(independent_set)
        if length > alpha:
            alpha = length
    charges = []
    for _ in range(alpha + 1):
        charges.append([])
    for independent_set in independent_sets:
        length = len(independent_set)
        product = Pauli.identity(fendley.hamiltonian.n)
        weight = 1
        for vertex in independent_set:
            op = fendley.hamiltonian.operators[vertex]
            w = fendley.hamiltonian.weights[vertex]
            product = product.multiply_as_paulis(op)
            weight *= w
        already_in = False
        for i, (w, op) in enumerate(charges[length]):
            if product.is_proportional_to(op):
                phase = product.phase_difference(op)
                charges[length][i] = (w + weight * (1j) ** phase, op)
                already_in = True
                break
        if not already_in:
            charges[length].append((weight, product))

    # for charge in charges:
    #     for w, op in charge:
    #         print(f"{w:.2f}, {op.to_string()}")
    #     print()

    for i in range(len(charges)):
        for j in range(i + 1, len(charges)):
            left = paulis.list_multiplication(charges[i], charges[j])
            right = paulis.list_multiplication(charges[j], charges[i])
            right = [(w * -1, op) for w, op in right]
            commutator = paulis.list_addition(left, right)
            print(f"Commutator of charges {i} and {j}:")
            print([f"{w:.2f}, {op.to_string()}" for w, op in commutator])

    t_plus_u = []
    t_minus_u = []
    for k in range(len(charges)):
        charge = charges[k]
        t_minus_u.append(charge)
        sign = (-1) ** k
        charge = [(w * sign, op) for w, op in charge]
        t_plus_u.append(charge)

    poly = dict()
    for i, charge_i in enumerate(t_plus_u):
        for j, charge_j in enumerate(t_minus_u):
            degree = i + j
            prod = paulis.list_multiplication(charge_i, charge_j)
            if degree not in poly:
                poly[degree] = prod
            else:
                poly[degree] = paulis.list_addition(poly[degree], prod)

    poly = sorted(poly.items())
    for degree, terms in poly:
        print(f"Degree {degree}:")
        for w, op in terms:
            print(f"{w:.2f}, {op.to_string()}")
        print()


def currents_plot():

    low_num_triangles = 1
    up_num_triangles = 6

    weight = 1
    # seed = 3
    seed = None
    alpha = ConstantWeight(np.float64(1))
    beta = ConstantWeight(np.float64(1))
    gamma = ConstantWeight(np.float64(1))
    currents_alpha = ConstantWeight(np.float64(1))
    # alpha = RandomWeight(-np.float64(1), np.float64(1), seed=seed)
    # beta = RandomWeight(-np.float64(1), np.float64(1), seed=seed)
    # gamma = RandomWeight(-np.float64(1), np.float64(1), seed=seed)
    # currents_alpha = RandomWeight(-np.float64(1), np.float64(1), seed=seed)
    simplicial_mode_choice = "IIIIX"
    # simplicial_mode_choice = "ZZZZZ"
    # simplicial_mode_choice = "IZZZZ"

    # load_data = False
    load_data = True

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
            print("got currents")
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


def run():
    num_triangles = 2
    fendley = Fendley(num_triangles)
    simplicial_mode = fendley.example_simplicial_modes["IIIIX"][0]
    generators = Generators(simplicial_mode, fendley.hamiltonian)
    generators.init_eta_currents()

    fendley_extended = fendley.clone()
    fendley_extended.extend_with_currents(
        generators.eta_currents,
        current_alpha=[np.float64(1.0)] * len(generators.eta_currents),
    )
    graph = fendley_extended.hamiltonian.get_frustration_graph()
    labeled_graph: Graph = graph.relabel(
        lambda x: f"({fendley_extended.labels[x]}, {fendley_extended.weights[x]:.2f})",
        inplace=False,
    )
    labeled_graph.plot().save_image(  # pyright: ignore
        "output/fendley_extended_graph.png"
    )

    house_of_graphs.save_adj_matrix(graph)

    return

    claws = set()
    # for claw in labeled_graph.subgraph_search_iterator(
    for claw in graph.subgraph_search_iterator(
        graphs.ClawGraph(), induced=True, return_graphs=False
    ):
        claw_unique = tuple(sorted(claw))
        if claw_unique in claws:
            continue
        claws.add(claw_unique)

    claws_ops = []
    for claw in claws:
        ops = [
            (
                fendley_extended.hamiltonian.weights[i],
                fendley_extended.hamiltonian.operators[i],
            )
            for i in claw
        ]
        claws_ops.append(ops)

    print(len(claws_ops), "claws found in the frustration graph")

    while len(claws_ops) > 0:
        claw = claws_ops.pop()
        for other in claws_ops:
            prod = paulis.list_multiplication(claw, other)
            # print(len(prod))
            if len(prod) == 4:
                print([(f"{w:.2f}, {op.to_string()}") for w, op in prod])
                weights = []
                ops = []
                for w, op in prod:
                    weights.append(w)
                    ops.append(op)
                hamiltonian = Hamiltonian(weights, ops)
                graph = hamiltonian.get_frustration_graph()
                graph.plot().save_image(  # pyright: ignore
                    "output/claw_product_graph.png"
                )
                break
        break


def trying_to_reconstruct_fukai_from_bilinears():
    num_triangles = 3
    fendley = Fendley(num_triangles)
    fukai = Fukai(
        num_triangles,
        beta_3=ConstantWeight(np.float64(1)),
        beta_5=ConstantWeight(np.float64(1)),
    )

    fukai_graph = fukai.hamiltonian.get_frustration_graph()
    fukai_labeled_graph: Graph = fukai_graph.relabel(
        lambda x: f"({fukai.labels[x]}, {fukai.hamiltonian.weights[x]:.2f})",
        inplace=False,
    )
    fukai_labeled_graph.plot().save_image("output/fukai_graph.png")  # pyright: ignore
    for op in fukai.hamiltonian.operators:
        print(op.to_string())

    simplicial_mode = fendley.example_simplicial_modes["IIIIX"][0]
    generators = Generators(simplicial_mode, fendley.hamiltonian)
    fendley_with_simplicial_mode = fendley.hamiltonian
    fendley_with_simplicial_mode.operators.append(simplicial_mode)
    fendley_with_simplicial_mode.weights.append(np.float64(1.0))
    fendley_with_simplicial_mode.num_ops += 1
    fendley_graph = fendley_with_simplicial_mode.get_frustration_graph()
    labels = fendley.labels + ["simplicial_mode"]
    fendley_graph.relabel(
        lambda x: f"({labels[x]})",
        inplace=True,
    )
    fendley_graph.plot().save_image("output/fendley_graph.png")  # pyright: ignore

    print(f"Number of generators: {generators.num_generators}")

    print([(w, op.to_string()) for op, w in zip(fukai.fukai_ops, fukai.fukai_weights)])

    generators.init_gamma_bilinears()

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

    print([f"{coeff:.4f}, {op.to_string()}" for coeff, op in reconstructed])
