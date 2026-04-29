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
from models.fukai import Fukai
from models.weights import ConstantWeight, RandomWeight
from krylov import Generators
import house_of_graphs
import phase_diagram


def calc_gap(num_triangles, alpha, beta, gamma, extend: bool) -> float:
    print(alpha, beta, gamma)
    fendley = Fendley(
        num_triangles,
        ConstantWeight(alpha),
        ConstantWeight(beta),
        ConstantWeight(gamma),
    )
    expected_rank = 2 * num_triangles  # no +1 because we choose the 3-connect

    # simplicial_mode = fendley.example_simplicial_modes["IIIIX"][0]
    # choose a mode that is connected to all three vertices, so that it is (more)
    # symmetric (e.g., otherwise if alpha is 0 and we only connect to the first
    # vertex, this vertex doesn't actually exist...) -> can use triangle_grid_inner to
    # avoid any zero values
    simplicial_mode = fendley.example_simplicial_modes["IIYII"][0]

    eta_normalisation_factor_makes_sense = np.float64(
        len(fendley.hamiltonian.operators)
        / fendley.hamiltonian.pauli_l1_norm
        # 1.0
    )
    # this _extra factor improves the stability of the gram schmidt process, currently
    # we factor it out again afterwards when we extend the model with the currents (so
    # that the normalisation factor looks nicer)
    # TODO: check that we actually do that (because I might decide against doing that
    # to also keep the currents more normalised)
    eta_normalisation_factor_extra = np.float64(1 / (np.sqrt(np.sqrt(2.7))))
    # eta_normalisation_factor_extra = np.float64(1.0)
    eta_normalisation_factor = (
        eta_normalisation_factor_makes_sense * eta_normalisation_factor_extra
    )

    generators = Generators(
        (1.0, simplicial_mode),
        fendley.hamiltonian,
        renormalise=True,
        eta_normalisation_factor=eta_normalisation_factor,
        max_search_eta_index=expected_rank - 1,
    )

    generators.init_gammas(False, False)
    generators.init_gamma_bilinears()

    total_coeffs = dict()
    for w, op in zip(fendley.hamiltonian.weights, fendley.hamiltonian.operators):
        coeff = generators.bilinear_gamma_projection(op)
        for j, k, c in coeff:
            assert c.imag == 0
            total_coeffs[(j, k)] = total_coeffs.get((j, k), 0) + c.real * w

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
    # print(h)

    lm, _ = phase_diagram.skew_diagonalise(h)
    phase_diagram.smoothen_lamda(lm, eps=1e-10)
    # print(lm)
    lm_pairs = phase_diagram.get_lamda_pairs(lm)
    # print(lm_pairs)
    all_values = phase_diagram.get_lamda_eigenvalues(lm_pairs)
    min_value = all_values[0]
    gap = phase_diagram.get_gap(all_values)

    # print(lm_pairs)
    # print(all_values)
    print(min_value)
    print(gap)

    return gap


# for the parallelisation we need a function that is not defined in another function
# (otherwise there is some picklelisation error...)
def calc_gap_wrapper(args):
    num_triangles, alpha, beta, gamma = args
    return calc_gap(num_triangles, alpha, beta, gamma, True)


def get_phase_diagram():
    from multiprocessing import Pool

    low_num_triangles = 5
    up_num_triangles = 6
    num_samples = 15
    factor = 3
    points = phase_diagram.triangle_grid(num_samples, factor)

    do_calculation = True
    # do_calculation = False

    if do_calculation == True:
        all_values = []
        for num_triangles in range(low_num_triangles, up_num_triangles + 1):
            tpoints = []
            for alpha, beta, gamma in points:
                tpoints.append((num_triangles, alpha, beta, gamma))
            # points = phase_diagram.triangle_grid_inner(8, factor)
            values = [calc_gap_wrapper(point) for point in tpoints]
            # TODO: there is quite some multiprocessing overhead...; it probably would be
            # better if I parallelise externally with multiple jobs...
            # with Pool() as pool:
            #     values = pool.map(calc_gap_wrapper, tpoints)
            all_values.append(values)
        with open("output/raw_phase_diagram_data.json", "w") as f:
            json.dump(all_values, f)
    else:
        with open("output/raw_phase_diagram_data.json", "r") as f:
            all_values = json.load(f)

    data = {
        "num_triangles_range": [low_num_triangles, up_num_triangles],
        "num_samples": num_samples,
        "factor": factor,
        "all_values": all_values,
    }
    with open("output/phase_diagram_data.json", "w") as f:
        json.dump(data, f)

    for num_triangles, values in zip(
        range(low_num_triangles, up_num_triangles + 1), all_values
    ):
        # values = [np.log10(value) for value in values]
        x, y = phase_diagram.points_to_plot_coordinates(points)
        triang = tri.Triangulation(x, y)
        _, ax = plt.subplots(figsize=(7, 6))
        tpc = ax.tripcolor(triang, values, shading="gouraud")
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
        place_label(
            ax, factor / 2, 0, factor / 2, r"$\beta=0$", rotation=60, ha="right"
        )
        place_label(
            ax, 0, factor / 2, factor / 2, r"$\alpha=0$", rotation=-60, ha="left"
        )
        cbar = plt.colorbar(tpc, ax=ax)
        cbar.set_label("Function value")
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"output/phase_diagram_{num_triangles}.pdf")


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
        charges.append(PauliSum([]))
    for independent_set in independent_sets:
        length = len(independent_set)
        product = Pauli.identity(fendley.hamiltonian.n)
        weight = 1
        for vertex in independent_set:
            op = fendley.hamiltonian.operators[vertex]
            w = fendley.hamiltonian.weights[vertex]
            product = product.multiply_as_paulis(op)
            weight *= w
        charges[length].single_add(weight, product)

    # for charge in charges:
    #     for w, op in charge:
    #         print(f"{w:.2f}, {op.to_string()}")
    #     print()

    for i in range(len(charges)):
        for j in range(i + 1, len(charges)):
            commutator = (charges[i].multiply(charges[j])).subtract(
                charges[j].multiply(charges[i])
            )
            print(f"Commutator of charges {i} and {j}:")
            print([f"{w:.2f}, {op.to_string()}" for w, op in commutator.to_py_list()])

    t_plus_u = []
    t_minus_u = []
    for k in range(len(charges)):
        charge = charges[k]
        t_minus_u.append(charge.deep_copy())
        # sign = (-1) ** k
        # charge = [(w * sign, op) for w, op in charge]
        if k % 2 == 1:
            charge.multiply_with_float(-1.0)
        t_plus_u.append(charge)

    poly = dict()
    for i, charge_i in enumerate(t_plus_u):
        for j, charge_j in enumerate(t_minus_u):
            degree = i + j
            prod = charge_i.multiply(charge_j)
            if degree not in poly:
                poly[degree] = prod
            else:
                poly[degree] = poly[degree].add(prod)

    poly = sorted(poly.items())
    for degree, terms in poly:
        print(f"Degree {degree}:")
        for w, op in terms.to_py_list():
            print(f"{w:.2f}, {op.to_string()}")
        print()


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
        ops = PauliSum(
            [
                (
                    fendley_extended.hamiltonian.weights[i],
                    fendley_extended.hamiltonian.operators[i],
                )
                for i in claw
            ]
        )
        claws_ops.append(ops)

    print(len(claws_ops), "claws found in the frustration graph")

    while len(claws_ops) > 0:
        claw = claws_ops.pop()
        for other in claws_ops:
            prod = claw.multiply(other)
            if prod.len() == 4:
                print([(f"{w:.2f}, {op.to_string()}") for w, op in prod.to_py_list()])
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

    generators.init_gammas()
    generators.init_gamma_bilinears()

    reconstructed = PauliSum([])
    # for fukai_op in fukai.fukai_ops:
    for fukai_op in fukai.fukai_ops[:1]:
        projection = generators.bilinear_gamma_projection(fukai_op)

        inner_product = 0

        for i, j, coeff in projection:
            inner_product += coeff**2
            part = generators.gamma_bilinears[(i, j)].deep_copy()
            part.multiply_with_float(coeff)
            reconstructed = reconstructed.add(part)

        print(inner_product)
        print()

    print(fukai.fukai_ops[0].to_string())

    print(
        [f"{coeff:.4f}, {op.to_string()}" for coeff, op in reconstructed.to_py_list()]
    )
