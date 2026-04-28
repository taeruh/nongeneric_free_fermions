import os
import json
import numpy as np
from scipy import optimize
from sage.all import graphs  # pyright: ignore  (this is sage.graphs ...)
import matplotlib.pyplot as plt

from models.fendley import Fendley
from models.integer_fendley import IntegerFendley
from models.weights import ConstantWeight, RandomWeight
from krylov import Generators
from integer_krylov import IntegerGenerators
from paulis import Pauli

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
    fitting_functions = [p7, p8, p9, exp]
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


def _run():
    alpha = 1
    beta = 1
    gamma = 1
    current_alpha = 1

    num_triangles = 7
    mode = "IIZZZ"
    # mode = "IZZZZ"
    # mode = "ZZZZZ"

    int_fendley = IntegerFendley(num_triangles, alpha, beta, gamma)
    int_simplicial_mode = int_fendley.example_simplicial_modes[mode][0]
    int_generators = IntegerGenerators(
        (1, int_simplicial_mode), int_fendley.hamiltonian
    )
    int_generators.init_eta_currents()
    int_fendley.extend_with_currents(
        int_generators.eta_currents,
        [current_alpha for _ in int_generators.eta_currents],
    )

    # print(len(generators.etas))
    for eta in int_generators.etas:
        # print(len(eta.ops))
        # print([(w, op.to_string()) for w, op in eta.ops])
        for i, op1 in enumerate(eta.ops):
            for op2 in eta.ops[i + 1 :]:
                assert not op1[1].is_proportional_to(
                    op2[1]
                ), "Found proportional operators!"

    fendley = Fendley(
        num_triangles,
        ConstantWeight(np.float64(alpha)),
        ConstantWeight(np.float64(beta)),
        ConstantWeight(np.float64(gamma)),
        # ConstantWeight(np.float64(np.sqrt(2))),
        # ConstantWeight(np.float64(np.pi / 4)),
        # ConstantWeight(np.float64(1.0)),
    )
    simplicial_mode = fendley.example_simplicial_modes[mode][0]
    eta_normalisation_factor = np.float64(
        len(fendley.hamiltonian.operators)
        / fendley.hamiltonian.pauli_l1_norm
        / (np.sqrt(np.sqrt(2.7)))
    )
    # eta_normalisation_factor = 1.0
    generators = Generators(
        (1, simplicial_mode),
        fendley.hamiltonian,
        orthogonal_tolerance=1e-9,
        renormalise=True,
        eta_normalisation_factor=eta_normalisation_factor,
    )
    generators.init_eta_currents()
    fendley.extend_with_currents(
        generators.eta_currents,
        [
            np.float64(1.0 / eta_normalisation_factor**l)
            for l in range(len(generators.eta_currents))
        ],
    )

    # int_eta = int_generators.etas[8]

    # # print(len(generators.etas))
    # for eta in generators.etas:
    #     #     print(eta.len())
    #     #     print([(w, op.to_string()) for w, op in eta.to_py_list()])
    #     for i, op1 in enumerate(eta.to_py_list()):
    #         for op2 in eta.to_py_list()[i + 1 :]:
    #             assert not op1[1].is_proportional_to(
    #                 op2[1]
    #             ), "Found proportional operators!"

    # eta = generators.etas[6]

    # # η_6 = 12 χ - 18 a b χ - 25 a c χ - 7 a b d e χ - 8 a b d f χ - 8 a c d f χ - 8 a c e f χ - 9 a c e g χ - a b d e g h χ - a b d e g i χ - a b d f g i χ - a b d f h i χ - a b d f h j χ - a c d f g i χ - a c d f h i χ - a c d f h j χ - a c e f h i χ - a c e f h j χ - a c e g h j χ - a c e g i j χ

    # letter_to_vert_index = {
    #     "a": 0,
    #     "b": 1,
    #     "c": 2,
    #     "d": 3,
    #     "e": 4,
    #     "f": 5,
    #     "g": 6,
    #     "h": 7,
    #     "i": 8,
    #     "j": 9,
    # }
    # alt_ops = []
    # for factor, word in [
    #     (12.0, ""),
    #     (18.0, "a b"),
    #     (25.0, "a c"),
    #     (7.0, "a b d e"),
    #     (8.0, "a b d f"),
    #     (8.0, "a c d f"),
    #     (8.0, "a c e f"),
    #     (9.0, "a c e g"),
    #     (1.0, "a b d e g h"),
    #     (1.0, "a b d e g i"),
    #     (1.0, "a b d f g i"),
    #     (1.0, "a b d f h i"),
    #     (1.0, "a b d f h j"),
    #     (1.0, "a c d f g i"),
    #     (1.0, "a c d f h i"),
    #     (1.0, "a c d f h j"),
    #     (1.0, "a c e f h i"),
    #     (1.0, "a c e f h j"),
    #     (1.0, "a c e g h j"),
    #     (1.0, "a c e g i j"),
    # ]:
    #     op = Pauli.identity(fendley.n)
    #     for letter in word.split():
    #         vert_index = letter_to_vert_index[letter]
    #         op = op.multiply_as_paulis(fendley.hamiltonian.operators[vert_index])
    #     op = op.multiply_as_paulis(simplicial_mode)
    #     alt_ops.append((factor, op))

    # # for op in eta.to_py_list():
    # for op in int_eta.ops:
    #     # for op in alt_ops:
    #     print(op[0], op[1].to_string())

    # print(len(int_eta.ops))
    # # print(len(int_generators.etas))

    # print(len(int_generators.etas))

    # for op in int_fendley.hamiltonian.operators:
    #     print(op.to_string())

    # op = Pauli.identity(fendley.n)
    # for letter in "a c e g i k".split():
    #     vert_index = letter_to_vert_index[letter]
    #     op = op.multiply_as_paulis(fendley.hamiltonian.operators[vert_index])

    print(int_fendley.hamiltonian.get_frustration_graph().num_verts())
    print(fendley.hamiltonian.get_frustration_graph().num_verts())

    # print(int_fendley.hamiltonian.operators[0].to_string())
    operators = dict()
    for op, weight in zip(
        int_fendley.hamiltonian.operators, int_fendley.hamiltonian.weights
    ):
        label, sign = op.to_string().split(", ")
        weight *= int(sign)
        operators[label] = weight
        if abs(weight) < 1e-6:
            print(label, weight)

    for op, weight in zip(fendley.hamiltonian.operators, fendley.hamiltonian.weights):
        label, sign = op.to_string().split(", ")
        sign_weight = weight * int(sign)
        int_weight = operators.get(label, None)
        if int_weight is None:
            # raise ValueError(f"Could not find operator {label} in integer hamiltonian")
            print(label, sign_weight)
        else:
            # if sign_weight != int_weight:
            if not np.isclose(sign_weight, int_weight, atol=1e-6):
                print(sign, int(sign), weight)
                print(f"{label}: {sign_weight} , {int_weight}")
