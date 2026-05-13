import json
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

from models.fendley import Fendley
from models.integer_fendley import IntegerFendley
from models.weights import ConstantWeight
from krylov import Generators
from integer_krylov import IntegerGenerators
from rust_backend.krylov_without_gram_schmidt import GeneratorsWithoutGramSchmidt
from rust_backend.fendley import Fendley as RustFendley


def run():
    low_num_triangles = 1
    up_num_triangles = 8
    # up_num_triangles = 15

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
        f"num_vertices_{alpha}_{beta}_{gamma}_{currents_alpha}"
        + "-".join(simplicial_mode_choices)
        + f"_{low_num_triangles}_{up_num_triangles}"
    )
    data_file = f"output/final/data/{file_identifier}.json"

    if do_calculation:

        all_num_vertices = []
        for simplicial_mode_choice in simplicial_mode_choices:
            num_vertices = []
            for num_triangles in range(low_num_triangles, up_num_triangles + 1):
                expected_rank = 2 * num_triangles + 1
                print(f"Processing num_triangles={num_triangles}...")
                fendley = Fendley(num_triangles, alpha, beta, gamma)
                rust_fendley = RustFendley(num_triangles, alpha(), beta(), gamma())
                integer_fendley = IntegerFendley(
                    num_triangles, int(alpha()), int(beta()), int(gamma())
                )
                simplicial_mode, mode_neighbours = fendley.example_simplicial_modes[
                    simplicial_mode_choice
                ]
                int_simplicial_mode, _ = integer_fendley.example_simplicial_modes[
                    simplicial_mode_choice
                ]
                int_simplicial_mode = (1, int_simplicial_mode)
                simplicial_mode = (1.0, simplicial_mode)
                if mode_neighbours == 3:
                    expected_rank = expected_rank - 1

                generators_wgs = GeneratorsWithoutGramSchmidt(
                    simplicial_mode,
                    rust_fendley,
                    expected_rank - 1,
                    renormalise=True,
                    eta_normalisation_factor=np.float64(
                        len(fendley.hamiltonian.operators)  # pyright: ignore
                        / fendley.hamiltonian.pauli_l1_norm  # pyright: ignore
                        / (np.sqrt(np.sqrt(2.7)))
                    ),
                )
                generators_wgs.init_eta_currents()

                rust_fendley.extend_with_currents(
                    generators_wgs,
                    [
                        currents_alpha()
                        for _ in range(generators_wgs.num_eta_currents())
                    ],
                )
                num_verts_wgs = rust_fendley.num_operators()

                if num_triangles < 8:
                    generators = Generators(
                        simplicial_mode,
                        fendley.hamiltonian,  # pyright: ignore
                        eta_normalisation_factor=np.float64(
                            len(fendley.hamiltonian.operators)  # pyright: ignore
                            / fendley.hamiltonian.pauli_l1_norm  # pyright: ignore
                            / (np.sqrt(np.sqrt(2.7)))
                        ),
                        renormalise=True,
                        orthogonal_tolerance=1e-8,
                        max_search_eta_index=expected_rank - 1,
                    )
                    assert generators.num_generators == expected_rank
                    generators.init_eta_currents()
                    fendley.extend_with_currents(  # pyright: ignore
                        generators.eta_currents,
                        [currents_alpha() for _ in generators.eta_currents],
                    )
                    num_verts = len(fendley.hamiltonian.operators)  # pyright: ignore
                    if num_triangles < 6:
                        int_generators = IntegerGenerators(
                            int_simplicial_mode, integer_fendley.hamiltonian
                        )
                        int_generators.init_eta_currents()
                        integer_fendley.extend_with_currents(
                            int_generators.eta_currents,
                            [1 for _ in int_generators.eta_currents],
                        )
                        assert num_verts == len(integer_fendley.hamiltonian.operators)
                    assert num_verts == num_verts_wgs

                num_vertices.append(num_verts_wgs)
            all_num_vertices.append(num_vertices)
        with open(data_file, "w") as f:
            json.dump(
                {
                    "all_num_vertices": all_num_vertices,
                },
                f,
            )
    else:
        with open(data_file, "rb") as f:
            data = json.load(f)
            all_num_vertices = data["all_num_vertices"]

    if do_plot:
        num_axes = len(simplicial_mode_choices)
        fig = plt.figure(figsize=(10, 5 * num_axes))
        gs = fig.add_gridspec(num_axes, 1)
        axes = []
        # fitting_functions = [p2, p3, p4, p5, p6, p7, p8, p9, p10, exp_2, exp]
        fitting_functions = [p, exp]
        colormap = plt.get_cmap("plasma")
        colors = [
            colormap(i / len(fitting_functions)) for i in range(len(fitting_functions))
        ]
        x = [i for i in range(low_num_triangles, up_num_triangles + 1)]
        for i, (y, label) in enumerate(zip(all_num_vertices, simplicial_mode_choices)):
            xcut = x[0:-1]
            ycut = y[0:-1]
            # xcut = x
            # ycut = y
            ax = fig.add_subplot(gs[i, 0])
            axes.append(ax)
            ax.plot(x, y, label="data", color="black")
            ax.set_ylabel(f"Number of vertices with {label} simplicial mode")
            ax.set_xticks(x)
            ax.set_yscale("log")

            for i, fn in enumerate(fitting_functions):
                try:
                    popt, _ = optimize.curve_fit(fn, xcut, ycut)
                    print(popt)
                    long_x = np.arange(low_num_triangles, up_num_triangles + 1, 0.1)
                    ax.plot(
                        long_x,
                        fn(np.array(long_x), *popt),
                        label=f"{fn.__name__} fit",
                        linestyle="dashed",
                        color=colors[i],
                    )
                except RuntimeError:
                    print(f"Could not fit {fn.__name__} for {label} simplicial mode")

        ax = axes[0]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        ax = axes[num_axes - 1]
        ax.set_xlabel("Number of triangles")

        # plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.06, left=0.08, right=0.95)
        plt.savefig("output/final/number_of_vertices.pdf")


def p(
    x,
    a,
    b,
    c,
    d,
    e,
    # f,
    # g,
    # h,
    # i,
    # j,
    # k,
):
    return (
        a * x**2
        + b * x**3
        + c * x**4
        + d * x**5
        + e * x**6
        # + f * x**7
        # + g * x**8
        # + h * x**9
        # +i * x**10
        # + j * x**11
        # + k * x**12
    )


def exp_2(x, a, b):
    return a * np.exp(x ** (1 / 2)) + b


def exp(x, a, b):
    return a * np.exp(x) + b
