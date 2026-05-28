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
from . import utils


def run():
    low_num_triangles = 1
    up_num_triangles = 13
    # up_num_triangles = 8

    alpha = ConstantWeight(np.float64(1))
    beta = ConstantWeight(np.float64(1))
    gamma = ConstantWeight(np.float64(1))
    currents_alpha = ConstantWeight(np.float64(1))
    simplicial_mode_choices = ["IIZZZ", "IZZZZ", "ZZZZZ"]
    # simplicial_mode_choices = ["IIZZZ"]

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
        utils.paper_setup()
        fig = plt.figure(figsize=utils.set_size(height_in_width=0.7))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        poly_degree = 5
        fitting_functions = [
            (exp, rf"$\text{{exp}}$"),
            (mono(poly_degree), rf"$\text{{mono}}[{poly_degree}]$"),
            (poly(poly_degree), rf"$\text{{poly}}[{poly_degree}]$"),
        ]
        linestyles = ["dashed", "dashdot", "dotted"]
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        labels = [
            r"\sigma^z_1 \sigma^z_2 \sigma^z_3",
            r"\sigma^z_1 \sigma^z_2 \sigma^z_3 \sigma^z_4",
            r"\sigma^z_1 \sigma^z_2 \sigma^z_3 \sigma^z_4 \sigma^z_5",
        ]
        x = [i for i in range(low_num_triangles, up_num_triangles + 1)]
        for i, (y, label) in enumerate(zip(all_num_vertices, simplicial_mode_choices)):
            ax.plot(x, y, label=rf"$\chi = {labels[i]}$", color=colors[i])

        y = all_num_vertices[0]
        # xcut = x[1:-1]
        # ycut = y[1:-1]
        xcut = x
        ycut = y
        for i, (fn, fn_name) in enumerate(fitting_functions):
            try:
                popt, _ = optimize.curve_fit(fn, xcut, ycut)
                print(popt)
                long_x = np.arange(low_num_triangles, up_num_triangles + 1, 0.1)
                ax.plot(
                    long_x,
                    fn(np.array(long_x), *popt),
                    label=rf"{fn_name} fit ${labels[0]}$",
                    linestyle=linestyles[i],
                    color=colors[3],
                )
            except RuntimeError:
                print(f"Could not fit {fn.__name__}")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        ax.set_ylabel(r"Number of vertices $N_V$")
        ax.set_xticks(x)
        ax.set_yscale("log")
        ax.set_xlabel(r"Number of triangles $N$")
        plt.subplots_adjust(top=0.97, bottom=0.10, left=0.08, right=0.95)
        plt.savefig("output/final/number_of_vertices.pdf")


def poly(degree):
    if degree == 0:
        return lambda _, a: a
    elif degree == 1:
        return lambda x, a, b: a * x + b
    elif degree == 2:
        return lambda x, a, b, c: a * x**2 + b * x + c
    elif degree == 3:
        return lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d
    elif degree == 4:
        return lambda x, a, b, c, d, e: a * x**4 + b * x**3 + c * x**2 + d * x + e
    elif degree == 5:
        return lambda x, a, b, c, d, e, f: (
            a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
        )
    elif degree == 6:
        return lambda x, a, b, c, d, e, f, g: (
            a * x**6 + b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g
        )
    elif degree == 7:
        return lambda x, a, b, c, d, e, f, g, h: (
            a * x**7 + b * x**6 + c * x**5 + d * x**4 + e * x**3 + f * x**2 + g * x + h
        )
    elif degree == 8:
        return lambda x, a, b, c, d, e, f, g, h, i: (
            a * x**8
            + b * x**7
            + c * x**6
            + d * x**5
            + e * x**4
            + f * x**3
            + g * x**2
            + h * x
            + i
        )
    elif degree == 9:
        return lambda x, a, b, c, d, e, f, g, h, i, j: (
            a * x**9
            + b * x**8
            + c * x**7
            + d * x**6
            + e * x**5
            + f * x**4
            + g * x**3
            + h * x**2
            + i * x
            + j
        )
    else:
        raise ValueError("not implemented")


def mono(degree):
    if degree == 0:
        return lambda _, a: a
    elif degree == 1:
        return lambda x, a: a * x
    elif degree == 2:
        return lambda x, a: a * x**2
    elif degree == 3:
        return lambda x, a: a * x**3
    elif degree == 4:
        return lambda x, a: a * x**4
    elif degree == 5:
        return lambda x, a: a * x**5
    elif degree == 6:
        return lambda x, a: a * x**6
    elif degree == 7:
        return lambda x, a: a * x**7
    elif degree == 8:
        return lambda x, a: a * x**8
    elif degree == 9:
        return lambda x, a: a * x**9
    else:
        raise ValueError("not implemented")


def exp_2(x, a, b):
    return a * np.exp(x ** (1 / 2)) + b


def exp(x, a, b):
    return a * np.exp(x) + b
