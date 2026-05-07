import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import energy
from energy import Wolfram
from models.fendley import Fendley
from models.weights import ConstantWeight
import independence_polynomial


def add_currents_phase():
    num_triangles = 100
    # num_triangles = 10
    alpha2 = 1
    beta2 = 1
    gamma2 = 1
    # alpha2 = 4
    # beta2 = 8
    # gamma2 = 8

    # num_triangles = 18
    # alpha = np.pi
    # beta = np.sqrt(2)
    # gamma = 10.0

    # fendley = Fendley(
    #     num_triangles-1,
    #     ConstantWeight(alpha),
    #     ConstantWeight(beta),
    #     ConstantWeight(gamma),
    # )
    # graph = fendley.hamiltonian.get_frustration_graph()
    # weights = fendley.hamiltonian.weights
    # p = independence_polynomial.independence_polynomial(
    #     graph, weights, num_triangles-1 + 1
    # )
    # print(p)

    with Wolfram() as wolfram:
        calculation = energy.Calculation(
            num_triangles,
            alpha2,
            beta2,
            gamma2,
            wolfram,
        )
    # print(calculation.pm_factors)
    # print(type(calculation.pm_factors[0]))
    # return
    calculation.calculate_hl_matrices()
    # calculation.extend_model([1.0 for _ in calculation.hl_norms], 1.0)
    # calculation.compute_gap()
    # return
    # print(calculation.lagrange)
    # print(calculation.norm)
    # print(calculation.hl_norms)

    fendley_weight = 1.0
    weight = 1.0 * 10**(10)
    x = [i for i in range(len(calculation.hl_norms) + 1)]
    y = []
    for num_currents in x:
        currents_weights = [weight for _ in range(num_currents)]
        for _ in range(len(calculation.hl_norms) - len(currents_weights)):
            currents_weights.append(0)
        calculation.extend_model(currents_weights, fendley_weight)
        calculation.compute_gap()
        y.append(calculation.gap)
        print(calculation.gap)

    _, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("# Currents")
    ax.set_ylabel("Gap")
    plt.tight_layout()
    plt.savefig(f"output/add_currents_phase_{num_triangles}.pdf")


def currents_phase():
    # {{{ interesting settings

    # num_triangles = 15
    # x = np.linspace(0.12879530, 0.12879531, 100)
    # factor = 10**18
    # shared_factor = 10**(-10)

    # num_triangles = 15
    # x = np.linspace(0.77183422398 * 10**6, 0.771834224 * 10**6, 500)
    # factor = 10 ** (5.0)
    # shared_factor = 10 ** (2)

    # num_triangles = 15
    # x = np.linspace(0.0, 2, 300)
    # # x = np.linspace(0.25, 0.5, 300)
    # factor = 10 ** (5.0)
    # shared_factor = 10 ** (2)

    # num_triangles = 15
    # x = np.linspace(0. * 10**6, 0.772 * 10**6, 500)
    # factor = 10 ** (5.0)
    # shared_factor = 10 ** (2)

    # num_triangles = 50
    # x = np.linspace(0.0, 0.3, 300)
    # factor = 10 ** (4.0)
    # shared_factor = 10 ** (2)

    # num_triangles = 50
    # x = np.linspace(0.0, 0.01, 30)
    # factor = 10 ** (1.0)
    # shared_factor = 10 ** (2)

    # }}}}

    num_triangles = 75
    alpha2 = 1
    beta2 = 1
    gamma2 = 1
    x = np.linspace(0.0, 500, 20)
    factor = 10 ** (-1.0)
    shared_factor = 10 ** (+3.0)

    with Wolfram() as wolfram:
        calculation = energy.Calculation(
            num_triangles,
            alpha2,
            beta2,
            gamma2,
            wolfram,
        )
    return
    calculation.calculate_hl_matrices()

    fendley_weight = 1.0
    y = []
    for weight in x:
        currents_weights = [
            weight * factor * shared_factor for _ in calculation.hl_norms
        ]
        fendley_weight = 1.0 * shared_factor
        calculation.extend_model(currents_weights, fendley_weight)
        calculation.compute_gap()
        y.append(calculation.gap)
        print(calculation.gap)

    _, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("Currents weight")
    ax.set_ylabel("Gap")
    plt.tight_layout()
    plt.savefig(f"output/currents_phase_{num_triangles}.pdf")


def fendley_phase():
    num_triangles = 50
    # num_triangles = 100
    if num_triangles % 2 == 0:
        num_currents = num_triangles // 2
    else:
        num_currents = (num_triangles + 1) // 2
    currents_weights = [5.0 for _ in range(num_currents)]
    fendley_weight = 1.0

    # calculation = energy.Calculation(num_triangles, 1., 0., 0., currents_alpha)

    lower = 0
    upper = 5
    num_samples = 5
    points = energy.integer_triangle_grid(num_samples, lower, upper)

    values = []
    with Wolfram() as wolfram:
        for alpha2, beta2, gamma2 in points:
            print(alpha2, beta2, gamma2)
            vals = [alpha2, beta2, gamma2]
            vals.sort()
            if vals[0] == 0 and vals[1] == 0:
                values.append(upper)
                continue
            calculation = energy.Calculation(
                num_triangles, alpha2, beta2, gamma2, wolfram
            )
            calculation.calculate_hl_matrices()
            currents_weights, fendley_weight = (
                calculation.hl_norms,
                calculation.fendley_norm,
            )
            calculation.extend_model(currents_weights, fendley_weight)
            # calculation.extend_model(currents_weights, 0.0)
            # calculation.extend_model([0 for _ in currents_weights], fendley_weight)
            calculation.compute_gap()
            values.append(calculation.gap)
            # print(alpha, beta, gamma)
            # print(alpha, beta, gamma, calculation.gap)

    # print(values)

    x, y = energy.integer_points_to_plot_coordinates(points, lower, upper)
    triang = tri.Triangulation(x, y)
    _, ax = plt.subplots(figsize=(7, 6))
    values = np.log(values)
    # tpc = ax.tripcolor(triang, values, shading="gouraud")
    tpc = ax.tripcolor(triang, values, shading="flat", cmap="viridis")
    vertices_bary = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=float,
    )
    vx, vy = energy.points_to_plot_coordinates(vertices_bary)
    ax.plot(vx, vy, lw=1)

    def place_label(ax, alpha, beta, gamma, text, **kwargs):
        p = np.array([[alpha, beta, gamma]], dtype=float)
        p = (p - lower) / (upper - lower)
        x, y = energy.points_to_plot_coordinates(p)
        ax.text(x[0], y[0], text, **kwargs)

    place_label(ax, upper, 0, 0, rf"$\alpha={upper}$", ha="right", va="top")
    place_label(ax, 0, upper, 0, rf"$\beta={upper}$", ha="left", va="top")
    place_label(ax, 0, 0, upper, rf"$\gamma={upper}$", ha="center", va="bottom")
    place_label(ax, upper / 2, upper / 2, 0, r"$\gamma=0$", ha="center", va="top")
    place_label(ax, upper / 2, 0, upper / 2, r"$\beta=0$", rotation=60, ha="right")
    place_label(ax, 0, upper / 2, upper / 2, r"$\alpha=0$", rotation=-60, ha="left")
    cbar = plt.colorbar(tpc, ax=ax)
    cbar.set_label("Function value")
    ax.set_aspect("equal")
    ax.axis("off")
    # ax.set_xlim(-0.05, 1.05)
    # ax.set_ylim(-0.05, np.sqrt(3)/2 + 0.05)
    plt.tight_layout()
    plt.savefig(f"output/phase_diagram_{num_triangles}.pdf")
