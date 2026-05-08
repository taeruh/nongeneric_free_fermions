import os
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
    weight = 1.0 * 10 ** (0.0)
    x = [i for i in range(len(calculation.hl_norms) + 1)][0:9]
    y = []
    for num_currents in x:
        currents_weights = [weight for _ in range(num_currents)]
        # currents_weights = [calculation.hl_norms[i] for i in range(num_currents)]
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

    print(calculation.hl_norms[0:10])


def currents_phase():
    # {{{ interesting settings

    # num_triangles = 75
    # alpha2 = 1
    # beta2 = 1
    # gamma2 = 1
    # x = np.linspace(000, 500, 100)
    # factor = 10 ** (8.0)
    # shared_factor = 10 ** (0.0)
    # x = x * factor

    # num_triangles = 75
    # alpha2 = 1
    # beta2 = 1
    # gamma2 = 1
    # x = np.linspace(120, 180, 100)
    # factor = 10 ** (8.0)
    # # factor = 0.
    # shared_factor = 10 ** (0.0)
    # x = x * factor

    # num_triangles = 100
    # alpha2 = 1
    # beta2 = 1
    # gamma2 = 1
    # x = np.linspace(0.728, 0.73, 200)
    # factor = 10 ** (1.0)
    # shared_factor = 10 ** (0.0)
    # x = x * factor

    # }}}}

    num_triangles = 300
    alpha2 = 1
    beta2 = 1
    gamma2 = 1
    # x = np.linspace(0.715, 0.740, 200)
    x = np.linspace(0.0, 1, 10)
    factor = 10 ** (1.0)
    shared_factor = 10 ** (0.0)
    x = x * factor

    with Wolfram() as wolfram:
        calculation = energy.Calculation(
            num_triangles,
            alpha2,
            beta2,
            gamma2,
            wolfram,
        )
    calculation.calculate_hl_matrices()

    fendley_weight = 1.0
    y = []
    for weight in x:
        currents_weights = [weight * 0 * shared_factor for _ in calculation.hl_norms]
        fendley_weight = 1.0 * shared_factor
        # currents_weights, fendley_weight = (
        #     calculation.hl_norms,
        #     calculation.fendley_norm,
        # )
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


def phase_diagram_calculation(
    wolfram,
    num_triangles,
    currents_weights,
    fendley_weight,
    lower,
    upper,
    num_samples,
    file_name,
):
    points = energy.integer_triangle_grid(num_samples, lower, upper)

    tripoint_gap = None
    corner_gap = None

    values = []
    for alpha2, beta2, gamma2 in points:
        print(alpha2, beta2, gamma2)
        vals = [alpha2, beta2, gamma2]
        vals.sort()
        if vals[0] == 0 and vals[1] == 0:
            values.append(np.sqrt(vals[2]))
            corner_gap = np.sqrt(vals[2])
            continue
        calculation = energy.Calculation(num_triangles, alpha2, beta2, gamma2, wolfram)
        calculation.calculate_hl_matrices()
        # currents_weights, fendley_weight = (
        #     calculation.hl_norms,
        #     calculation.fendley_norm,
        # )
        calculation.extend_model(currents_weights, fendley_weight)
        # calculation.extend_model(currents_weights, 0.0)
        # calculation.extend_model([0 for _ in currents_weights], fendley_weight)
        calculation.compute_gap()
        values.append(calculation.gap)
        if alpha2 == beta2 == gamma2:
            tripoint_gap = calculation.gap
        if alpha2 == lower and beta2 == lower:
            corner_gap = calculation.gap
        # print(alpha, beta, gamma)
        # print(alpha, beta, gamma, calculation.gap)

    print("Tripoint gap:", tripoint_gap)
    print("Corner gap:", corner_gap)

    # print(values)

    x, y = energy.integer_points_to_plot_coordinates(points, lower, upper)
    triang = tri.Triangulation(x, y)
    fig, ax = plt.subplots(figsize=(7, 6))
    # islog = False
    islog = True
    if islog:
        values = np.log(values)
    tpc = ax.tripcolor(triang, values, shading="gouraud")
    # "flat" shading effectively takes the mean value of each triangle corner points and
    # assigns that value to the triangle; that's only sensible if the values at the
    # corners are not too different
    # tpc = ax.tripcolor(
    #     triang, values, shading="flat", vmin=np.min(values), vmax=np.max(values)
    # )
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

    def place_label(ax, alpha, beta, gamma, text, offset=(0, 0), **kwargs):
        p = np.array([[alpha, beta, gamma]], dtype=float)
        p = (p - lower) / (upper - lower)
        x, y = energy.points_to_plot_coordinates(p)
        ax.text(
            x[0] + offset[0],
            y[0] + offset[1],
            text,
            **kwargs,
        )

    place_label(
        ax,
        upper,
        lower,
        lower,
        rf"$\alpha={upper}$",
        ha="right",
        va="top",
        offset=(-0.03, -0.02),
    )
    place_label(
        ax,
        lower,
        upper,
        lower,
        rf"$\beta={upper}$",
        ha="left",
        va="top",
        offset=(0.03, -0.02),
    )
    place_label(
        ax,
        lower,
        lower,
        upper,
        rf"$\gamma={upper}$",
        ha="center",
        va="bottom",
        offset=(0.0, 0.03),
    )
    place_label(
        ax,
        (upper - lower) / 2,
        (upper + lower) / 2,
        lower,
        rf"$\gamma={lower}$",
        ha="center",
        va="top",
        offset=(0.0, -0.04),
    )
    place_label(
        ax,
        (upper + lower) / 2,
        lower,
        (upper + lower) / 2,
        rf"$\beta={lower}$",
        rotation=60,
        ha="right",
        va="center",
        offset=(-0.04, 0.015),
    )
    place_label(
        ax,
        lower,
        (upper + lower) / 2,
        (upper + lower) / 2,
        rf"$\alpha={lower}$",
        rotation=-60,
        ha="left",
        va="center",
        offset=(0.04, 0.015),
    )
    cbar = plt.colorbar(tpc, ax=ax)
    if islog:
        cbar.set_label("Log of gap")
    else:
        cbar.set_label("Gap")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"num_triangles={num_triangles}, currents_weight={currents_weights[0]}, fendley_weight={fendley_weight}",
        pad=50,
    )
    plt.tight_layout()
    plt.savefig(f"output/{file_name}.pdf")


def run_config(config):
    num_triangles, currents_weight, fendley_weight, lower, upper = config
    print(
        f"num_triangles={num_triangles}, currents_weight={currents_weight}, fendley_weight={fendley_weight}, lower={lower}, upper={upper}"
    )
    if num_triangles % 2 == 0:
        num_currents = num_triangles // 2
    else:
        num_currents = (num_triangles + 1) // 2
    phase_diagram_calculation(
        Wolfram(),
        num_triangles,
        [currents_weight for _ in range(num_currents)],
        fendley_weight,
        lower,
        upper,
        upper - lower,
        f"phase_examples/num_triangles_{num_triangles}_currents_weight_{currents_weight}_fendley_weight_{fendley_weight}_lower_{lower}_upper_{upper}",
    )


def fendley_phase():

    # num_triangles = 30
    # # num_triangles = 100
    # if num_triangles % 2 == 0:
    #     num_currents = num_triangles // 2
    # else:
    #     num_currents = (num_triangles + 1) // 2
    # currents_weights = [4.0 for _ in range(num_currents)]
    # fendley_weight = 1.0

    # lower = 1
    # upper = 25
    # num_samples = upper - lower
    # # num_samples = 12

    # phase_diagram_calculation(
    #     num_triangles,
    #     currents_weights,
    #     fendley_weight,
    #     lower,
    #     upper,
    #     num_samples,
    #     f"fendley_phase_{num_triangles}",
    # )

    os.makedirs("output/phase_examples", exist_ok=True)
    configs = [
        # (25, 0.0, 1.0, 1, 20),
        # (25, 1.0, 1.0, 1, 20),
        # (25, 2.0, 1.0, 1, 20),
        # (25, 3.0, 1.0, 1, 20),
        # (25, 4.0, 1.0, 1, 20),
        # (25, 5.0, 1.0, 1, 20),
        # (25, 6.0, 1.0, 1, 20),
        # (25, 7.0, 1.0, 1, 20),
        # (25, 8.0, 1.0, 1, 20),
        # (25, 9.0, 1.0, 1, 20),
        # (25, 10.0, 1.0, 1, 20),
        # (25, 100.0, 1.0, 1, 20),
        # (25, 1000.0, 1.0, 1, 20),
        # (25, 0.0, 1.0, 1, 40),
        # (25, 1.0, 1.0, 1, 40),
        # (25, 2.0, 1.0, 1, 40),
        # (25, 3.0, 1.0, 1, 40),
        # (25, 4.0, 1.0, 1, 40),
        # (25, 5.0, 1.0, 1, 40),
        # (25, 6.0, 1.0, 1, 40),
        # (25, 7.0, 1.0, 1, 40),
        # (25, 8.0, 1.0, 1, 40),
        # (25, 9.0, 1.0, 1, 40),
        # (25, 10.0, 1.0, 1, 40),
        # (25, 100.0, 1.0, 1, 40),
        # (25, 1000.0, 1.0, 1, 40),
        # (50, 0.0, 1.0, 1, 20),
        # (50, 1.0, 1.0, 1, 20),
        # (50, 2.0, 1.0, 1, 20),
        # (50, 3.0, 1.0, 1, 20),
        # (50, 4.0, 1.0, 1, 20),
        # (50, 5.0, 1.0, 1, 20),
        # (50, 6.0, 1.0, 1, 20),
        # (50, 7.0, 1.0, 1, 20),
        # (50, 8.0, 1.0, 1, 20),
        # (50, 9.0, 1.0, 1, 20),
        # (50, 10.0, 1.0, 1, 20),
        # (50, 100.0, 1.0, 1, 20),
        # (50, 1000.0, 1.0, 1, 20),
        # (50, 10000.0, 1.0, 1, 20),
        # (50, 20000.0, 1.0, 1, 20),
        # (50, 30000.0, 1.0, 1, 20),
        # (50, 40000.0, 1.0, 1, 20),
        # (50, 50000.0, 1.0, 1, 20),
        # (50, 100000.0, 1.0, 1, 20),
        # (50, 200000.0, 1.0, 1, 20),
        # (50, 300000.0, 1.0, 1, 20),
        # (50, 400000.0, 1.0, 1, 20),
        # (50, 500000.0, 1.0, 1, 20),
        # (50, 100000.0, 0.0, 1, 20),
        # (50, 200000.0, 0.0, 1, 20),
        # (50, 300000.0, 0.0, 1, 20),
        # (50, 400000.0, 0.0, 1, 20),
        # (50, 500000.0, 0.0, 1, 20),
        # (25, 0.0, 1.0, 1, 20),
        # (25, 0.1, 1.0, 1, 20),
        # (25, 0.2, 1.0, 1, 20),
        # (25, 0.3, 1.0, 1, 20),
        # (25, 0.4, 1.0, 1, 20),
        # (25, 0.5, 1.0, 1, 20),
        # (25, 0.7, 1.0, 1, 20),
        # (25, 0.8, 1.0, 1, 20),
        # (25, 0.9, 1.0, 1, 20),
        # (25, 0.10, 1.0, 1, 20),
        # (25, 0.11, 1.0, 1, 20),
        # (25, 0.12, 1.0, 1, 20),
        # (25, 0.13, 1.0, 1, 20),
        # (25, 0.14, 1.0, 1, 20),
        # (25, 0.15, 1.0, 1, 20),
        # (25, 0.16, 1.0, 1, 20),
        # (25, 0.17, 1.0, 1, 20),
        # (25, 0.18, 1.0, 1, 20),
        # (25, 0.19, 1.0, 1, 20),
        # (25, 0.20, 1.0, 1, 20),
        # (25, 0.21, 1.0, 1, 20),
        # (25, 0.22, 1.0, 1, 20),
        # (25, 0.23, 1.0, 1, 20),
        # (25, 0.24, 1.0, 1, 20),
        # (25, 0.25, 1.0, 1, 20),
        # (25, 0.26, 1.0, 1, 20),
        # (25, 0.27, 1.0, 1, 20),
        # (25, 0.28, 1.0, 1, 20),
        # (25, 0.29, 1.0, 1, 20),
        # (25, 0.30, 1.0, 1, 20),
        # (25, 0.91, 1.0, 1, 20),
        # (25, 0.92, 1.0, 1, 20),
        # (25, 0.93, 1.0, 1, 20),
        # (25, 0.94, 1.0, 1, 20),
        # (25, 0.95, 1.0, 1, 20),
        # (25, 0.96, 1.0, 1, 20),
        # (25, 0.97, 1.0, 1, 20),
        # (25, 0.98, 1.0, 1, 20),
        # (25, 0.99, 1.0, 1, 20),
        # (25, 0.101, 1.0, 1, 20),
        # (25, 0.102, 1.0, 1, 20),
        # (25, 0.103, 1.0, 1, 20),
        # (25, 0.104, 1.0, 1, 20),
        # (25, 0.105, 1.0, 1, 20),
        # (25, 0.106, 1.0, 1, 20),
        # (25, 0.107, 1.0, 1, 20),
        # (25, 0.108, 1.0, 1, 20),
        # (25, 0.109, 1.0, 1, 20),
        (25, 0.999, 1.0, 1, 20),
        (25, 0.9999, 1.0, 1, 20),
        (25, 0.99999, 1.0, 1, 20),
        (25, 0.999999, 1.0, 1, 20),
        (25, 1.0, 1.0, 1, 20),
        (25, 1.100001, 1.0, 1, 20),
        (25, 1.10001, 1.0, 1, 20),
        (25, 1.1001, 1.0, 1, 20),
        (25, 1.101, 1.0, 1, 20),
    ]

    # with Wolfram() as wolfram:
    #     for num_triangles, currents_weight, fendley_weight, lower, upper in configs:
    #         print(
    #             f"num_triangles={num_triangles}, currents_weight={currents_weight}, fendley_weight={fendley_weight}, lower={lower}, upper={upper}"
    #         )
    #         if num_triangles % 2 == 0:
    #             num_currents = num_triangles // 2
    #         else:
    #             num_currents = (num_triangles + 1) // 2
    #         phase_diagram_calculation(
    #             wolfram,
    #             num_triangles,
    #             [currents_weight for _ in range(num_currents)],
    #             fendley_weight,
    #             lower,
    #             upper,
    #             upper - lower,
    #             f"phase_examples/num_triangles_{num_triangles}_currents_weight_{currents_weight}_fendley_weight_{fendley_weight}_lower_{lower}_upper_{upper}",
    #         )

    from multiprocessing import Pool
    with Pool() as pool:
        pool.map(run_config, configs)
