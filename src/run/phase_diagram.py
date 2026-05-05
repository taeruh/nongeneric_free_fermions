import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import energy
from energy import Wolfram


def run():
    num_triangles = 15
    # num_triangles = 100
    if num_triangles % 2 == 0:
        num_currents = num_triangles // 2
    else:
        num_currents = (num_triangles + 1) // 2
    currents_weights = [1.0 for _ in range(num_currents)]
    fendley_weight = 1.0

    # calculation = energy.Calculation(num_triangles, 1., 0., 0., currents_alpha)

    factor = 3.0
    num_samples = 25
    # num_samples = 50
    # points = energy.triangle_grid(num_samples, factor)
    points = energy.triangle_grid_with_minimum_bound(
        num_samples, factor, minimum_bound=0.40
    )

    # with Wolfram() as wolfram:
    #     calculation = energy.Calculation(
    #         num_triangles,
    #         1.0**2,
    #         1.0**2,
    #         1.0**2,
    #         wolfram,
    #     )
    #     calculation.calculate_hl_matrices()
    #     # currents_weights, fendley_weight = (
    #     #     calculation.hl_norms,
    #     #     calculation.fendley_norm,
    #     # )
    #     calculation.extend_model(currents_weights, fendley_weight)
    #     calculation.compute_gap()
    #     print(calculation.hl_norms)
    #     print(calculation.fendley_norm)
    #     print(calculation.gap)
    #     assert False

    values = []
    with Wolfram() as wolfram:
        for alpha, beta, gamma in points:
            # print(alpha, beta, gamma)
            vals = [alpha, beta, gamma]
            vals.sort()
            if vals[0] == 0 and vals[1] == 0:
                values.append(factor)
                continue
            calculation = energy.Calculation(
                num_triangles, alpha**2, beta**2, gamma**2, wolfram
            )
            calculation.calculate_hl_matrices()
            # currents_weights, fendley_weight = (
            #     calculation.hl_norms,
            #     calculation.fendley_norm,
            # )
            # calculation.extend_model(currents_weights, fendley_weight)
            calculation.extend_model(currents_weights, 0.0)
            calculation.compute_gap()
            values.append(calculation.gap)
            # print(alpha, beta, gamma)
            # print(alpha, beta, gamma, calculation.gap)

    # print(values)

    x, y = energy.points_to_plot_coordinates(points)
    triang = tri.Triangulation(x, y)
    _, ax = plt.subplots(figsize=(7, 6))
    values = np.log(values)
    tpc = ax.tripcolor(triang, values, shading="gouraud")
    vertices_bary = np.array(
        [
            [factor, 0, 0],
            [0, factor, 0],
            [0, 0, factor],
            [factor, 0, 0],  # close the loop
        ]
    )
    vx, vy = energy.points_to_plot_coordinates(vertices_bary)
    ax.plot(vx, vy, lw=1)

    def place_label(ax, alpha, beta, gamma, text, **kwargs):
        x, y = energy.points_to_plot_coordinates(np.array([[alpha, beta, gamma]]))
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
    plt.savefig(f"output/phase_diagram_{num_triangles}.pdf")
