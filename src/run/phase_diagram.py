import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import energy


def run():
    num_triangles = 200
    if num_triangles % 2 == 0:
        num_currents = num_triangles // 2
    else:
        num_currents = (num_triangles - 1) // 2
    currents_alpha = [1.0 for _ in range(num_currents)]

    # calculation = energy.Calculation(num_triangles, 1., 0., 0., currents_alpha)

    factor = 3.0
    num_samples = 40
    points = energy.triangle_grid(num_samples, factor)

    values = []

    alpha, beta, gamma = 1., np.pi, 1.1
    # calculation = energy.SamsCalculation(
    #     3 * num_triangles, alpha, beta, gamma, 1, 1e-4, 1e-9
    # )
    # calculation = energy.Calculation(
    #     num_triangles, alpha**2, beta**2, gamma**2, currents_alpha
    # )

    for alpha, beta, gamma in points:
        calculation = energy.Calculation(
            num_triangles, alpha**2, beta**2, gamma**2, currents_alpha
        )
        sub = "mine"
        # calculation = energy.SamsCalculation(
        #     3 * num_triangles, alpha, beta, gamma, 1, 1e-4, 1e-9
        # )
        # sub = "sams"
        values.append(calculation.gap)
        # print(alpha, beta, gamma, calculation.gap_real)
        # print(alpha, beta, gamma)

    x, y = energy.points_to_plot_coordinates(points)
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
    plt.savefig(f"output/phase_diagram_{num_triangles}_{sub}.pdf")
