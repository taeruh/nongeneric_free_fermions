#!/usr/bin/env python

import os
import json
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from numpy.typing import NDArray


def main():
    os.makedirs("output", exist_ok=True)
    with open("phase_diagram_data.json", "r") as f:
        data = json.load(f)
    num_triangles_range = data["num_triangles_range"]
    factor = data["factor"]
    points = triangle_grid(data["num_samples"], factor)
    all_values = np.array(data["all_values"])

    ######################################################################################

    # plotting options {{{
    shading = True
    # shading = False

    # cf. https://matplotlib.org/stable/tutorials/colors/colormaps.html
    colormap = "viridis"
    # colormap = "plasma"

    scale_fn = lambda x: x
    # scale_fn = lambda x: np.log10(x)
    # with that we can already see numerical inaccuracies, I believe, for 6 triangles
    # scale_fn = lambda x: 10 ** x
    # }}}

    # example data point {{{
    num_triangle_index = 3
    data_index = 42

    num_triangles = [
        i for i in range(num_triangles_range[0], num_triangles_range[1] + 1)
    ][num_triangle_index]
    values = all_values[num_triangle_index]
    point = points[data_index]
    value = values[data_index]
    print(f"num_triangles: {num_triangles}")
    print(f"point: alphas={point[0]}, beta={point[1]}, gamma={point[2]}")
    print(f"value: {value}")
    # }}}

    #######################################################################################

    for i, num_triangles in enumerate(
        range(num_triangles_range[0], num_triangles_range[1] + 1)
    ):
        x, y = points_to_plot_coordinates(points)
        values = [scale_fn(v) for v in all_values[i]]
        triang = tri.Triangulation(x, y)
        _, ax = plt.subplots(figsize=(7, 6))
        if shading:
            tpc = ax.tripcolor(triang, values, shading="gouraud", cmap=colormap)
        else:
            tpc = ax.tripcolor(triang, values, cmap=colormap)
        vertices_bary = np.array(
            [
                [factor, 0, 0],
                [0, factor, 0],
                [0, 0, factor],
                [factor, 0, 0],
            ]
        )
        vx, vy = points_to_plot_coordinates(vertices_bary)
        ax.plot(vx, vy, lw=1)

        def place_label(ax, alpha, beta, gamma, text, **kwargs):
            x, y = points_to_plot_coordinates(np.array([[alpha, beta, gamma]]))
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


def triangle_grid(num_samples: int, factor: float = 3) -> NDArray:
    points = []
    for i in range(num_samples + 1):
        for j in range(num_samples + 1 - i):
            k = num_samples - i - j
            alpha = i / num_samples
            beta = j / num_samples
            gamma = k / num_samples
            points.append([alpha, beta, gamma])
    return np.array(points) * factor


def points_to_plot_coordinates(points: NDArray) -> tuple[NDArray, NDArray]:
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.5, np.sqrt(3) / 2])
    xy = points[:, 0, None] * a + points[:, 1, None] * b + points[:, 2, None] * c
    return xy[:, 0], xy[:, 1]


if __name__ == "__main__":
    main()
