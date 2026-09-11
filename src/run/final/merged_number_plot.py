import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.set_loglevel("debug")
from scipy import optimize
import json

from models.weights import ConstantWeight
from . import utils, number_of_vertices

exp = number_of_vertices.exp
mono = number_of_vertices.mono


def run():
    alpha = ConstantWeight(np.float64(1))
    beta = ConstantWeight(np.float64(1))
    gamma = ConstantWeight(np.float64(1))
    currents_alpha = ConstantWeight(np.float64(1))
    simplicial_mode_choices = ["IIZZZ", "IZZZZ", "ZZZZZ"]

    vertex_low_num_triangles = 1
    vertex_up_num_triangles = 13
    vertex_identifier = (
        f"num_vertices_{alpha}_{beta}_{gamma}_{currents_alpha}"
        + "-".join(simplicial_mode_choices)
        + f"_{vertex_low_num_triangles}_{vertex_up_num_triangles}"
    )
    claw_low_num_triangles = 1
    claw_up_num_triangles = 6
    claw_identifier = (
        f"num_claws_{alpha}_{beta}_{gamma}_{currents_alpha}_{claw_low_num_triangles}_{claw_up_num_triangles}_"
        + "-".join(simplicial_mode_choices)
    )

    data_file = lambda identifier: f"output/final/data/{identifier}.json"

    with open(data_file(vertex_identifier), "r") as f:
        vertex_data = json.load(f)
    with open(data_file(claw_identifier), "r") as f:
        claw_data = json.load(f)
    all_num_vertices = vertex_data["all_num_vertices"]
    all_num_claws = claw_data["all_num_claws"]

    utils.paper_setup()
    fig = plt.figure(figsize=utils.set_size(height_in_width=1.3))
    gs = fig.add_gridspec(2, 1)
    gs.update(hspace=0.35)

    ax = fig.add_subplot(gs[0, 0])

    poly_degree = 5
    fitting_functions = [
        (exp, rf"$\text{{exp}}$"),
        (mono(poly_degree), rf"$\text{{mono}}[{poly_degree}]$"),
    ]
    linestyles = ["dashed", "dotted", "dashdot"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    labels = [
        r"Z_1 \cdots Z_3",
        r"Z_1 \cdots Z_4",
        r"Z_1 \cdots Z_5",
    ]
    x = [i for i in range(vertex_low_num_triangles, vertex_up_num_triangles + 1)]
    for i, (y, label) in enumerate(zip(all_num_vertices, simplicial_mode_choices)):
        # ax.plot(x, y, label=rf"$\chi = {labels[i]}$", color=colors[i])
        ax.plot(x, y, color=colors[i])

    y = all_num_vertices[0]
    # xcut = x[1:-1]
    # ycut = y[1:-1]
    xcut = x
    ycut = y
    for i, (fn, fn_name) in enumerate(fitting_functions):
        try:
            popt, _ = optimize.curve_fit(fn, xcut, ycut)
            print(popt)
            long_x = np.arange(
                vertex_low_num_triangles, vertex_up_num_triangles + 1, 0.1
            )
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
    ax.set_xlabel(r"Independence number $\alpha(G)$")
    ax.text(-0.1, 1.00, r"(a)", transform=ax.transAxes)

    ax = fig.add_subplot(gs[1, 0])
    x = [i for i in range(claw_low_num_triangles, claw_up_num_triangles + 1)]
    labels = [
        r"Z_1 \cdots Z_3",
        r"Z_1 \cdots Z_4",
        r"Z_1 \cdots Z_5",
    ]
    for y, label in zip(
        all_num_claws,
        labels,
    ):
        ax.plot(x, y, label=rf"$\chi = {label}$")
    ax.set_ylabel(
        # matplotlib does something with minus signs so instead of -\hspace{-1.2pt}\langle
        # we do the following
        r"Number of claws $N_{\text{-}\hspace{-1.2pt}\text{-}\hspace{-1.8pt}\langle}$"
    )
    # ax.set_ylabel(r"Number of claws $N_{\hyphen\vspace{-2pt}\langle}$")
    # ax.set_ylabel(r"Number of claws $--N_{\vspace{-2pt}\langle}$")
    # ax.set_ylabel(r"Number of claws $N_{-\langle}$")
    # ax.set_ylabel(fr"Number of claws ${u"\u2212"}N$")
    matplotlib.rcParams["axes.unicode_minus"] = True
    ax.set_xticks(x)
    ax.set_yscale("log")
    ax.set_xlabel(r"Independence number $\alpha(G)$")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left")
    ax.text(-0.1, 1.00, r"(b)", transform=ax.transAxes)

    plt.subplots_adjust(top=0.97, bottom=0.085, left=0.13, right=0.97)
    # plt.subplots_adjust(top=0.97, bottom=0.085, left=0.33, right=0.97)
    plt.savefig("output/final/merged_number_plot.pdf")
