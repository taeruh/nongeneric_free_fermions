import json
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize
import energy
from energy import Wolfram
from . import utils


def run():

    name = "dip1"
    appendix = True
    num_triangles = 20
    fendley_weight = 1.0
    abc_lower = 1
    abc_upper = 300
    num_samples = abc_upper - abc_lower
    cw_low = 1.5
    cw_high = 2.1
    num_points = 15
    num_rows = 4
    num_cols = 4

    # name = "dip2"
    # appendix = True
    # num_triangles = 20
    # fendley_weight = 1.0
    # abc_lower = 1
    # abc_upper = 300
    # num_samples = abc_upper - abc_lower
    # cw_low = 25
    # cw_high = 39
    # num_points = 15
    # num_rows = 4
    # num_cols = 4

    # name = "dip1"
    # appendix = False
    # num_triangles = 20
    # fendley_weight = 1.0
    # abc_lower = 1
    # abc_upper = 300
    # num_samples = abc_upper - abc_lower
    # cw_low = 1.75
    # cw_high = 1.85
    # num_points = 3
    # num_rows = 2
    # num_cols = 2

    diff = num_rows * num_cols - num_points
    assert (
        diff == 1
    ), f"num_points should be one less than gridspec size, but got diff={diff} for num_points={num_points}, num_rows={num_rows}, num_cols={num_cols}"

    data_file = (
        "output/final/data/triangle_phase_"
        + f"{num_triangles}_{fendley_weight}_{abc_lower}_{abc_upper}_{num_samples}_"
        + f"{cw_low}_{cw_high}_{num_points}.json"
    )

    currents_weight_factors = np.linspace(cw_low, cw_high, num_points)
    # print(currents_weight_factors)
    # return
    configs = [
        (num_triangles, x, fendley_weight, abc_lower, abc_upper, num_samples)
        for x in currents_weight_factors
    ]
    configs.append(
        (num_triangles, 0.0, fendley_weight, abc_lower, abc_upper, num_samples)
    )

    # do_calculation = True
    do_calculation = False
    # islog = False
    islog = True

    if do_calculation:
        all_values = []
        with Pool(8) as pool:
            pool.map(run_config, configs)

        for _, currents_weight_factor, _, _, _, _ in configs:
            file_name = (
                "output/data_triangle_phase_"
                + f"{num_triangles}_{currents_weight_factor}_{fendley_weight}_"
                + f"{abc_lower}_{abc_upper}_{num_samples}.json"
            )
            with open(file_name, "r") as f:
                data = json.load(f)
                all_values.append(data["values"])
        with open(data_file, "w") as f:
            json.dump(
                {
                    "num_triangles": num_triangles,
                    "fendley_weight": fendley_weight,
                    "abc_lower": abc_lower,
                    "abc_upper": abc_upper,
                    "num_samples": num_samples,
                    "cw_low": cw_low,
                    "cw_high": cw_high,
                    "num_points": num_points,
                    "all_values": all_values,
                },
                f,
            )
    else:
        with open(data_file, "r") as f:
            all_values = json.load(f)["all_values"]

    utils.paper_setup()

    if appendix:
        columnwidth = 510
    else:
        columnwidth = 246
    fig = plt.figure(figsize=utils.set_size(columnwidth))
    gs = fig.add_gridspec(num_rows, num_cols)
    if appendix:
        gs.update(wspace=0.25, hspace=-0.5)
    else:
        gs.update(wspace=0.25, hspace=-0.3)

    points = energy.integer_triangle_grid(num_samples, abc_lower, abc_upper)
    x, y = energy.integer_points_to_plot_coordinates(points, abc_lower, abc_upper)
    triang = tri.Triangulation(x, y)

    # # this does not give really nice plots
    # #
    # # use the calculated norm in the triplot below
    # all_values_for_scaling = []
    # if islog:
    #     for values in all_values:
    #         for value in values:
    #             all_values_for_scaling.append(np.log(value))
    # else:
    #     for values in all_values:
    #         for value in values:
    #             all_values_for_scaling.append(value)
    # min_value = min(all_values_for_scaling)
    # max_value = max(all_values_for_scaling)
    # plot_norm = Normalize(vmin=min_value, vmax=max_value)

    def plot_it(ax, values, cw):
        if islog:
            values = np.log(values)
        # tpc = ax.tripcolor(triang, values, shading="gouraud",norm=plot_norm)
        tpc = ax.tripcolor(triang, values, shading="gouraud")
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

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.09)
        cbar = plt.colorbar(tpc, cax=cax)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            rf"$a = {cw:.3f}$",
            pad=7,
        )
        return cbar

    for i, currents_weight_factor in enumerate(currents_weight_factors):
        ax = fig.add_subplot(gs[i // num_cols, i % num_cols])
        values = all_values[i]
        _ = plot_it(ax, values, currents_weight_factor)

    ax = fig.add_subplot(gs[-1, -1])
    values = all_values[-1]
    cbar = plot_it(ax, values, 0.0)
    if islog:
        cbar.set_label(r"$\ln\left(\Delta\right)$")
    else:
        cbar.set_label(r"$\Delta$")

    def place_label(ax, alpha, beta, gamma, text, offset=(0, 0), **kwargs):
        p = np.array([[alpha, beta, gamma]], dtype=float)
        p = (p - abc_lower) / (abc_upper - abc_lower)
        x, y = energy.points_to_plot_coordinates(p)
        ax.text(
            x[0] + offset[0],
            y[0] + offset[1],
            text,
            **kwargs,
        )

    place_label(
        ax,
        abc_upper,
        abc_lower,
        abc_lower,
        rf"$\alpha={abc_upper}$",
        ha="right",
        va="top",
        offset=(0.13, -0.04),
    )
    place_label(
        ax,
        abc_lower,
        abc_upper,
        abc_lower,
        rf"$\beta={abc_upper}$",
        ha="left",
        va="top",
        offset=(-0.13, -0.04),
    )
    place_label(
        ax,
        abc_lower,
        abc_lower,
        abc_upper,
        rf"$\gamma={abc_upper}$",
        ha="center",
        va="bottom",
        offset=(0.0, -0.02),
    )
    place_label(
        ax,
        (abc_upper - abc_lower) / 2,
        (abc_upper + abc_lower) / 2,
        abc_lower,
        rf"$\gamma={abc_lower}$",
        ha="center",
        va="top",
        offset=(0.0, -0.04),
    )
    place_label(
        ax,
        (abc_upper + abc_lower) / 2,
        abc_lower,
        (abc_upper + abc_lower) / 2,
        rf"$\beta={abc_lower}$",
        rotation=60,
        ha="right",
        va="center",
        offset=(-0.04, 0.015),
    )
    place_label(
        ax,
        abc_lower,
        (abc_upper + abc_lower) / 2,
        (abc_upper + abc_lower) / 2,
        rf"$\alpha={abc_lower}$",
        rotation=-60,
        ha="left",
        va="center",
        offset=(0.04, 0.015),
    )

    if appendix:
        plt.subplots_adjust(left=0.01, right=0.95, top=1.15, bottom=-0.15)
    else:
        plt.subplots_adjust(left=0.01, right=0.91, top=1.2, bottom=-0.2)
    # plt.savefig(f"output/final/triangle_phase_{name}_appendix_{appendix}.pdf")
    plt.savefig(f"output/final/triangle_phase_{name}_appendix_{appendix}.png")


def run_config(config):
    num_triangles, currents_weight_factor, fendley_weight, lower, upper, num_samples = (
        config
    )
    with Wolfram() as wolfram:
        ret = phase_diagram_calculation(
            wolfram,
            num_triangles,
            currents_weight_factor,
            fendley_weight,
            lower,
            upper,
            num_samples,
            True,
        )
    return ret


def phase_diagram_calculation(
    wolfram,
    num_triangles,
    currents_weight_factor,
    fendley_weight,
    lower,
    upper,
    num_samples,
    serialize,
):
    if num_triangles % 2 == 0:
        num_currents = num_triangles // 2
    else:
        num_currents = (num_triangles + 1) // 2
    currents_weights = [currents_weight_factor] * num_currents

    points = energy.integer_triangle_grid(num_samples, lower, upper)

    ret = []
    for alpha2, beta2, gamma2 in points:
        # print(alpha2, beta2, gamma2)
        vals = [alpha2, beta2, gamma2]
        vals.sort()
        # all vertices are independent so the gap is just the vertex weight
        # NOTE: this actually not that simple with the currents; the currents are not
        # actually zero in this case, so they add something
        # EDIT: this case behaves very differently in general, because then we only get 2
        # etas in the krylov basis (instead of 2alpha(-1)); this then gives us a single
        # current which is just the first vertex (or second or third, depending on which
        # weight is nonzero) but with twice the weight (with the same sign); so for
        # positive currents_weights[0] the gap does not change, but for negative it may
        # change, e.g., if we choose currents_weights[0] = -fendley_weight / num_triangles
        # then the resulting first vertex weight is zero!
        if vals[0] == 0 and vals[1] == 0:
            fendley_vertex_weight = np.sqrt(vals[2])
            fendley_norm = num_triangles * fendley_vertex_weight
            currents_vertex_weight = 2 * fendley_vertex_weight
            currents_norm = currents_vertex_weight
            first_vertex_weight = (
                fendley_weight * fendley_vertex_weight
                + currents_weights[0]
                * currents_vertex_weight
                * fendley_norm
                / currents_norm
            )
            other_vertex_weight = fendley_weight * fendley_vertex_weight
            print("Vertex weights:", first_vertex_weight, other_vertex_weight)
            gap = min(abs(first_vertex_weight), abs(other_vertex_weight))
        else:
            calculation = energy.Calculation(
                num_triangles, alpha2, beta2, gamma2, wolfram
            )
            calculation.calculate_hl_matrices()
            calculation.extend_model(currents_weights, fendley_weight)
            calculation.compute_gap()
            # cf. comment in currents_phase.py about the gap precision
            gap = np.float64(calculation.gap)
        ret.append(gap)

    if serialize:
        file_name = (
            "output/data_triangle_phase_"
            + f"{num_triangles}_{currents_weight_factor}_{fendley_weight}_"
            + f"{lower}_{upper}_{num_samples}.json"
        )
        with open(file_name, "w") as f:
            json.dump(
                {
                    "values": ret,
                },
                f,
            )
    else:
        return ret
