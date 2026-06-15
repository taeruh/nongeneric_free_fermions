import json
import numpy as np
import matplotlib.pyplot as plt
import energy
from energy import Wolfram
from . import utils


def run():
    alpha2 = 1
    beta2 = 1
    gamma2 = 1
    fendley_weight = 1.0
    cw_low = -1
    cw_high = 14
    num_points = 40000
    # num_points = 20000
    # num_points = 20
    nums_triangles = [20, 30, 50, 100]
    # nums_triangles = [100]

    # cw_low = 0
    # cw_high = np.log10(2)
    # num_points = 20
    # nums_triangles = [20]

    data_file = (
        "output/final/data/currents_phase_"
        + f"{alpha2}_{beta2}_{gamma2}_{fendley_weight}_{cw_low}_{cw_high}_{num_points}_"
        + "-".join(map(str, nums_triangles))
        + ".json"
    )

    currents_weight = np.linspace(cw_low, cw_high, num_points)
    currents_weight = 10**currents_weight

    do_calculation = True
    # do_calculation = False

    if do_calculation:
        gaps = []
        with Wolfram() as wolfram:
            for num_triangles in nums_triangles:
                print(f"Calculating for {num_triangles} triangles...")
                calculation = energy.Calculation(
                    num_triangles,
                    alpha2,
                    beta2,
                    gamma2,
                    wolfram,
                )
                calculation.calculate_hl_matrices()
                gap = []
                for weight in currents_weight:
                    currents_weights = [weight] * len(calculation.hl_norms)
                    calculation.extend_model(currents_weights, fendley_weight)
                    calculation.compute_gap()
                    # the gap comes as np.float128, for which the serialization is not
                    # implemented, so we just convert it to np.float64, since it does not
                    # actually require the precision of np.float128 here
                    gap.append(np.float64(calculation.gap))
                gaps.append(gap)
        with open(data_file, "w") as f:
            json.dump(
                {
                    "alpha2": alpha2,
                    "beta2": beta2,
                    "gamma2": gamma2,
                    "fendley_weight": fendley_weight,
                    "cw_low": cw_low,
                    "cw_high": cw_high,
                    "num_points": num_points,
                    "nums_triangles": nums_triangles,
                    "gaps": gaps,
                },
                f,
            )
    else:
        with open(data_file, "r") as f:
            gaps = json.load(f)["gaps"]

    # for w, g in zip(currents_weight, gaps[0]):
    #     print(f"weight={w}, gap={g}")
    #     if w > 10:
    #         break
    # return

    utils.paper_setup()

    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    max_gap = 0
    for num_triangles, gap in zip(nums_triangles, gaps):
        ax.plot(currents_weight, gap, label=f"N = {num_triangles}")
        max_gap = max(max_gap, max(gap))

    ax.set_xlabel(r"Currents weight $a$")
    ax.set_ylabel(r"Gap $\Delta$")
    ax.set_xscale("log")
    ax.set_xlim(currents_weight[0], currents_weight[-1])
    ax.set_ylim(0, max_gap * 1.03)
    handle, label = ax.get_legend_handles_labels()
    ax.legend(handle, label, loc="upper right")
    ax.tick_params(axis="x", pad=10)
    ax.grid()
    plt.subplots_adjust(top=0.95, bottom=0.13, left=0.09, right=0.96)
    plt.savefig(f"output/final/currents_phase.pdf")
