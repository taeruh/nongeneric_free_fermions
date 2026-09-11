#!/usr/bin/env python

import os
from run import (
    run,
    num_claws,
    num_vertices,
    phase_diagram,
    path_decompositions,
    test_lie_condition,
    test_general_model_for_eta_paths,
)
from run import final


def main():

    os.makedirs("output", exist_ok=True)

    final.currents_phase.run()
    # final.triangle_phase.run()
    # final.number_of_claws.run()
    # final.number_of_vertices.run()
    # final.merged_number_plot.run()

    # test_general_model_for_eta_paths.run()

    # test_lie_condition.run()
    # path_decompositions.run()
    # test()
    # phase_diagram.fendley_phase()
    # phase_diagram.add_currents_phase()
    # phase_diagram.currents_phase()
    # num_claws.run()
    # num_vertices.run()
    # run.run()
    # run.get_phase_diagram()
    # run.test_t()
    # run.currents_plot()
    # run.trying_to_reconstruct_fukai_from_bilinears()
    # import krylov
    # krylov.test_projections()


if __name__ == "__main__":
    main()
