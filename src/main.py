#!/usr/bin/env python

import os
from run import run, num_claws, num_vertices, phase_diagram

def main():
    os.makedirs("output", exist_ok=True)

    phase_diagram.run()
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
