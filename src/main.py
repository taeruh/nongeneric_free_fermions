#!/usr/bin/env python

import os
import run

def main():
    os.makedirs("output", exist_ok=True)
    # run.run()
    run.get_phase_diagram()
    # run.currents_plot()
    # run.trying_to_reconstruct_fukai_from_bilinears()


if __name__ == "__main__":
    main()
