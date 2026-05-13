import os
os.makedirs("output/final", exist_ok=True)
os.makedirs("output/final/data", exist_ok=True)

from . import currents_phase, triangle_phase, number_of_claws, number_of_vertices
