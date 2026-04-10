import numpy as np

from hamiltonian import Hamiltonian
from paulis import Pauli
from .weights import Weight, ConstantWeight


# ZZX representation with open boundery conditions (https://arxiv.org/pdf/1901.08078
# section 2.1; https://arxiv.org/pdf/2508.05789 equation 2.5)
class Fendley:
    def __init__(
        self,
        num_triangles: int,
        alpha: Weight = ConstantWeight(1.0),
        beta: Weight = ConstantWeight(1.0),
        gamma: Weight = ConstantWeight(1.0),
    ):
        self.num_triangles = num_triangles
        self.cap_m = 3 * num_triangles  # number ops
        self.n = self.cap_m + 2  # number qubits (open boundary conditions)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        ops = []
        weights = []
        for i in range(self.num_triangles):
            for j, parameter in enumerate([self.alpha, self.beta, self.gamma]):
                op = Pauli(
                    self.n,
                    np.zeros(self.n, dtype=bool),
                    np.zeros(self.n, dtype=bool),
                    0,
                )
                op.z[3 * i + j] = True
                op.z[3 * i + (j + 1)] = True
                op.x[3 * i + (j + 2)] = True
                ops.append(op)
                weights.append(parameter())

        self.hamiltonian = Hamiltonian(weights, ops)

        self.example_simplicial_mode = Pauli(
            self.n,
            np.zeros(self.n, dtype=bool),
            np.zeros(self.n, dtype=bool),
            0,
        )
        self.example_simplicial_mode.x[0] = True

        self.labels = [f"f{i+1}" for i in range(self.cap_m)]
