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
        alpha: Weight = ConstantWeight(np.float64(1.0)),
        beta: Weight = ConstantWeight(np.float64(1.0)),
        gamma: Weight = ConstantWeight(np.float64(1.0)),
    ):
        self.num_triangles = num_triangles
        self.cap_m = 3 * num_triangles  # number ops
        self.n = self.cap_m + 2  # number qubits (open boundary conditions)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.ops = []
        self.weights = []
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
                self.ops.append(op)
                self.weights.append(parameter())

        self.hamiltonian = Hamiltonian(self.weights, self.ops)

        # not exhaustive
        self.example_simplicial_modes = {
            # { connect to IIXZZ
            "IIIIX": (Pauli.from_indices(self.n, [], [0], 0), 1),
            "IIZZZ": (Pauli.from_indices(self.n, [0, 1, 2], [], 0), 1),
            # } { connect to IIXZZ,
            #                IXZZI
            "IIIYZ": (Pauli.from_indices(self.n, [0, 1], [1], 3), 2),
            "IZZZZ": (Pauli.from_indices(self.n, [0, 1, 2, 3], [], 0), 2),
            # } { connect to IIXZZ,
            #                XZZII
            "IIXXI": (Pauli.from_indices(self.n, [], [1, 2], 0), 2),
            "IZYZI": (Pauli.from_indices(self.n, [1, 2, 3], [2], 3), 2),
            # } { connect to IIXZZ, IXZZI, XZZII
            "ZZZZZ": (Pauli.from_indices(self.n, [0, 1, 2, 3, 4], [], 0), 3),
            "IIYII": (Pauli.from_indices(self.n, [2], [2], 3), 3),
            # }
        }

        self.labels = [f"f{i+1}" for i in range(self.cap_m)]

    def clone(self):
        clone = Fendley(self.num_triangles, self.alpha, self.beta, self.gamma)
        return clone

    def extend_with_currents(
        self,
        currents: list[list[tuple[np.complex128, Pauli]]],
        current_alpha: list[np.float64],
    ):
        assert len(currents) == len(current_alpha)
        self.current_alpha = current_alpha

        for alpha, current in zip(current_alpha, currents):
            for w, op in current:
                already_in = False
                for i, self_op in enumerate(self.ops):
                    if op.is_proportional_to(self_op):
                        phase = op.phase_difference(self_op)
                        self.weights[i] += alpha * w * (1j) ** phase
                        already_in = True
                        break
                if not already_in:
                    self.ops.append(op)
                    self.weights.append(alpha * w)
        to_remove = []
        for i, w in enumerate(self.weights):
            if np.isclose(w, 0):
                to_remove.append(i)
        for i in reversed(to_remove):
            del self.weights[i]
            del self.ops[i]
            if i < self.cap_m:
                del self.labels[i]
        len_labels = len(self.labels)
        for i in range(len_labels, len(self.ops)):
            self.labels.append(f"c_{i - len_labels + 1}")

        self.hamiltonian = Hamiltonian(self.weights, self.ops)
