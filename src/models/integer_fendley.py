from integer_hamiltonian import IntegerHamiltonian
from integer_paulis import IntegerPauli, IntegerPauliSum


# ZZX representation with open boundery conditions (https://arxiv.org/pdf/1901.08078
# section 2.1; https://arxiv.org/pdf/2508.05789 equation 2.5)
class IntegerFendley:
    def __init__(
        self,
        num_triangles: int,
        alpha: int = 1,
        beta: int = 1,
        gamma: int = 1,
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
            for j, weight in enumerate([self.alpha, self.beta, self.gamma]):
                if weight != 0:
                    op = IntegerPauli.from_indices(
                        self.n,
                        [3 * i + j, 3 * i + (j + 1)],
                        [3 * i + (j + 2)],
                        0,
                    )
                    self.ops.append(op)
                    self.weights.append(weight)

        # self.weights = self.weights[: -2]
        # self.ops = self.ops[: -2]
        self.hamiltonian = IntegerHamiltonian(self.weights, self.ops)

        # not exhaustive
        self.example_simplicial_modes = {
            # { connect to IIXZZ
            "IIIIX": (IntegerPauli.from_indices(self.n, [], [0], 0), 1),
            "IIZZZ": (IntegerPauli.from_indices(self.n, [0, 1, 2], [], 0), 1),
            # } { connect to IIXZZ,
            #                IXZZI
            "IIIYZ": (IntegerPauli.from_indices(self.n, [0, 1], [1], 3), 2),
            "IZZZZ": (IntegerPauli.from_indices(self.n, [0, 1, 2, 3], [], 0), 2),
            # } { connect to IIXZZ, IXZZI, XZZII
            "IIYII": (IntegerPauli.from_indices(self.n, [2], [2], 3), 3),
            "ZZZZZ": (IntegerPauli.from_indices(self.n, [0, 1, 2, 3, 4], [], 0), 3),
            # }
        }

        self.labels = [f"f{i+1}" for i in range(self.cap_m)]

    def clone(self):
        clone = IntegerFendley(self.num_triangles, self.alpha, self.beta, self.gamma)
        return clone

    def extend_with_currents(
        self,
        currents: list[IntegerPauliSum],
        current_alpha: list[int],
    ):
        assert len(currents) == len(current_alpha)
        self.current_alpha = current_alpha

        for alpha, current in zip(current_alpha, currents):
            for w, op in current.ops:
                already_in = False
                for i, self_op in enumerate(self.ops):
                    if op.is_proportional_to(self_op):
                        phase = op.phase_difference(self_op)
                        assert phase in [0, 2]
                        self.weights[i] += alpha * w * (-1) ** (phase // 2)
                        already_in = True
                        break
                if not already_in:
                    self.ops.append(op)
                    self.weights.append(alpha * w)
        to_remove = []
        for i, w in enumerate(self.weights):
            if w == 0:
                to_remove.append(i)
        for i in reversed(to_remove):
            del self.weights[i]
            del self.ops[i]
            if i < self.cap_m:
                del self.labels[i]
        len_labels = len(self.labels)
        for i in range(len_labels, len(self.ops)):
            self.labels.append(f"c_{i - len_labels + 1}")

        self.hamiltonian = IntegerHamiltonian(self.weights, self.ops)
