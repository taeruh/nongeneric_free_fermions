import numpy as np

from hamiltonian import Hamiltonian
from paulis import Pauli
from .weights import Weight, ConstantWeight


# ZZX representation with open boundery conditions (https://arxiv.org/pdf/2508.05789
# equation 2.5); extending fendley.Fendley
class Fukai:
    def __init__(
        self,
        num_triangles: int,
        fendley_alpha: Weight = ConstantWeight(1.0),
        fendley_beta: Weight = ConstantWeight(1.0),
        fendley_gamma: Weight = ConstantWeight(1.0),
        beta_3: Weight = ConstantWeight(1.0),
        beta_5: Weight = ConstantWeight(1.0),
    ):
        """
        notes:
        - beta_3 and beta_5 are not completely arbitrary as certain choices causes
          zero-divisions, e.g., if fendley_(alpha,beta,gamma) = 1, then beta_3 = 0 and
          beta_5 = 1 causes a zero division when calculating beta_7
        - setting beta_3 to zero does not cause beta_5 to be zero, but setting beta_5 to
          zero causes all following betas to be zero
        """
        self.num_triangles = num_triangles
        self.cap_m = 3 * num_triangles
        self.cap_m_2 = self.cap_m // 2
        self.n = self.cap_m + 2
        self.fendley_alpha = fendley_alpha
        self.fendley_beta = fendley_beta
        self.fendley_gamma = fendley_gamma
        self.beta_3 = beta_3()
        self.beta_5 = beta_5()

        # note that when using these fendley_ lists indexed with the m from the extra
        # fukai terms we have to subtract an additional -1 because in the paper it starts
        # counting at 1 and goes to cap_m (but the lists start at 0 and go to cap_m-1)
        self.fendley_ops: list[Pauli] = []
        self.fendley_weights = []

        for m in range(self.num_triangles):
            for j, parameter in enumerate(
                [self.fendley_alpha, self.fendley_beta, self.fendley_gamma]
            ):
                op = Pauli(
                    self.n,
                    np.zeros(self.n, dtype=bool),
                    np.zeros(self.n, dtype=bool),
                    0,
                )
                op.z[3 * m + j] = True
                op.z[3 * m + (j + 1)] = True
                op.x[3 * m + (j + 2)] = True
                self.fendley_ops.append(op)
                self.fendley_weights.append(parameter())

        weights = self.fendley_weights.copy()
        ops = self.fendley_ops.copy()

        # the first two are beta_-1 and beta_1, which are never used and not defined in
        # the paper, but I put them in here to make the indexing more consistent and
        # easier to read
        betas = [0, 0, self.beta_3, self.beta_5]

        self.fukai_ops: list[Pauli] = []
        self.fukai_weights: list[float] = []

        if self.cap_m_2 < 4:
            first_iter_range = range(2, self.cap_m_2 + 1)
            second_iter_range = None
        else:
            first_iter_range = range(2, 3 + 1)
            second_iter_range = range(4, self.cap_m_2 + 1)
        for m in first_iter_range:
            self.fukai_ops.append(next_pauli(m, self.fendley_ops))
            self.fukai_weights.append(betas[m])
        if second_iter_range is not None:
            for m in second_iter_range:
                beta = next_beta(m, betas, self.fendley_weights)
                betas.append(beta)
                self.fukai_ops.append(next_pauli(m, self.fendley_ops))
                self.fukai_weights.append(beta)

        weights.extend(self.fukai_weights)
        ops.extend(self.fukai_ops)

        self.hamiltonian = Hamiltonian(weights, ops)

        self.labels = [f"f{i+1}" for i in range(self.cap_m)]
        for m in range(2, self.cap_m_2 + 1):
            self.labels.append(f"k{2*m-1}")


def next_beta(m: int, betas: list[float], fendley_weights: list[float]) -> float:
    assert m > 1
    return betas[m - 1] / (
        fendley_weights[2 * m - 4 - 1] ** 2 * betas[m - 2]
        - fendley_weights[2 * m - 1] ** 2 * betas[m - 1]
        + 1
    )


def next_pauli(m: int, fendley_ops: list[Pauli]) -> Pauli:
    return (
        fendley_ops[2 * m - 2 - 1]
        .multiply_as_paulis(fendley_ops[2 * m - 3 - 1])
        .multiply_as_paulis(fendley_ops[2 * m - 1])
    )
