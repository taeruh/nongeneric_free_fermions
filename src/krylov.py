import numpy as np
from numpy import linalg

from hamiltonian import Hamiltonian
import paulis
from paulis import Pauli


class Generators:
    def __init__(
        self,
        simplicial_mode: Pauli,
        hamiltonian: Hamiltonian,
        max_search: int | None = None,
    ):
        """
        The hamiltonian is assumed to be simplicial and claw-free, and the simplicial mode
        is assumed to be a simplicial mode of the hamiltonian (it probably still runs fine
        if not, however, the results might be unexpected and it may take forever).
        """
        self.n = simplicial_mode.n
        self.etas = [[(1.0, simplicial_mode)]]
        self.vectors = [np.array([1.0])]
        self.vector_to_pauli_map = [simplicial_mode]

        index = 0
        stop_signal = lambda index: max_search is not None and index >= max_search
        # just used for an assertion, but it is interesting to note that the number of
        # zero-weight deletions in each eta is quite high, which is probably the magic due
        # to the fact that we are simplicial and claw-free
        total_num_zero_weight_deletions = 0
        while not stop_signal(index):
            last_eta = self.etas[index]
            eta = []
            vector = np.zeros(len(self.vector_to_pauli_map))
            for weight, op in last_eta:
                for ham_weight, ham_op in zip(
                    hamiltonian.weights, hamiltonian.operators
                ):
                    if ham_op.symplectic_inner_product(op) == 1:
                        comm_weight = weight * ham_weight / 2
                        comm_op = ham_op.multiply_as_paulis(op)
                        comm_op.phase = (comm_op.phase + 1) % 4

                        already_in_vectors = False
                        for i, vec_op in enumerate(self.vector_to_pauli_map):
                            if comm_op.is_proportional_to(vec_op):
                                sign_phase = comm_op.multiply_as_paulis(vec_op).phase
                                assert sign_phase in [0, 2]
                                vector[i] += comm_weight * (-1) ** (sign_phase // 2)
                                already_in_vectors = True
                                break
                        if not already_in_vectors:
                            self.vector_to_pauli_map.append(comm_op)
                            vector = np.append(vector, comm_weight)
                            for i, v in enumerate(self.vectors):
                                self.vectors[i] = np.append(v, 0.0)

                        already_in_etas = False
                        for i, (eta_weight, eta_op) in enumerate(eta):
                            if comm_op.is_proportional_to(eta_op):
                                sign_phase = comm_op.multiply_as_paulis(eta_op).phase
                                assert sign_phase in [0, 2]
                                eta[i] = (
                                    eta_weight
                                    + comm_weight * (-1) ** (sign_phase // 2),
                                    eta_op,
                                )
                                already_in_etas = True
                                assert already_in_vectors, (
                                    "if it's already in etas, "
                                    "then it also must be already in vectors"
                                )
                                break
                        if not already_in_etas:
                            eta.append((comm_weight, comm_op))

            assert linalg.matrix_rank(self.vectors) == index + 1
            self.vectors.append(vector)
            rank = linalg.matrix_rank(self.vectors)
            if rank == index + 1:
                self.vectors.pop()
                break
            else:
                to_delete = []
                for i, (eta_weight, eta_op) in enumerate(eta):
                    if abs(eta_weight) == 0:
                        to_delete.append(i)
                # reversed because I think python shifts from right to left when deleting
                # inidices in a list, but I might be wrong
                total_num_zero_weight_deletions += len(to_delete)
                for i in reversed(to_delete):
                    eta.pop(i)
                self.etas.append(eta)
                index += 1

        total_num_op_in_etas = sum(len(eta) for eta in self.etas)
        assert total_num_op_in_etas + total_num_zero_weight_deletions >= len(
            self.vector_to_pauli_map
        ), (
            "not necessarily a bug, but if that doesn't hold, then there are some ",
            "zero-weight operators in the (probobly last) etas, which can be removed",
        )

        self.num_generators = len(self.etas)

        anti_comm_mat_etas = np.zeros((self.num_generators, self.num_generators))
        for i in range(self.num_generators):
            for j in range(i, self.num_generators):
                total_trace = 0  # implicitly divided by dim(hilbert space)
                for weight_i, op_i in self.etas[i]:
                    for weight_j, op_j in self.etas[j]:
                        prod = op_i.multiply_as_paulis(op_j)
                        if prod.is_proportional_to(Pauli.identity(op_i.n)):
                            assert prod.phase in [0, 2]
                            total_trace += (
                                2 * weight_i * weight_j * (-1) ** (prod.phase // 2)
                            )
                anti_comm_mat_etas[i, j] = total_trace
                anti_comm_mat_etas[j, i] = total_trace

        eigvals, eigvecs = linalg.eigh(anti_comm_mat_etas)
        for val in eigvals:
            assert not np.isclose(val, 0.0)
            assert val > 0.0

        self.gammas = []
        for i in range(self.num_generators):
            # a little bit different to eq. 129 in chapman_unified as we already have the
            # "i" factor in the etas and we define the anti_comm_mat_etas with a different
            # factor (only divided by dim(hilbert space) instead of 2 * dim(hilbert space)
            # TODO:  double check on that these two statements; I'm just guessing here and
            # set the factor so that the gammas are properly normalised
            factor = (2 / eigvals[i]) ** (0.5)
            # factor = (1j) ** (i % 2) / ( eigvals[i] ** (0.5))
            gamma_vector = np.zeros(len(self.vector_to_pauli_map))
            for j in range(self.num_generators):
                gamma_vector += factor * eigvecs[j, i] * self.vectors[j]
            gamma = []
            for weight, op in zip(gamma_vector, self.vector_to_pauli_map):
                if abs(weight) > 1e-10:
                    gamma.append((weight, op))
            self.gammas.append(gamma)

        # exhaustively check that the gammas behave correctly {{{
        anti_comm_mat_gammas = np.zeros(
            (self.num_generators, self.num_generators), dtype=complex
        )
        for i in range(self.num_generators):
            for j in range(self.num_generators):
                total_trace = 0
                for weight_i, op_i in self.gammas[i]:
                    for weight_j, op_j in self.gammas[j]:
                        prod = op_i.multiply_as_paulis(op_j)
                        if prod.is_proportional_to(Pauli.identity(op_i.n)):
                            assert prod.phase in [0, 2]
                            total_trace += (
                                2 * weight_i * weight_j * (-1) ** (prod.phase // 2)
                            )
                anti_comm_mat_gammas[i, j] = total_trace

        assert np.allclose(anti_comm_mat_gammas, 2 * np.identity(self.num_generators))

        if self.n <= 8:  # otherwise this is too expensive
            for i in range(self.num_generators):
                for j in range(self.num_generators):
                    prod = paulis.list_to_matrix(
                        self.gammas[i]
                    ) @ paulis.list_to_matrix(self.gammas[j])
                    prod_inverse = paulis.list_to_matrix(
                        self.gammas[j]
                    ) @ paulis.list_to_matrix(self.gammas[i])
                    if i == j:
                        assert np.allclose(prod, np.identity(2**self.n))
                    else:
                        assert np.allclose(prod, -prod_inverse)
        # }}}

# old notes, maybe useful later:

# row_length = len(vector_to_pauli_map)
# a = np.array(vectors[:-1])
# for _ in range(row_length - (len(vectors) - 1)):
#     a = np.vstack([a, np.zeros(row_length)])
# a = a.T
# print(a)
# b = vectors[-1]
# print(b)

# solution = np.linalg.solve(a, b)
# print("Solution:", solution)

# for eta in etas:
#     for weight, op in eta:
#         print(f"{weight:.4f} {op.to_string()}")
#     print()
