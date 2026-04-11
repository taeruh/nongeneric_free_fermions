import numpy as np
from numpy import linalg

from hamiltonian import Hamiltonian
from paulis import Pauli


class Generators:
    def __init__(
        self,
        simplicial_mode: Pauli,
        hamiltonian: Hamiltonian,
        max_search: int | None = None,
    ):
        self.etas = [[(1.0, simplicial_mode)]]
        self.vectors = [np.array([1.0])]
        self.vector_to_pauli_map = [simplicial_mode]

        index = 0
        stop_signal = lambda index: max_search is not None and index >= max_search
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
                                assert (
                                    already_in_vectors
                                ), "if it's already in etas, then it also must be already in vectors"
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
                for i in reversed(to_delete):
                    eta.pop(i)
                self.etas.append(eta)
                index += 1

        total_num_op_in_etas = sum(len(eta) for eta in self.etas)
        assert total_num_op_in_etas >= len(
            self.vector_to_pauli_map
        ), "not necessarily a bug, but if that doesn't hold, then there are some zero-weight operators in the (probobly last) etas, which can be removed"

        self.num_etas = len(self.etas)

        anti_comm_mat = np.zeros((self.num_etas, self.num_etas))
        for i in range(self.num_etas):
            for j in range(i, self.num_etas):
                total_trace = 0
                for weight_i, op_i in self.etas[i]:
                    for weight_j, op_j in self.etas[j]:
                        prod = op_i.multiply_as_paulis(op_j)
                        if prod.is_proportional_to(Pauli.identity(op_i.n)):
                            assert prod.phase in [0, 2]
                            total_trace += (
                                weight_i * weight_j * (-1) ** (prod.phase // 2)
                            )
                anti_comm_mat[i, j] = total_trace
                anti_comm_mat[j, i] = total_trace

        eigvals, eigvecs = linalg.eigh(anti_comm_mat)
        print(eigvals)
        for val in eigvals:
            assert not np.isclose(val, 0.0)
            assert val > 0.0

        # self.gammas = []
        # for i in range(self.num_etas):
        #     factor = eigvals


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
