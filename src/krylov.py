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

                        already_in = False
                        for i, (eta_weight, eta_op) in enumerate(eta):
                            if comm_op.is_proportional_to(eta_op):
                                sign_phase = comm_op.multiply_as_paulis(eta_op).phase
                                assert sign_phase in [0, 2]
                                eta[i] = (
                                    eta_weight
                                    + comm_weight * (-1) ** (sign_phase // 2),
                                    eta_op,
                                )
                                already_in = True
                                break
                        if not already_in:
                            eta.append((comm_weight, comm_op))

                            already_in = False
                            for i, vec_op in enumerate(self.vector_to_pauli_map):
                                if comm_op.is_proportional_to(vec_op):
                                    sign_phase = comm_op.multiply_as_paulis(
                                        vec_op
                                    ).phase
                                    assert sign_phase in [0, 2]
                                    vector[i] += comm_weight * (-1) ** (sign_phase // 2)
                                    already_in = True
                                    break
                            if not already_in:
                                self.vector_to_pauli_map.append(comm_op)
                                vector = np.append(vector, comm_weight)
                                for i, v in enumerate(self.vectors):
                                    self.vectors[i] = np.append(v, 0.0)

            assert linalg.matrix_rank(self.vectors) == index + 1
            self.vectors.append(vector)
            rank = linalg.matrix_rank(self.vectors)
            if rank == index + 1:
                self.vectors.pop()
                break
            else:
                self.etas.append(eta)
                index += 1

        print(len(self.etas))
        # for eta in self.etas:
        #     # print([(w, p.to_string()) for w, p in eta])
        #     print([p.to_string() for _, op in eta])


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
