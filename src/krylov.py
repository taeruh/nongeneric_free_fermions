import numpy as np
from numpy import linalg

from hamiltonian import Hamiltonian
from paulis import Pauli


class Generators:
    def __init__(self, simplicial_mode: Pauli, hamiltonian: Hamiltonian):
        etas = [[(1.0, simplicial_mode)]]
        vectors = [np.array([1.0])]
        vector_to_pauli_map = [simplicial_mode]

        for i in range(10):
            last_eta = etas[i]
            eta = []
            vector = np.zeros(len(vector_to_pauli_map))
            for weight, op in last_eta:
                for ham_weight, ham_op in zip(
                    hamiltonian.weights, hamiltonian.operators
                ):
                    if ham_op.symplectic_inner_product(op) == 1:
                        comm_weight = weight * ham_weight / 2
                        comm_op = ham_op.multiply_as_paulis(op)
                        comm_op.phase = (op.phase + 1) % 4

                        already_in = False
                        for i, (eta_weight, eta_op) in enumerate(eta):
                            if op.is_proportional_to(eta_op):
                                sign_phase = op.multiply_as_paulis(eta_op).phase
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
                        for i, vec_op in enumerate(vector_to_pauli_map):
                            if comm_op.is_proportional_to(vec_op):
                                sign_phase = comm_op.multiply_as_paulis(vec_op).phase
                                assert sign_phase in [0, 2]
                                vector[i] += comm_weight * (-1) ** (sign_phase // 2)
                                already_in = True
                                break
                        if not already_in:
                            vector_to_pauli_map.append(comm_op)
                            vector = np.append(vector, comm_weight)
                            for i, v in enumerate(vectors):
                                vectors[i] = np.append(v, 0.0)

            vectors.append(vector)
            etas.append(eta)

        for vec in vectors:
            print(vec)

        print(linalg.matrix_rank(vectors[:-1]))
        print(linalg.matrix_rank(vectors))

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

