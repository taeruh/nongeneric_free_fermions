from integer_paulis import IntegerPauli, IntegerPauliSum
from integer_hamiltonian import IntegerHamiltonian


class IntegerGenerators:
    def __init__(
        self,
        simplicial_mode: tuple[int, IntegerPauli],
        hamiltonian: IntegerHamiltonian,
        max_search: int | None = None,
    ):
        """
        The hamiltonian is assumed to be simplicial and claw-free, and the simplicial mode
        is assumed to be a simplicial mode of the hamiltonian (it probably still runs fine
        if not, however, the results might be unexpected and it may take forever).


        a potentially good choice for the eta_normalisation_factor is
        len(hamiltonian.operators) / hamiltonian.pauli_l1_norm / (np.sqrt(np.sqrt(2.7)))
        so that we don't run into numerical issues when checking for linear independence
        computing the eigenvalues of the anti_comm_mat_etas (the pauli_l2_norm works not
        so well) (numpy.linalg.(matrix_rank, eigh) go completely batshit when the numbers
        are too large; using numpy.linalg.svd directly) -> instead of using matrix_rank, I
        do a Gram-Schmidt process below with tolerance which is potentially more stable
        than an singular value decomposition with tolerance (which is what matrix_rank
        basically does, I think), however it still goes batshit when we don't normalise
        because the coefficients in the eta vectors just explode without normalisation;
        even with normalisation one has to be careful and it is probably best to always
        print the norms in the Gram-Schmidt process and check that it looks sensible
        """
        self.n = simplicial_mode[1].n
        self.eta_vector_to_pauli_map = [simplicial_mode[1]]
        self.etas: list[IntegerPauliSum] = [
            IntegerPauliSum([(simplicial_mode[0], simplicial_mode[1])])
        ]
        self.eta_vectors = [[simplicial_mode[0]]]
        self.eta_normalisation_factors = [1.0]

        index = 0
        stop_signal = lambda index: max_search is not None and index == max_search
        # just used for an assertion, but it is interesting to note that the number of
        # zero-weight deletions in each eta is quite high, which is probably the magic due
        # to the fact that we are simplicial and claw-free
        self.gram_schmidt_process = GramSchmidtProcess(self.eta_vectors[0])
        self.gram_schmidt_terminated = False
        while True:
            last_eta = self.etas[index]
            eta = IntegerPauliSum([])
            vector = [0] * len(self.eta_vector_to_pauli_map)
            for weight, op in last_eta.ops:
                for ham_weight, ham_op in zip(
                    hamiltonian.weights, hamiltonian.operators
                ):
                    if ham_op.symplectic_inner_product(op) == 1:
                        # cf. paper definition (the 1/2 cancels since we get the product
                        # twice from the commutator)
                        comm_weight = weight * ham_weight
                        comm_op = ham_op.multiply_as_paulis(op)
                        # we give them an extra i to make them hermitian
                        comm_op.add_to_phase(1)

                        already_in_vectors = False
                        for i, vec_op in enumerate(self.eta_vector_to_pauli_map):
                            if comm_op.is_proportional_to(vec_op):
                                sign_phase = comm_op.phase_difference(vec_op)
                                assert sign_phase in [0, 2]
                                vector[i] += comm_weight * (-1) ** (sign_phase // 2)
                                already_in_vectors = True
                                break
                        if not already_in_vectors:
                            self.eta_vector_to_pauli_map.append(comm_op)
                            vector = vector + [comm_weight]
                            self.gram_schmidt_process.append_zeros()
                            # while we don't need to append zeros for what we do in this
                            # method, it will be more convenient when we define the gammas
                            for i in range(len(self.eta_vectors)):
                                self.eta_vectors[i].append(0)

                        # PERF: if already_in_vectors is false, we could do something more
                        # efficient in single_add, as we know that it cannot be in eta
                        eta.single_add(comm_weight, comm_op)

            current_rank = self.gram_schmidt_process.basis.shape[1]
            assert current_rank == index + 1
            if not self.gram_schmidt_process.add_vector(vector):
                # TODO: see the todo below in GramSchmidtProcess.add_vector,
                print(f"Gram-Schmidt process terminated at rank {current_rank}")
                self.gram_schmidt_terminated = True
                break
            # I don't stop the while loop earlier on the stop_signal because I want to
            # know what the norm of the orthogonal component is at the max_search index,
            elif stop_signal(index):
                print(f"Stopped after reaching max_search index of {max_search}")
                break
            else:
                index += 1
                eta.remove_zero_weights()
                self.etas.append(eta)
                self.eta_vectors.append(vector)

        self.num_generators = len(self.etas)

    def init_eta_currents(self):
        """multiplied an "i" in to make them hermitian"""
        self.eta_currents: list[IntegerPauliSum] = []
        for l in range(self.num_generators):
            current = IntegerPauliSum([])
            for k in range(l):
                l_k = l - k
                if k == l_k:
                    continue
                # TODO: Maybe we can do something smarter here (a little bit): it looks
                # like eta[l] and eta[l-k] nearly anticommute if i != j, the
                # anticommutator is most of the time just the identity (up to a factor)
                # (except for some excpetions, I think ...)
                prod = self.etas[k].multiply(self.etas[l_k])
                prod_inv = self.etas[l_k].multiply(self.etas[k])
                commutator = prod.subtract(prod_inv)
                if k % 2 == 1:
                    current = current.subtract(commutator)
                else:
                    current = current.add(commutator)
            current.remove_zero_weights()
            # make them hermitian:
            current.multiply_with_one_imag_unit()
            for _, op in current.ops:
                assert op.get_hermitian_phase() in [0, 2]
            self.eta_currents.append(current)

import numpy as np
class GramSchmidtProcess:
    def __init__(self, first_vector: list[int], tolerance: float = 1e-5):
        norm = np.linalg.norm(first_vector)
        assert norm > 0.0
        self.basis = np.array([np.array(first_vector) / norm], dtype=np.float128).T
        self.tolerance = tolerance
        self.norms = [norm]

    def append_zeros(self):
        self.basis = np.vstack(
            [self.basis, np.zeros((1, self.basis.shape[1]), dtype=np.float128)]
        )

    def add_vector(self, vector: list[int]) -> bool:
        assert len(vector) == self.basis.shape[0]
        projection = self.basis @ (self.basis.T @ vector)
        orthogonal_component = vector - projection
        norm = np.linalg.norm(orthogonal_component)
        # TODO: print this here into some file which one should always check for sensible
        # values
        self.norms.append(norm)
        print(f"Gram-Schmidt: norm of orthogonal component is {norm}")
        if norm > self.tolerance:
            new_basis_vector = orthogonal_component / norm
            self.basis = np.hstack([self.basis, new_basis_vector[:, np.newaxis]])
            return True
        else:
            return False
