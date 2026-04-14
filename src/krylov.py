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
        eta_normalisation_factor: np.float64 = np.float64(1.0),
        orthogonal_tolerance: float = 1e-9,
        max_search: int | None = None,
    ):
        """
        The hamiltonian is assumed to be simplicial and claw-free, and the simplicial mode
        is assumed to be a simplicial mode of the hamiltonian (it probably still runs fine
        if not, however, the results might be unexpected and it may take forever).


        a potentially good choice for the eta_normalisation_factor is
        len(hamiltonian.operators) / hamiltonian.pauli_l1_norm
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
        self.n = simplicial_mode.n
        self.etas: list[list[tuple[np.complex128, Pauli]]] = [
            [(np.complex128(1.0), simplicial_mode)]
        ]
        self.eta_vectors = [np.array([1.0])]
        self.eta_vector_to_pauli_map = [simplicial_mode]

        index = 0
        stop_signal = lambda index: max_search is not None and index >= max_search
        # just used for an assertion, but it is interesting to note that the number of
        # zero-weight deletions in each eta is quite high, which is probably the magic due
        # to the fact that we are simplicial and claw-free
        total_num_zero_weight_deletions = 0
        gram_schmidt_process = GramSchmidtProcess(
            self.eta_vectors[0], tolerance=orthogonal_tolerance
        )
        while not stop_signal(index):
            last_eta = self.etas[index]
            eta = []
            vector = np.zeros(len(self.eta_vector_to_pauli_map), dtype=complex)
            for weight, op in last_eta:
                for ham_weight, ham_op in zip(
                    hamiltonian.weights, hamiltonian.operators
                ):
                    if ham_op.symplectic_inner_product(op) == 1:
                        # cf. paper definition (the 1/2 cancels since we get the product
                        # twice from the commutator)
                        # comm_weight = weight * ham_weight / hamiltonian.pauli_l2_norm
                        comm_weight = weight * ham_weight * eta_normalisation_factor
                        comm_op = ham_op.multiply_as_paulis(op)
                        # we give them an extra i to make them hermitian
                        comm_op.phase = (comm_op.phase + 1) % 4

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
                            vector = np.append(vector, comm_weight)
                            # for i, v in enumerate(self.eta_vectors):
                            #     self.eta_vectors[i] = np.append(v, 0.0)
                            gram_schmidt_process.append_zeros()

                        already_in_etas = False
                        for i, (eta_weight, eta_op) in enumerate(eta):
                            if comm_op.is_proportional_to(eta_op):
                                sign_phase = comm_op.phase_difference(eta_op)
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

                        # PERF: we can actually skip the already_in_etas check is
                        # already_in_vectors is false, but for now just leave it there and
                        # do this assert here to catch bugs
                        if not already_in_vectors:
                            assert not already_in_etas, (
                                "if it's not already in vectors, "
                                "then it cannot be already in etas"
                            )

            current_rank = gram_schmidt_process.basis.shape[1]
            assert current_rank == index + 1
            if not gram_schmidt_process.add_vector(vector):
                # TODO: see the todo below in GramSchmidtProcess.add_vector,
                print(f"Gram-Schmidt process terminated at rank {current_rank}")
                break
            else:
                to_delete = []
                for i, (eta_weight, eta_op) in enumerate(eta):
                    if np.isclose(eta_weight, 0):
                        to_delete.append(i)
                total_num_zero_weight_deletions += len(to_delete)
                # reversed because I think python shifts from right to left when deleting
                # inidices in a list, but I might be wrong
                for i in reversed(to_delete):
                    eta.pop(i)
                self.etas.append(eta)
                index += 1

        total_num_op_in_etas = sum(len(eta) for eta in self.etas)
        assert total_num_op_in_etas + total_num_zero_weight_deletions >= len(
            self.eta_vector_to_pauli_map
        ), (
            "not necessarily a bug, but if that doesn't hold, then there are some ",
            "zero-weight operators in the (probobly last) etas, which can be removed",
        )

        self.num_generators = len(self.etas)

    def init_gammas(self):
        anti_comm_mat_etas = np.zeros(
            (self.num_generators, self.num_generators), dtype=complex
        )
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

        print(anti_comm_mat_etas)
        eigvals, eigvecs = linalg.eigh(anti_comm_mat_etas)
        print(eigvals)
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
            factor = np.complex128((2 / eigvals[i]) ** (0.5))
            # factor = (1j) ** (i % 2) / ( eigvals[i] ** (0.5))
            gamma_vector = np.zeros(len(self.eta_vector_to_pauli_map), dtype=complex)
            for j in range(self.num_generators):
                gamma_vector += factor * eigvecs[j, i] * self.eta_vectors[j]
            gamma = []
            for weight, op in zip(gamma_vector, self.eta_vector_to_pauli_map):
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

    def init_gamma_bilinears(self):
        # PERF: this loop takes quite some time
        self.gamma_bilinears: dict[
            tuple[int, int], list[tuple[np.complex128, Pauli]]
        ] = dict()
        for i in range(self.num_generators):
            for j in range(i, self.num_generators):
                product = paulis.list_multiplication(self.gammas[i], self.gammas[j])
                for w, op in product:
                    assert w.imag == 0.0
                    if i != j:
                        # need to make them hermitian
                        op.phase = (op.phase + 1) % 4
                    assert op.get_hermitian_phase() in [0, 2]
                self.gamma_bilinears[(i, j)] = product

    def bilinear_gamma_projection(self, pauli: Pauli) -> list[tuple[int, int, float]]:
        """given a pauli, return the coefficients of its projection onto the gammas"""
        coeffs = []
        for (i, j), ops in self.gamma_bilinears.items():
            coeff = paulis.list_and_single_hilbert_schmidt_inner_product(ops, pauli)
            if coeff != 0.0:
                coeffs.append((i, j, coeff))
        return coeffs

    def init_eta_currents(self):
        self.eta_currents: list[list[tuple[np.complex128, Pauli]]] = []
        for l in range(self.num_generators):
            current = []
            for k in range(l):
                l_k = l - k
                if k == l_k:
                    continue
                # TODO: Maybe we can do something smarter here (a little bit): it looks
                # like eta[l] and eta[l-k] nearly anticommute if i != j, the
                # anticommutator is most of the time just the identity (up to a factor)
                # (except for some excpetions, I think ...)
                prod = paulis.list_multiplication(self.etas[k], self.etas[l_k])
                prod_inv = paulis.list_multiplication(self.etas[l_k], self.etas[k])
                commutator = paulis.list_addition(
                    prod, [(-w, op) for w, op in prod_inv]
                )
                if k % 2 == 1:
                    for i, (w, op) in enumerate(commutator):
                        commutator[i] = (-w, op)
                current = paulis.list_addition(current, commutator)
            to_delete = []
            for i, (weight, op) in enumerate(current):
                if abs(weight) == 0.0:
                    to_delete.append(i)
            for i in reversed(to_delete):
                current.pop(i)
            # make them hermitian:
            for i, (weight, op) in enumerate(current):
                op.phase = (op.phase + 1) % 4
                assert op.get_hermitian_phase() in [0, 2]
                current[i] = (weight, op)
            self.eta_currents.append(current)


class GramSchmidtProcess:
    def __init__(self, first_vector: np.ndarray, tolerance: float = 1e-9):
        norm = linalg.norm(first_vector)
        assert norm > 0.0
        self.basis = np.array([first_vector / norm], dtype=complex).T
        self.tolerance = tolerance

    def append_zeros(self):
        self.basis = np.vstack(
            [self.basis, np.zeros((1, self.basis.shape[1]), dtype=complex)]
        )

    def add_vector(self, vector: np.ndarray) -> bool:
        # print(vector)
        assert len(vector) == self.basis.shape[0]
        projection = self.basis @ (self.basis.conj().T @ vector)
        orthogonal_component = vector - projection
        norm = linalg.norm(orthogonal_component)
        # TODO: print this here into some file which one should always check for sensible
        # values
        print(f"Gram-Schmidt: norm of orthogonal component is {norm}")
        if norm > self.tolerance:
            new_basis_vector = orthogonal_component / norm
            self.basis = np.hstack([self.basis, new_basis_vector[:, np.newaxis]])
            return True
        else:
            return False
