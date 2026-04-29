import numpy as np
from numpy import linalg
import scipy

from hamiltonian import Hamiltonian
from rust_backend.paulis import Pauli, PauliSum
import paulis


class Generators:
    def __init__(
        self,
        simplicial_mode: tuple[np.float64 | float, Pauli],
        hamiltonian: Hamiltonian,
        renormalise: bool = False,
        eta_normalisation_factor: np.float64 = np.float64(1.0),
        orthogonal_tolerance: float = 1e-10,
        max_search_eta_index: int | None = None,
    ):
        # TODO: check that we undo the renormalisation when required (cf. below in
        # """...""" (e.g., when caculating the currents); in general it probably has to be
        # undone whenever we have a some of etas (or products of etas) with weights that
        # are fixed by some definition
        """
        The hamiltonian is assumed to be simplicial and claw-free, and the simplicial mode
        is assumed to be a simplicial mode of the hamiltonian (it probably still runs fine
        if not, however, the results might be unexpected and it may take forever).

        `renormalise` might help against some numerical issues, however, note that this
        causes a non-constant renormalisation (while eta_normalisation_factor is constant
        in the sense that eta_k is normalised with eta_normalisation_factor^k); this
        normalisation must be undone occasionally, e.g., when calculating the currents
        (eta_normalisation_factor is fine, as it just causes current_l to be renormalised
        with eta_normalisation_factor^l, which can be captured in the currents_alpha).

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
        self.n = simplicial_mode[1].n()
        self.eta_vector_to_pauli_map = [simplicial_mode[1]]
        if renormalise:
            self.etas: list[PauliSum] = [
                PauliSum([(np.float64(1.0), simplicial_mode[1])])
            ]
            self.eta_vectors = [np.array([1.0])]
            self.eta_renormalisation_factor = [1.0 / simplicial_mode[0]]
        else:
            self.etas: list[PauliSum] = [
                PauliSum([(np.float64(simplicial_mode[0]), simplicial_mode[1])])
            ]
            self.eta_vectors = [np.array([simplicial_mode[0]])]
            self.eta_renormalisation_factor = None
        self.eta_normalisation_factors = [1.0]

        index = 0
        stop_signal = (
            lambda index: max_search_eta_index is not None
            and index == max_search_eta_index
        )
        # just used for an assertion, but it is interesting to note that the number of
        # zero-weight deletions in each eta is quite high, which is probably the magic due
        # to the fact that we are simplicial and claw-free
        total_num_zero_weight_deletions = 0
        self.gram_schmidt_process = GramSchmidtProcess(
            # self.gram_schmidt_process = MpmathGramSchmidtProcess(
            self.eta_vectors[0],
            tolerance=orthogonal_tolerance,
        )
        self.gram_schmidt_terminated = False
        while True:
            last_eta = self.etas[index]
            eta = PauliSum([])
            vector = np.zeros(len(self.eta_vector_to_pauli_map), dtype=np.float64)
            for weight, op in last_eta.to_py_list():
                for ham_weight, ham_op in zip(
                    hamiltonian.weights, hamiltonian.operators
                ):
                    if ham_op.symplectic_inner_product(op):
                        # cf. paper definition (the 1/2 cancels since we get the product
                        # twice from the commutator)
                        comm_weight = weight * ham_weight * eta_normalisation_factor
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
                            vector = np.append(vector, comm_weight)
                            self.gram_schmidt_process.append_zeros()
                            # while we don't need to append zeros for what we do in this
                            # method, it will be more convenient when we define the gammas
                            for i, v in enumerate(self.eta_vectors):
                                self.eta_vectors[i] = np.append(v, 0.0)

                        # PERF: if already_in_vectors is false, we could do something more
                        # efficient in single_add, as we know that it cannot be in eta
                        eta.single_add(comm_weight, comm_op)

            current_rank = self.gram_schmidt_process.num_vectors()
            assert current_rank == index + 1
            if not self.gram_schmidt_process.add_vector(vector):
                # TODO: see the todo below in GramSchmidtProcess.add_vector,
                print(f"Gram-Schmidt process terminated at rank {current_rank}")
                self.gram_schmidt_terminated = True
                break
            # I don't stop the while loop earlier on the stop_signal because I want to
            # know what the norm of the orthogonal component is at the max_search index,
            elif stop_signal(index):
                print(
                    f"Stopped after reaching max_search index of {max_search_eta_index}"
                )
                break
            else:
                index += 1
                eta.remove_zero_weights()
                if renormalise:
                    norm = 0
                    num_ops = 0
                    for weight, _ in eta.to_py_list():
                        num_ops += 1
                        norm += weight**2
                    norm = np.sqrt(norm) / num_ops
                    eta.multiply_with_float(1.0 / norm)
                    vector = vector / norm
                    self.eta_renormalisation_factor.append(  # pyright: ignore
                        self.eta_renormalisation_factor[-1] / norm  # pyright: ignore
                    )
                self.etas.append(eta)
                self.eta_vectors.append(vector)
                self.eta_normalisation_factors.append(
                    self.eta_normalisation_factors[-1] * eta_normalisation_factor
                )

        # total_num_op_in_etas = sum(eta.len() for eta in self.etas)
        # assert total_num_op_in_etas + total_num_zero_weight_deletions >= len(
        #     self.eta_vector_to_pauli_map
        # ), (
        #     "not necessarily a bug, but if that doesn't hold, then there are some ",
        #     "zero-weight operators in the (probobly last) etas, which can be removed",
        # )

        self.num_generators = len(self.etas)

    def init_gammas(self, do_checks: bool = True, do_eigval_zero_check: bool = True):
        """
        do_eigval_zero_check is not correct anymore when we have many triangles (about 5
        and more; might also depend on alpha, beta, gamma) because then we actually get
        some very small eigenvalues
        """

        import mpmath as mp

        mp.mp.dps = 500

        anti_comm_mat_etas = np.zeros(
            (self.num_generators, self.num_generators), dtype=np.float64
        )
        # anti_comm_mat_etas = mp.matrix(np.zeros(
        #     (self.num_generators, self.num_generators), dtype=np.float64
        # ).tolist())
        for i in range(self.num_generators):
            for j in range(i, self.num_generators):
                total_trace = 0  # implicitly divided by dim(hilbert space)
                for weight_i, op_i in self.etas[i].to_py_list():
                    for weight_j, op_j in self.etas[j].to_py_list():
                        prod = op_i.multiply_as_paulis(op_j)
                        if prod.is_proportional_to(Pauli.identity(op_i.n())):
                            assert prod.phase() in [0, 2]
                            total_trace += (
                                2 * weight_i * weight_j * (-1) ** (prod.phase() // 2)
                            )
                anti_comm_mat_etas[i, j] = total_trace
                anti_comm_mat_etas[j, i] = total_trace
                # print(anti_comm_mat_etas[i, j], "anti-commutator of eta", i, "and eta", j)

        anti_comm_mat_etas /= 2  # per definition

        scale = linalg.norm(anti_comm_mat_etas)
        print(scale, "scale")
        print(linalg.cond(anti_comm_mat_etas))
        anti_comm_mat_etas = anti_comm_mat_etas / scale
        print(linalg.cond(anti_comm_mat_etas))

        # print(anti_comm_mat_etas.shape)
        # eigvals, eigvecs = linalg.eigh(anti_comm_mat_etas)
        eigvals, eigvecs = scipy.linalg.eigh(anti_comm_mat_etas, driver="evr")
        # print(anti_comm_mat_etas)

        # anti_comm_mat_etas = mp.matrix(anti_comm_mat_etas.tolist())
        # eigvals_mp, eigvecs_mp = mp.eig(anti_comm_mat_etas)
        # print(eigvals_mp)
        # eigvals = np.array(eigvals_mp, dtype=float)
        # eigvecs = np.array(eigvecs_mp.tolist(), dtype=float)

        assert np.allclose(eigvecs @ np.diag(eigvals) @ eigvecs.T, anti_comm_mat_etas)

        # print()
        # print(anti_comm_mat_etas)
        # print()

        print(eigvals)
        # for val in eigvals_mp:
        for val in eigvals:
            if do_eigval_zero_check:
                print(val)
                assert not np.isclose(val, 0.0)
            assert val > 0.0

        self.gamma_d = eigvals * scale
        self.gamma_u = eigvecs.T

        # self.gammas: list[list[tuple[np.float64, Pauli]]] = []
        self.gammas: list[PauliSum] = []
        for i in range(self.num_generators):
            # a little bit different to eq. (80) in chapman_unified (why is there this
            # i^(j mod 2)? I calculated the anticommutator and the conjugation by hand and
            # I don't think there should be this i) as we define the anti_comm_mat_etas
            # with a different factor (only divided by dim(hilbert space) instead of 2 *
            # dim(hilbert space)
            # TODO:  double check on that these two statements; I'm just guessing here and
            # set the factor so that the gammas are properly normalised
            factor = np.float64((1 / self.gamma_d[i]) ** (0.5))
            gamma_vector = np.zeros(len(self.eta_vector_to_pauli_map), dtype=np.float64)
            for j in range(self.num_generators):
                gamma_vector += self.gamma_u[i, j] * self.eta_vectors[j]
            gamma_vector = gamma_vector * factor
            gamma = PauliSum([])
            for weight, op in zip(gamma_vector, self.eta_vector_to_pauli_map):
                gamma.single_add(weight, op)
            gamma.remove_zero_weights()
            self.gammas.append(gamma)

        # exhaustively check that the gammas behave correctly (it is quite slow and the
        # check on anti_comm_mat_gammas can fail due to numerical inaccuracies) {{{
        if do_checks:
            print()
            anti_comm_mat_gammas = np.zeros(
                (self.num_generators, self.num_generators), dtype=np.float64
            )
            for i in range(self.num_generators):
                for j in range(self.num_generators):
                    total_trace = 0
                    for weight_i, op_i in self.gammas[i].to_py_list():
                        for weight_j, op_j in self.gammas[j].to_py_list():
                            prod = op_i.multiply_as_paulis(op_j)
                            if prod.is_proportional_to(Pauli.identity(op_i.n())):
                                assert prod.phase() in [0, 2]
                                total_trace += (
                                    2
                                    * weight_i
                                    * weight_j
                                    * (-1) ** (prod.phase() // 2)
                                )
                    if i == j:
                        print(total_trace, "should be 2")
                    anti_comm_mat_gammas[i, j] = total_trace

            # print()
            # print(self.gammas[0].to_py_list())
            # print(anti_comm_mat_gammas[0, 0], "should be 2")
            # print(anti_comm_mat_gammas[1, 1], "should be 2")

            diff = anti_comm_mat_gammas - 2 * np.identity(self.num_generators)
            norm = 0
            print()
            # print(anti_comm_mat_gammas)
            for i in range(self.num_generators):
                for j in range(self.num_generators):
                    # print(abs(diff[i, j]))
                    norm += abs(diff[i, j])
            print(norm, "total")
            assert np.allclose(
                anti_comm_mat_gammas, 2 * np.identity(self.num_generators)
            )

            # if self.n <= 8:  # otherwise this is too expensive
            #     for i in range(self.num_generators):
            #         for j in range(self.num_generators):
            #             prod = paulis.list_to_matrix(
            #                 self.gammas[i].to_py_list()
            #             ) @ paulis.list_to_matrix(self.gammas[j].to_py_list())
            #             prod_inverse = paulis.list_to_matrix(
            #                 self.gammas[j].to_py_list()
            #             ) @ paulis.list_to_matrix(self.gammas[i].to_py_list())
            #             if i == j:
            #                 assert np.allclose(prod, np.identity(2**self.n))
            #             else:
            #                 assert np.allclose(prod, -prod_inverse)
            # print("checks passed")

        # }}}

    def init_gamma_bilinears(self):
        """multiplied an "i" in to make them hermitian"""
        # PERF: this loop takes quite some time
        self.gamma_bilinears: dict[tuple[int, int], PauliSum] = dict()
        for i in range(self.num_generators):
            for j in range(i + 1, self.num_generators):
                product = self.gammas[i].multiply(self.gammas[j])
                # need to make them hermitian
                # print([(w, op.to_string()) for w, op in product.to_py_list()])
                product.multiply_with_one_imag_unit()
                for w, op in product.to_py_list():
                    # print(w, op.to_string(), op.phase())
                    # assert op.get_hermitian_phase() in [0, 2]
                    pass
                self.gamma_bilinears[(i, j)] = product
        self.gamma_bilinears[(0, 0)] = PauliSum(
            [(np.float64(1.0), Pauli.identity(self.n))]
        )

    def bilinear_gamma_projection(
        self, pauli: Pauli
    ) -> list[tuple[int, int, np.complex128]]:
        """given a pauli, return the coefficients of its projection onto the gammas"""
        coeffs = []
        for (i, j), ops in self.gamma_bilinears.items():
            # coeff = paulis.list_and_single_hilbert_schmidt_inner_product(ops, pauli)
            coeff = ops.single_hilbert_schmidt_inner_product(pauli)
            if coeff != 0.0:
                coeffs.append((i, j, coeff))
        return coeffs

    def init_eta_currents(self):
        """multiplied an "i" in to make them hermitian; note the the potential
        `renormalise` of the etas (in __init__) is undone here, so that we get the correct
        currents"""
        self.eta_currents: list[PauliSum] = []
        for l in range(self.num_generators):
            current = PauliSum([])
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
                if self.eta_renormalisation_factor is not None:
                    factor = 1.0 / (
                        self.eta_renormalisation_factor[k]
                        * self.eta_renormalisation_factor[l_k]
                    )
                    prod.multiply_with_float(factor)
                    prod_inv.multiply_with_float(factor)
                commutator = prod.subtract(prod_inv)
                if k % 2 == 1:
                    current = current.subtract(commutator)
                else:
                    current = current.add(commutator)
            current.remove_zero_weights()
            # make them hermitian:
            current.multiply_with_one_imag_unit()
            for _, op in current.to_py_list():
                assert op.get_hermitian_phase() in [0, 2]
            self.eta_currents.append(current)

    def eta_currents_bilinear_gamma_projection(
        self,
    ) -> list[list[tuple[int, int, float]]]:
        ret = []
        for l in range(self.num_generators):
            current_coeffs = []
            for a in range(self.num_generators):
                for b in range(a + 1, self.num_generators):
                    coeff = 0
                    for k in range(l):
                        l_k = l - k
                        if k == l_k:
                            continue
                        coeff += (-1) ** k * (
                            self.gamma_u[a, k] * self.gamma_u[b, l_k]
                            - self.gamma_u[b, k] * self.gamma_u[a, l_k]
                        )
                    if coeff != 0.0:
                        assert coeff.imag == 0.0
                        current_coeffs.append(
                            (
                                a,
                                b,
                                2
                                * np.sqrt(self.gamma_d[a] * self.gamma_d[b])
                                * coeff.real,
                            )
                        )
            ret.append(current_coeffs)
        return ret


import mpmath as mp

mp.mp.dps = 100


# way slower with high dps, but high dps allows us to keep the residual norm more stable
class MpmathGramSchmidtProcess:
    def __init__(self, first_vector: np.ndarray, tolerance: float = 1e-10):
        norm = linalg.norm(first_vector)
        assert norm > 0.0
        # we always make sure everything is hermitian with real weights, so real values
        # are fine here
        # self.basis = np.array([first_vector / norm], dtype=np.float128).T
        self.basis = mp.matrix(first_vector / norm).T
        self.tolerance = tolerance
        self.norms = [norm]

    def num_vectors(self) -> int:
        return self.basis.cols

    def append_zeros(self):
        new_basis = mp.matrix(self.basis.rows + 1, self.basis.cols)
        for i in range(self.basis.rows):
            for j in range(self.basis.cols):
                new_basis[i, j] = self.basis[i, j]
        for j in range(self.basis.cols):
            new_basis[self.basis.rows, j] = 0.0
        self.basis = new_basis
        # )

    def add_vector(self, vector: np.ndarray) -> bool:
        assert len(vector) == self.basis.rows
        vector = mp.matrix(vector)  # pyright: ignore
        projection = self.basis * (self.basis.T * vector)
        orthogonal_component = vector - projection
        norm = mp.norm(orthogonal_component)
        # TODO: print this here into some file which one should always check for sensible
        # values
        self.norms.append(norm)
        print(f"Gram-Schmidt: norm of orthogonal component is {norm}")
        if norm > self.tolerance:
            new_basis_vector = orthogonal_component / norm
            new_basis = mp.matrix(self.basis.rows, self.basis.cols + 1)
            for i in range(self.basis.rows):
                for j in range(self.basis.cols):
                    new_basis[i, j] = self.basis[i, j]
            for i in range(self.basis.rows):
                new_basis[i, self.basis.cols] = new_basis_vector[i, 0]
            self.basis = new_basis
            return True
        else:
            return False


class GramSchmidtProcess:
    def __init__(self, first_vector: np.ndarray, tolerance: float = 1e-10):
        norm = linalg.norm(first_vector)
        assert norm > 0.0
        # we always make sure everything is hermitian with real weights, so real values
        # are fine here
        self.basis = np.array([first_vector / norm], dtype=np.float128).T
        self.tolerance = tolerance
        self.norms = [norm]

    def num_vectors(self) -> int:
        return self.basis.shape[1]

    def append_zeros(self):
        self.basis = np.vstack(
            [self.basis, np.zeros((1, self.basis.shape[1]), dtype=np.float128)]
        )

    def add_vector(self, vector: np.ndarray) -> bool:
        assert len(vector) == self.basis.shape[0]
        projection = self.basis @ (self.basis.conj().T @ vector)
        orthogonal_component = vector - projection
        norm = linalg.norm(orthogonal_component)
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


def test_projections():
    from models.fendley import Fendley
    from models.weights import ConstantWeight
    import phase_diagram

    fendley = Fendley(1, ConstantWeight(1.0), ConstantWeight(1.0), ConstantWeight(1.0))
    simplicial_mode = fendley.example_simplicial_modes["IIIIX"][0]
    generators = Generators(simplicial_mode, fendley.hamiltonian)

    dim = 2**fendley.n

    generators.init_gammas()
    generators.init_gamma_bilinears()
    generators.init_eta_currents()
    gamma_bilinears = generators.gamma_bilinears
    eta_currents = generators.eta_currents

    currents_coeffs = generators.eta_currents_bilinear_gamma_projection()

    fendley_coeffs = dict()
    for w, op in zip(fendley.hamiltonian.weights, fendley.hamiltonian.operators):
        coeff = generators.bilinear_gamma_projection(op)
        for j, k, c in coeff:
            assert c.imag == 0
            fendley_coeffs[(j, k)] = fendley_coeffs.get((j, k), 0) + c.real * w

    mat = fendley.hamiltonian.to_matrix()
    mat_reconstructed = np.zeros_like(mat)
    for (i, j), c in fendley_coeffs.items():
        for w, op in gamma_bilinears[(i, j)]:
            mat_reconstructed += c * w * op.to_matrix()
    assert np.allclose(mat, mat_reconstructed)

    mat = np.zeros((dim, dim), dtype=complex)
    for current in eta_currents:
        for w, op in current:
            mat += w * op.to_matrix()
    mat_reconstructed = np.zeros((dim, dim), dtype=complex)
    for coeffs in currents_coeffs:
        for i, j, w in coeffs:
            for w2, op in gamma_bilinears[(i, j)]:
                mat_reconstructed += w * w2 * op.to_matrix()
    assert np.allclose(mat, mat_reconstructed)

    fendley.extend_with_currents(
        generators.eta_currents, [np.float64(1.0) for _ in generators.eta_currents]
    )

    total_coeffs = dict()
    for w, op in zip(fendley.hamiltonian.weights, fendley.hamiltonian.operators):
        coeff = generators.bilinear_gamma_projection(op)
        for j, k, c in coeff:
            assert c.imag == 0
            total_coeffs[(j, k)] = total_coeffs.get((j, k), 0) + c.real * w
    mat = fendley.hamiltonian.to_matrix()
    mat_reconstructed = np.zeros_like(mat)
    for (i, j), c in total_coeffs.items():
        for w, op in generators.gamma_bilinears[(i, j)]:
            mat_reconstructed += c * w * op.to_matrix()
    assert np.allclose(mat, mat_reconstructed)

    reconstruct_total_coeffs = fendley_coeffs.copy()
    for coeffs in currents_coeffs:
        for i, j, c in coeffs:
            reconstruct_total_coeffs[(i, j)] = (
                reconstruct_total_coeffs.get((i, j), 0) + c
            )

    for key in set(total_coeffs.keys()).union(set(reconstruct_total_coeffs.keys())):
        assert np.isclose(
            total_coeffs.get(key), reconstruct_total_coeffs.get(key)  # pyright: ignore
        )

    h = np.zeros((generators.num_generators, generators.num_generators))
    for (i, j), coeff in reconstruct_total_coeffs.items():
        h[i, j] = coeff / 2
        h[j, i] = -coeff / 2
    # print(h)

    lm, _ = phase_diagram.skew_diagonalise(h)
    phase_diagram.smoothen_lamda(lm)
    lm_pairs = phase_diagram.get_lamda_pairs(lm)
    all_possible_vals = phase_diagram.get_lamda_eigenvalues(lm_pairs)

    vals, _ = fendley.hamiltonian.diagonalise()
    unique_vals = set()
    for v in vals:
        already_in = False
        for u in unique_vals:
            if np.isclose(v, u):
                already_in = True
                break
        if not already_in:
            unique_vals.add(v)

    unique_vals = sorted(unique_vals)
    all_possible_vals = sorted(all_possible_vals)
    assert len(unique_vals) == len(all_possible_vals)
    for i in range(len(unique_vals)):
        assert np.isclose(unique_vals[i], all_possible_vals[i])
