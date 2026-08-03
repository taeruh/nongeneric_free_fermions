import numpy as np
from sage.all import Graph
from numpy import linalg
from numpy.typing import NDArray
import scipy

from hamiltonian import Hamiltonian
from rust_backend.paulis import Pauli, PauliSum
import graph_helper


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
        self.simplicial_mode = simplicial_mode
        self.hamiltonian = hamiltonian
        self.eta_vector_to_pauli_map = [simplicial_mode[1]]
        self.pauli_to_eta_vector_map = {simplicial_mode[1].to_string(): 0}
        self.eta_vector_to_maybe_path_map = [([-1], 0)]
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
                for ham_label, (ham_weight, ham_op) in enumerate(
                    zip(hamiltonian.weights, hamiltonian.operators)
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
                            self.pauli_to_eta_vector_map[comm_op.to_string()] = (
                                len(self.eta_vector_to_pauli_map) - 1
                            )
                            # op could be in pauli_to_eta_vector_map with a minus sign,
                            # e.g., the following sequence would have done that (note that
                            # we add to eta independently of already_in_vectors): op is
                            # not in eta and not in eta_vector_to_pauli_map -> get
                            # weight;op so that op is in eta_vector_to_pauli_map and
                            # pauli_to_eta_vector_map and add weight;op to eta -> get
                            # -weight;op so that it is removed from eta -> get
                            # any_weight,-op which gets not added to
                            # pauli_to_eta_vector_map because it is already in
                            # eta_vector_to_pauli_map but it initialises a new entry with
                            # -op in eta as the +op entry has been removed in the previous
                            # step
                            if op.to_string() not in self.pauli_to_eta_vector_map:
                                op_shifted = op.copy()
                                op_shifted.add_to_phase(2)
                                op_index = self.pauli_to_eta_vector_map[
                                    op_shifted.to_string()
                                ]
                            else:
                                op_index = self.pauli_to_eta_vector_map[op.to_string()]
                            prev_path = self.eta_vector_to_maybe_path_map[op_index]
                            comm_op_maybe_path = prev_path[0].copy()
                            label_already_in_path = False
                            to_remove = 0
                            for i, vert in enumerate(comm_op_maybe_path):
                                if vert == ham_label:
                                    label_already_in_path = True
                                    to_remove = i
                                    break
                            if label_already_in_path:
                                del comm_op_maybe_path[to_remove]
                            else:
                                # reverse later
                                comm_op_maybe_path.append(ham_label)
                            self.eta_vector_to_maybe_path_map.append(
                                # the phase is wrong, but we don't care write now
                                (comm_op_maybe_path, (len(comm_op_maybe_path) - 1) % 4)
                            )

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

        for path, _ in self.eta_vector_to_maybe_path_map:
            path.reverse()

        self.num_generators = len(self.etas)

    def path_to_operator(self, path: tuple[list[int], int]) -> Pauli:
        """
        helper to reconstruct the according pauli from sorted paths in
        self.eta_vector_to_path_map and self.eta_path_bilinears
        """
        path_op = Pauli.identity(self.n)
        for vertex in path[0]:
            if vertex == -1:
                next_op = self.simplicial_mode[1]
            else:
                next_op = self.hamiltonian.operators[vertex]
            path_op = path_op.multiply_as_paulis(next_op)
        path_op.add_to_phase(path[1])
        return path_op

    def test_eta_path_decompositions(self, graph: Graph | None = None):
        if graph is None:
            graph = self.hamiltonian.get_frustration_graph()
        for ieta, vec in enumerate(self.eta_vectors):
            print(ieta, ":")
            for i, w in enumerate(vec):
                if not np.isclose(w, 0.0):
                    op = self.eta_vector_to_pauli_map[i]
                    path = self.eta_vector_to_maybe_path_map[i]
                    if not op.is_proportional_to(self.path_to_operator(path)):
                        print()
                        print(
                            op.to_string(),
                            "is not proportional to",
                            self.path_to_operator(path).to_string(),
                            w,
                        )
                        assert False
                    path_without_mode = path[0].copy()
                    path_without_mode.remove(-1)
                    if len(path_without_mode) >= 2:
                        if not graph_helper.is_induced_path(graph, path_without_mode):
                            print(w, op.to_string())
                            print(path_without_mode)
            print()


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
