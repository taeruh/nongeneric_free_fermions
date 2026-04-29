import numpy as np

from hamiltonian import Hamiltonian
from rust_backend.paulis import Pauli, PauliSum


class GeneratorsWithoutGramSchmidt:
    def __init__(
        self,
        simplicial_mode: tuple[np.float64 | float, Pauli],
        hamiltonian: Hamiltonian,
        max_eta: int,
        renormalise: bool = False,
        eta_normalisation_factor: np.float64 = np.float64(1.0),
    ):
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

        for index in range(max_eta):
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

                        # PERF: if already_in_vectors is false, we could do something more
                        # efficient in single_add, as we know that it cannot be in eta
                        eta.single_add(comm_weight, comm_op)
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
            print("calculated eta", index + 1)

        self.num_generators = len(self.etas)

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
