from sage.all import Graph
from typing import Tuple
from integer_paulis import IntegerPauli


class IntegerHamiltonian:
    def __init__(self, weights: list[int], ops: list[IntegerPauli]):
        assert len(weights) == len(ops)
        self.num_ops = len(ops)
        if self.num_ops == 0:
            print("Warning: Hamiltonian has no terms; setting spin number to 1!")
            self.n = 1
        else:
            self.n = ops[0].n

        # note that the proportionality check also implicitly checks that all ops have the
        # same n, since otherwise they cannot be proportional (the method would raise an
        # error)
        has_prop_terms, _ = has_proportional_terms(ops)
        if has_prop_terms:
            raise ValueError("hamiltonian has proportional terms")
        has_non_herm_terms, _ = has_non_hermitian_terms(ops)
        if has_non_herm_terms:
            raise ValueError("hamiltonian has non-hermitian terms")

        self.weights = weights
        self.operators = ops

    def clone(self) -> "IntegerHamiltonian":
        clone_weights = self.weights.copy()
        clone_ops = [op.clone() for op in self.operators]
        return IntegerHamiltonian(clone_weights, clone_ops)

    def get_frustration_graph(self) -> Graph:
        """Return the frustration graph with the vertex weights."""
        g = Graph()
        for i in range(self.num_ops):
            g.add_vertex(i)
        for i in range(self.num_ops):
            for j in range(i + 1, self.num_ops):
                if self.operators[i].symplectic_inner_product(self.operators[j]):
                    g.add_edge(i, j)
        return g


def has_proportional_terms(
    ops: list[IntegerPauli],
) -> Tuple[bool, None | list[Tuple[int, int]]]:
    """Check if there are proportional terms in the list of Paulis."""
    proportional_pairs: list[Tuple[int, int]] = []
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            if ops[i].is_proportional_to(ops[j]):
                proportional_pairs.append((i, j))
    if len(proportional_pairs) > 0:
        return True, proportional_pairs
    else:
        return False, None


def has_non_hermitian_terms(ops: list[IntegerPauli]) -> Tuple[bool, None | list[int]]:
    """Check if there are non-Hermitian terms in the list of Paulis."""
    non_hermitian_indices: list[int] = []
    for i, op in enumerate(ops):
        if not op.get_hermitian_phase() in [0, 2]:
            non_hermitian_indices.append(i)
    if len(non_hermitian_indices) > 0:
        return True, non_hermitian_indices
    else:
        return False, None
