import numpy as np
from numpy.typing import NDArray
from rust_backend.paulis import Pauli
from typing import Sequence


def list_to_matrix(
    ops: Sequence[tuple[np.float64, Pauli]],
) -> NDArray[np.complex128]:
    """given a list of (weight, pauli) pairs, return the matrix representation of the
    sum of these operators"""
    if len(ops) == 0:
        return np.array([[0]], dtype=complex)
    n = ops[0][1].n
    result = np.zeros((2**n, 2**n), dtype=complex)
    for weight, pauli in ops:
        result += weight * pauli.to_matrix()
    return result
