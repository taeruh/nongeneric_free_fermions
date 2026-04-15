import numpy as np
from numpy.typing import NDArray


class Pauli:
    def __init__(self, n: int, z: NDArray[np.bool], x: NDArray[np.bool], phase: int):
        assert len(z) == n
        assert len(x) == n
        assert phase in [0, 1, 2, 3]
        self.n = n
        self.z = z
        self.x = x
        self.phase = phase

    def __repr__(self):
        return f"({self.z.astype(np.int_)},{self.x.astype(np.int_)}; {self.n}, {self.phase})"

    def clone(self):
        return Pauli(self.n, self.z.copy(), self.x.copy(), self.phase)

    @classmethod
    def identity(cls, n: int) -> "Pauli":
        return cls(n, np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), 0)

    @classmethod
    def from_indices(
        cls, n: int, z_trues: list[int], x_trues: list[int], phase: int
    ) -> "Pauli":
        z = np.zeros(n, dtype=bool)
        x = np.zeros(n, dtype=bool)
        for i in z_trues:
            z[i] = True
        for i in x_trues:
            x[i] = True
        return cls(n, z, x, phase)

    def to_string(self, flip: bool = False, with_phase: bool = True) -> str:
        """in the string we order from right to left, i.e., p_n-1, ... p_0 (as in a
        bitvector)"""
        pauli_str = ""
        if flip:
            iter = range(self.n)
        else:
            iter = range(self.n - 1, -1, -1)
        for i in iter:
            if self.z[i] and self.x[i]:
                pauli_str += "Y"
            elif self.z[i]:
                pauli_str += "Z"
            elif self.x[i]:
                pauli_str += "X"
            else:
                pauli_str += "I"
        if with_phase:
            phase = self.get_hermitian_phase()
            pauli_str += ", "
            if phase == 0:
                pauli_str += "+1"
            elif phase == 1:
                pauli_str += "+i"
            elif phase == 2:
                pauli_str += "-1"
            else:
                pauli_str += "-i"
        return pauli_str

    def to_matrix(self) -> NDArray[np.complex128]:
        """
        Return the matrix representation of the Pauli operator. The matrix indices are
        ordered as bitvectors, i.e., 0 = 000, 1 = 001, 2 = 010, 3 = 011, etc., where the
        paulis are associated to qubits from right to left (cf. to_pauli_string).
        """
        pauli_matrices = {
            (0, 0): np.array([[1, 0], [0, 1]], dtype=complex),
            (1, 0): np.array([[1, 0], [0, -1]], dtype=complex),
            (0, 1): np.array([[0, 1], [1, 0]], dtype=complex),
            (1, 1): np.array([[0, 1], [-1, 0]], dtype=complex),
        }
        result = np.array([[1]], dtype=complex)
        for i in range(self.n):
            z = int(self.z[i])
            x = int(self.x[i])
            result = np.kron(pauli_matrices[(z, x)], result)
        return np.complex128((1j) ** self.phase) * result

    def is_proportional_to(self, other: "Pauli") -> bool:
        return np.array_equal(self.z, other.z) and np.array_equal(self.x, other.x)

    def phase_difference(self, other: "Pauli") -> int:
        """
        assuming self and other are proportional, returns the phase difference between
        self and other, more precisely: self = (i^phase_difference) * other
        """
        assert self.is_proportional_to(other)
        return ((self.phase - other.phase) + 4) % 4

    def get_hermitian_phase(self) -> int:
        """
        the phase when writing the pauli as X/Y/Z string
        """
        phase = self.phase
        for i in range(self.n):
            if self.z[i] and self.x[i]:
                # multiply by 1 = i * -i and take i into the phase and make ZX to Y
                phase = (phase + 1) % 4
        return phase

    def symplectic_inner_product(self, other: "Pauli") -> int:
        """abc"""
        assert self.n == other.n
        ip = False
        for i in range(self.n):
            ip ^= (self.x[i] & other.z[i]) ^ (self.z[i] & other.x[i])
        return ip

    def multiply_as_paulis(self, other: "Pauli") -> "Pauli":
        assert self.n == other.n
        new_z = self.z ^ other.z
        new_x = self.x ^ other.x
        new_phase = (self.phase + other.phase) % 4
        for i in range(self.n):
            if self.x[i] and other.z[i]:
                new_phase = (new_phase + 2) % 4
        return Pauli(self.n, new_z, new_x, new_phase)


def list_to_matrix(ops: list[tuple[np.complex128, Pauli]]) -> NDArray[np.complex128]:
    """given a list of (weight, pauli) pairs, return the matrix representation of the
    sum of these operators"""
    if len(ops) == 0:
        return np.array([[0]], dtype=complex)
    n = ops[0][1].n
    result = np.zeros((2**n, 2**n), dtype=complex)
    for weight, pauli in ops:
        result += weight * pauli.to_matrix()
    return result


def list_multiplication(
    ops1: list[tuple[np.complex128, Pauli]], ops2: list[tuple[np.complex128, Pauli]]
) -> list[tuple[np.complex128, Pauli]]:
    """given two lists of (weight, pauli) pairs, return the list of (weight, pauli)
    pairs corresponding to the product of the sums of these operators"""
    result = []
    for weight1, pauli1 in ops1:
        for weight2, pauli2 in ops2:
            prod = pauli1.multiply_as_paulis(pauli2)
            already_in = False
            for i, (ret_weight, ret_pauli) in enumerate(result):
                if prod.is_proportional_to(ret_pauli):
                    phase = prod.phase_difference(ret_pauli)
                    result[i] = (
                        ret_weight + weight1 * weight2 * (1j) ** phase,
                        ret_pauli,
                    )
                    already_in = True
                    break
            if not already_in:
                result.append((weight1 * weight2, prod))
    to_remove = []
    for i, (weight, _) in enumerate(result):
        if np.isclose(weight, 0):
            to_remove.append(i)
    for i in reversed(to_remove):
        del result[i]
    return result


def list_addition(
    ops1: list[tuple[np.complex128, Pauli]], ops2: list[tuple[np.complex128, Pauli]]
) -> list[tuple[np.complex128, Pauli]]:
    """given two lists of (weight, pauli) pairs, return the list of (weight, pauli)
    pairs corresponding to the sum of the sums of these operators"""
    result = ops1.copy()
    for weight, pauli in ops2:
        already_in = False
        for i, (ret_weight, ret_pauli) in enumerate(result):
            if pauli.is_proportional_to(ret_pauli):
                phase = pauli.phase_difference(ret_pauli)
                result[i] = (
                    ret_weight + weight * (1j) ** phase,
                    ret_pauli,
                )
                already_in = True
                break
        if not already_in:
            result.append((weight, pauli))
    return result


def list_and_single_hilbert_schmidt_inner_product(
    ops: list[tuple[np.complex128, Pauli]], pauli: Pauli, unique_list: bool = False
) -> np.complex128:
    """given a list of (weight, pauli) pairs, return the hilbert schmidt inner product of
    the sum of these operators with another pauli; the paulis in the list should be
    hermition"""
    for _, op in ops:
        assert op.get_hermitian_phase() in [0, 2]
    total = np.complex128(0.0 + 0.0j)
    for weight, op in ops:
        if op.is_proportional_to(pauli):
            phase = pauli.phase_difference(op)
            total += weight * (1j) ** phase
            if unique_list:
                break
    return total
