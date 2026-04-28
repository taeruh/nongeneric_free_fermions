class IntegerPauli:
    def __init__(self, n: int, u: list[bool], l: list[bool], phase: int):
        assert len(u) == n
        assert len(l) == n
        assert phase in [0, 1, 2, 3]
        self.n = n
        self.u = u
        self.l = l
        self.phase = phase

    def __repr__(self):
        return f"({self.u},{self.l}, {self.phase}; {self.n})"

    def clone(self):
        return IntegerPauli(self.n, self.u.copy(), self.l.copy(), self.phase)

    def add_to_phase(self, phase_diff: int) -> None:
        self.phase = (self.phase + phase_diff) % 4

    @classmethod
    def identity(cls, n: int) -> "IntegerPauli":
        return cls(n, [False] * n, [False] * n, 0)

    @classmethod
    def from_indices(
        cls, n: int, z_trues: list[int], x_trues: list[int], phase: int
    ) -> "IntegerPauli":
        u = [False] * n
        l = [False] * n
        for i in z_trues:
            u[i] = True
        for i in x_trues:
            l[i] = True
        return cls(n, u, l, phase)

    def to_string(self, flip: bool = False, with_phase: bool = True) -> str:
        """in the string we order from right to left, i.e., p_n-1, ... p_0 (as in a
        bitvector)"""
        pauli_str = ""
        if flip:
            iter = range(self.n)
        else:
            iter = range(self.n - 1, -1, -1)
        for i in iter:
            if self.u[i] and self.l[i]:
                pauli_str += "Y"
            elif self.u[i]:
                pauli_str += "Z"
            elif self.l[i]:
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

    def is_proportional_to(self, other: "IntegerPauli") -> bool:
        return self.u == other.u and self.l == other.l

    def phase_difference(self, other: "IntegerPauli") -> int:
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
            if self.u[i] and self.l[i]:
                # multiply by 1 = i * -i and take i into the phase and make ZX to Y
                phase = (phase + 1) % 4
        return phase

    def symplectic_inner_product(self, other: "IntegerPauli") -> int:
        assert self.n == other.n
        ip = False
        for i in range(self.n):
            ip ^= (self.l[i] & other.u[i]) ^ (self.u[i] & other.l[i])
        return ip

    def multiply_as_paulis(self, other: "IntegerPauli") -> "IntegerPauli":
        assert self.n == other.n
        new_u = [a ^ b for a, b in zip(self.u, other.u)]
        new_l = [a ^ b for a, b in zip(self.l, other.l)]
        new_phase = (self.phase + other.phase) % 4
        for i in range(self.n):
            if self.l[i] and other.u[i]:
                new_phase = (new_phase + 2) % 4
        return IntegerPauli(self.n, new_u, new_l, new_phase)


class IntegerPauliSum:
    def __init__(self, ops: list[tuple[int, IntegerPauli]]):
        self.ops = ops

    def multiply(self, other: "IntegerPauliSum") -> "IntegerPauliSum":
        result = []
        for weight1, pauli1 in self.ops:
            for weight2, pauli2 in other.ops:
                prod = pauli1.multiply_as_paulis(pauli2)
                add_helper(result, weight1 * weight2, prod)
        removal_helper(result)
        return IntegerPauliSum(result)

    def add(self, other: "IntegerPauliSum") -> "IntegerPauliSum":
        result = self.ops.copy()
        for weight, pauli in other.ops:
            add_helper(result, weight, pauli)
        removal_helper(result)
        return IntegerPauliSum(result)

    def subtract(self, other: "IntegerPauliSum") -> "IntegerPauliSum":
        result = self.ops.copy()
        for weight, pauli in other.ops:
            add_helper(result, -weight, pauli)
        removal_helper(result)
        return IntegerPauliSum(result)

    def single_add(self, weight: int, pauli: IntegerPauli) -> None:
        add_helper(self.ops, weight, pauli)
        removal_helper(self.ops)

    def remove_zero_weights(self) -> None:
        removal_helper(self.ops)

    def multiply_with_one_imag_unit(self) -> None:
        for _, pauli in self.ops:
            pauli.phase = (pauli.phase + 1) % 4

    def multiply_with_int(self, scalar: int) -> None:
        for i in range(len(self.ops)):
            weight, pauli = self.ops[i]
            self.ops[i] = (weight * scalar, pauli)


def add_helper(
    list: list[tuple[int, IntegerPauli]], weight: int, pauli: IntegerPauli
) -> None:
    for i, (ret_weight, ret_pauli) in enumerate(list):
        if pauli.is_proportional_to(ret_pauli):
            phase = pauli.phase_difference(ret_pauli)
            assert phase in [0, 2]
            if phase == 0:
                list[i] = (ret_weight + weight, ret_pauli)
            else:
                list[i] = (ret_weight - weight, ret_pauli)
            return
    list.append((weight, pauli.clone()))


def removal_helper(ops: list[tuple[int, IntegerPauli]]) -> None:
    to_remove = []
    for i, (weight, _) in enumerate(ops):
        if weight == 0:
            to_remove.append(i)
    for i in reversed(to_remove):
        ops.pop(i)
