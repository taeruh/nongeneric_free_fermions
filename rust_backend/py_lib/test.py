#!/usr/bin/env python

from rust_backend.paulis import Pauli, PauliSum

def main():
    a = Pauli(3, [False, True, False], [False, False, True], 0)
    b = Pauli.from_indices(3, [0, 2], [1], 0)
    sum = PauliSum([(1., a), (1., b)])
    print([(w, p.to_string()) for w, p in sum.to_py_list()])

if __name__ == "__main__":
    main()
