import numpy as np
import scipy
from scipy import linalg
from numpy.typing import NDArray
from sage.all import Graph
import math

from typing import Callable, Iterable

CouplingFn = Callable[[float, float, float], float]


class PeriodicBBar:
    """
    Periodic odd overlined couplings generated from three seed functions:
      \bar b_1(alpha,beta,gamma),
      \bar b_3(alpha,beta,gamma),
      \bar b_5(alpha,beta,gamma),
    and then repeated with period 3 in m, i.e. in the sequence
      \bar b_{2m+1},  m = 0,1,2,3,...
    """

    def __init__(self, f_bbar1: CouplingFn, f_bbar3: CouplingFn, f_bbar5: CouplingFn):
        self.fns = (f_bbar1, f_bbar3, f_bbar5)

    def __call__(
        self, odd_index: int, alpha: float, beta: float, gamma: float
    ) -> float:
        """
        Return \bar b_{odd_index} for odd_index = 1,3,5,7,...
        Uses period 3 in m where odd_index = 2m+1.
        """
        if odd_index % 2 != 1:
            raise ValueError(f"Expected an odd index, got {odd_index}.")
        m = (odd_index - 1) // 2
        fn = self.fns[m % 3]
        return fn(alpha, beta, gamma)


def gap_from_coeff(
    coeff: list[float], imag_tol: float = 1e-4, root_eps: float = 1e-9
) -> float:
    """
    Estimate the lowest positive single-particle energy from the largest positive real root z_max:
        gap ~ 1 / sqrt(z_max)

    Returns np.nan if no reliable positive real root is found.
    """
    arr = np.array(coeff[::-1], dtype=np.complex128)  # descending powers for np.roots
    roots = np.roots(arr)
    # print(roots)

    realish = roots[np.abs(roots.imag) < imag_tol]
    real_pos = realish.real[realish.real > root_eps]
    if real_pos.size == 0:
        return np.nan

    zmax = np.max(real_pos)
    if not np.isfinite(zmax) or zmax <= 0:
        return np.nan

    return 1.0 / math.sqrt(zmax)


def compute_poly_even(
    M: int,
    alpha: float,
    beta: float,
    gamma: float,
    bbar: PeriodicBBar,
    sign: int = +1,
) -> list[float]:
    """
    Compute P_M(z) for even M using the general Hamiltonian-(3.16)-type recursion.

    Coefficients:
      S_{2m}^2 = b_{2m-1}^2 + b_{2m}^2 + \bar b_{2m+1}^2
      A_{2m-1}^2 = (b_{2m-3} b_{2m} + sign * b_{2m-2} \bar b_{2m+1})^2
      C_{2m-2}^2 = (b_{2m-3} \bar b_{2m+3})^2
    """
    if M % 2 != 0:
        raise ValueError(f"M must be even, got {M}.")

    P: dict[int, list[float]] = {-2: [1.0], 0: [1.0]}

    def get(n: int) -> list[float]:
        if n in P:
            return P[n]
        if n < 0:
            return [0.0]
        raise KeyError(n)

    for m in range(1, M // 2 + 1):
        b2m1 = b_orig(2 * m - 1, alpha, beta, gamma)
        b2m = b_orig(2 * m, alpha, beta, gamma)
        b2m_2 = b_orig(2 * m - 2, alpha, beta, gamma)
        b2m_3 = b_orig(2 * m - 3, alpha, beta, gamma)

        bb2m1 = bbar(2 * m + 1, alpha, beta, gamma)
        bb2m3 = bbar(2 * m + 3, alpha, beta, gamma)

        s2 = b2m1**2 + b2m**2 + bb2m1**2
        a2 = (b2m_3 * b2m + sign * b2m_2 * bb2m1) ** 2
        c2 = (b2m_3 * bb2m3) ** 2

        p = get(2 * m - 2)
        p = poly_add(p, get(2 * m - 4), scale=-s2, shift=1)
        p = poly_add(p, get(2 * m - 6), scale=a2, shift=2)
        p = poly_add(p, get(2 * m - 8), scale=-c2, shift=2)
        P[2 * m] = p

    return P[M]


def poly_add(
    p: list[float], q: list[float], scale: float = 1.0, shift: int = 0
) -> list[float]:
    """
    Add scale * z^shift * q(z) to p(z), with coefficient lists in ascending powers.
    """
    res = p.copy()
    need = len(q) + shift
    if len(res) < need:
        res.extend([0.0] * (need - len(res)))
    for i, c in enumerate(q):
        res[i + shift] += scale * c
    return res


def b_orig(n: int, alpha: float, beta: float, gamma: float) -> float:
    """
    3-periodic FFD coupling:
      b_{3m+1}=alpha, b_{3m+2}=beta, b_{3m}=gamma
    """
    r = n % 3
    return alpha if r == 1 else beta if r == 2 else gamma


class SamsCalculation:

    def __init__(
        self,
        M: int,
        alpha: float,
        beta: float,
        gamma: float,
        sign: int = +1,
        imag_tol: float = 1e-4,
        root_eps: float = 1e-9,
    ):

        bbar_fn = lambda alpha, beta, gamma: 0.0
        bbar = PeriodicBBar(bbar_fn, bbar_fn, bbar_fn)
        coeff = compute_poly_even(M, alpha, beta, gamma, bbar=bbar, sign=sign)
        # print(coeff)
        self.gap = gap_from_coeff(coeff, imag_tol=imag_tol, root_eps=root_eps)


class Calculation:
    def __init__(
        self,
        num_triangles: int,
        alpha: float,
        beta: float,
        gamma: float,
        currents_alpha: list[float],
    ):
        self.num_triangles = num_triangles
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.currents_alpha = currents_alpha

        # P*_1, P*_2, P*_3 with index  being the number of triangles modulo 3
        #           P*_-2, P*_-1, P*_0
        self.polynomials = [[], [], [1]]
        for num_t in range(1, num_triangles + 1):
            # print(num_t)
            idx = num_t - 1
            p_m1 = self.polynomials[(idx - 1) % 3]
            p_m2 = self.polynomials[(idx - 2) % 3]
            p_m3 = self.polynomials[(idx - 3) % 3]
            # p = [0] * (num_t + 2)
            p = [0] * (num_t + 1)
            for i, c in enumerate(p_m1):
                p[i] += c
                p[i + 1] -= (alpha + beta + gamma) * c
            for i, c in enumerate(p_m2):
                p[i + 2] -= (alpha * beta + alpha * gamma + beta * gamma) * c
            for i, c in enumerate(p_m3):
                p[i + 3] -= alpha * beta * gamma * c
            self.polynomials[idx % 3] = p
            # print(p)

        self.p = self.polynomials[(num_triangles - 1) % 3]
        self.pk = self.polynomials[(num_triangles - 2) % 3]
        # print(p)

        self.roots = np.roots(self.p[::-1])
        # print(self.roots)
        self.roots.sort()

        # assert num_triangles > 1

        # assert self.roots[0] < self.roots[1]

        # print(self.roots)
        # self.roots = self.roots[np.abs(self.roots.imag) < 1e-4]
        # print(self.roots)
        self.roots = self.roots.real[self.roots.real > 1e-9]
        # print(self.roots)
        if self.roots.size == 0:
            return np.nan

        zmax = np.max(self.roots)
        if not np.isfinite(zmax) or zmax <= 0:
            return np.nan

        self.gap = 1.0 / np.sqrt(zmax)


def skew_diagonalise(
    h: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """return (Lambda, K) such that h = K lambda K^dagger"""
    lm, km = linalg.schur(h, output="real")  # pyright: ignore
    assert np.allclose(h, km @ lm @ km.T)
    return lm, km  # pyright: ignore


def smoothen_lamda(lm: NDArray[np.float64], eps: float = 1e-12) -> None:
    dims = lm.shape
    for i in range(dims[0]):
        for j in range(dims[1]):
            if np.abs(lm[i, j]) < eps:
                lm[i, j] = 0.0


def get_lamda_pairs(lm: NDArray[np.float64]) -> list[tuple[tuple[int, int], float]]:
    dims = lm.shape
    lm_pairs = []
    i = 0
    while i < dims[0]:
        for j in range(dims[1]):
            x = lm[i, j]
            if x != 0.0:
                assert np.isclose(x, -lm[i + 1, j - 1])
                lm_pairs.append(((i, i + 1), x))
                i += 1
                break
        i += 1
    return lm_pairs


def get_lamda_minimum_eigenvalue(
    lm_pairs: list[tuple[tuple[int, int], float]],
) -> float:
    val = 0.0
    for _, x in lm_pairs:
        val -= abs(x)
    return 2 * val


# TODO: implement it with the efficient sorting (cf. linegraph guiding notes)
def get_lamda_eigenvalues(lm_pairs: list[tuple[tuple[int, int], float]]) -> list[float]:
    all_possible_vals = set()
    for i in range(0, 2 ** len(lm_pairs)):
        val = 0.0
        for j, (_, x) in enumerate(lm_pairs):
            if (i >> j) & 1:
                val += x
            else:
                val -= x
        all_possible_vals.add(2 * val)
    return sorted(all_possible_vals)


def get_gap(lm_eigenvalues: list[float]) -> float:
    """assume lm_eigenvalues is sorted in ascending order, return the gap between the minimum and the next one"""
    min_value = lm_eigenvalues[0]
    # loop to catch potential degeneracies
    gap = 0
    for val in lm_eigenvalues[1:]:
        if np.isclose(val, min_value):
            continue
        else:
            gap = val - min_value
            break
    return gap


def triangle_grid(num_samples: int, factor: float = 3) -> NDArray:
    points = []
    for i in range(num_samples + 1):
        for j in range(num_samples + 1 - i):
            k = num_samples - i - j
            alpha = i / num_samples
            beta = j / num_samples
            gamma = k / num_samples
            points.append([alpha, beta, gamma])
    return np.array(points) * factor


def triangle_grid_inner(num_samples, factor=3):
    points = triangle_grid(num_samples, factor)
    to_remove = []
    for i, point in enumerate(points):
        if point[0] == 0 or point[1] == 0 or point[2] == 0:
            to_remove.append(i)
    return np.delete(points, to_remove, axis=0)


def points_to_plot_coordinates(points: NDArray) -> tuple[NDArray, NDArray]:
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.5, np.sqrt(3) / 2])
    xy = points[:, 0, None] * a + points[:, 1, None] * b + points[:, 2, None] * c
    return xy[:, 0], xy[:, 1]
