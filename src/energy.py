import numpy as np
import scipy
from scipy import linalg
from numpy.typing import NDArray
from sage.all import Graph


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

        # self.p = [c / 1000 for c in self.p]

        self.roots = np.roots(self.p[::-1])

        from numpy.polynomial import Chebyshev

        import mpmath as mp
        mp.mp.dps = 100
        alt_roots = mp.polyroots(self.p[::-1], maxsteps=1000, error=False)

        alt_roots = np.array(alt_roots)
        alt_roots.sort()

        # has_imag = False
        # for root in self.roots:
        #     if np.abs(root.imag) > 1e-10:
        #         has_imag = True
        #         break
        # if has_imag:
        #     # assert False
        #     vals = [alpha, beta, gamma]
        #     vals.sort()
        #     print(vals)
        #     print(self.roots)
        #     # for all other cases the root finding seems accurate enough
        #     assert vals[0] == 0 and vals[1] == 0
        #     assert len(self.roots) == num_triangles
        #     # we know what the root should be in that case
        #     root = 1 / vals[2]
        #     # the inaccuracies are quite large in that case...
        #     # for r in self.roots:
        #     #     assert np.isclose(r, root), f"root {r} is not close to {root}"
        #     self.roots = np.array([root for _ in range(num_triangles)])
        # else:
        #     self.roots = self.roots.real
        # # self.roots = self.roots.real

        orig_roots = self.roots.copy()
        # self.roots = np.array([r.real + r.imag for r in self.roots])
        self.roots = np.abs(self.roots)

        self.roots.sort()
        self.roots = self.roots[::-1]

        self.eps2 = 1 / self.roots
        # print(self.eps2)
        self.eps = np.sqrt(self.eps2)

        self.lagrange = []
        for i in range(num_triangles):
            prod = 1.0
            for j in range(num_triangles):
                if i == j:
                    continue
                prod *= -self.eps2[j] / (self.eps2[i] - self.eps2[j])
                if self.eps2[i] == self.eps2[j]:
                    print(
                        "Warning: eps2[i] == eps2[j], this may cause numerical instability."
                    )
                    print(self.eps2[i], self.eps2[j])
                    print(self.roots[i], self.roots[j])
                    print(orig_roots[i], orig_roots[j])
                    print(orig_roots)
                    print(alt_roots)

                    import sympy
                    x = sympy.symbols("x")
                    p_sympy = sympy.Poly(self.p[::-1], x)
                    roots_sympy = sympy.roots(p_sympy)
                    print("Sympy roots:", roots_sympy)

                    assert False
            self.lagrange.append(prod)
        # print(self.lagrange)

        # fendley_gap_direct = 1.0 / np.sqrt(self.roots[-1])

        num_majoranas = 2 * num_triangles

        # print(self.roots)
        # print("eps:", self.eps)

        self.majorana_matrix = np.zeros((num_majoranas, num_majoranas))
        for i, eps in enumerate(self.eps):
            self.majorana_matrix[2 * i, 2 * i + 1] = -eps
            self.majorana_matrix[2 * i + 1, 2 * i] = eps

        lm, _ = skew_diagonalise(self.majorana_matrix)
        smoothen_lamda(lm)
        lm_pairs = get_lamda_pairs(lm)
        min_abs_lm = min(abs(x) for _, x in lm_pairs)

        self.gap = min_abs_lm


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
