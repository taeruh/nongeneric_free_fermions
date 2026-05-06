import numpy as np
import scipy
from scipy import linalg
from numpy.typing import NDArray
from sage.all import Graph


class Wolfram:
    """Preferably use it as context, e.g. `with Wolfram() as wolfram: ...` so that the
    session is properly closed even if exceptions occur."""

    def __init__(
        self,
        kernel_path: str = "/usr/local/Wolfram/Wolfram/14.3/Executables/WolframKernel",
    ):
        from wolframclient.evaluation import WolframLanguageSession

        print("Starting Wolfram session...")
        self.session = WolframLanguageSession(kernel_path)
        self.is_running = True

        print("Loading Wolfram code")
        self.session.evaluate("""
        ClearALL["Coefficients"]
        ClearALL["ComputeEpsilons"]
        Coefficients[n_Integer, a_, b_, c_] := Module[
          {
           buf = {{}, {}, {1}},
           p, pm1, pm2, pm3, f1, f2, f3, i, pmax
           },
          f1 = a^2 + b^2 + c^2;
          f2 = (a b)^2 + (a c)^2 + (b c)^2;
          f3 = (a b c)^2;
          Do[
           pm1 = buf[[Mod[i - 2, 3] + 1]];
           pm2 = buf[[Mod[i - 3, 3] + 1]];
           pm3 = buf[[Mod[i - 4, 3] + 1]];
           pmax = i + 1;
           p = Table[0, pmax];
           Do[p[[j]] += pm1[[j]], {j, Length[pm1]}];
           Do[p[[j + 1]] += f1 pm1[[j]], {j, Length[pm1]}];
           Do[p[[j + 2]] -= f2 pm2[[j]], {j, Length[pm2]}];
           Do[p[[j + 3]] += f3 pm3[[j]], {j, Length[pm3]}];
           buf[[Mod[i - 1, 3] + 1]] = p;
           , {i, 1, n}
           ];
          poly = buf[[Mod[n - 1, 3] + 1]];
          poly3 = buf[[Mod[n - 2, 3] + 1]];
          <|"Poly" -> poly, "Poly3" -> poly3|>
          ]
        ComputeEpsilons[n_Integer, a_, b_, c_] := Module[
          {x, coeffs, polycoeffs, poly3coeffs, unsortedroots, roots, negroots, eps2, 
           eps, poly3, pmfactors},
          coeffs = Coefficients[n, a, b, c];
          polycoeffs = coeffs["Poly"];
          poly3coeffs = coeffs["Poly3"];
          unsortedroots = 
          x /. Solve[
              Sum[polycoeffs[[i + 1]] x^i, {i, 0, Length[polycoeffs] - 1}] == 0, x
          ];
          roots = Sort[unsortedroots];
          negroots = -roots;
          eps2 = 1/negroots;
          eps = Sqrt[eps2];
          poly3[x_] := FromDigits[Reverse[poly3coeffs], x];
          pmfactors = poly3 /@ roots;
          pmfactors = N[pmfactors, {1000, 1000}];
          <|"Roots" -> roots, "NegRoots" -> negroots, "EpsilonSquared" -> eps2,
            "Epsilons" -> eps, "PMFactors" -> pmfactors, "Poly" -> polycoeffs,
            "Poly3" -> poly3coeffs|>
          ]
        """)

        print("Wolfram session is ready.")
        # poly3[x_] :=
        #  Sum[poly3coeffs[[i + 1]] x^i, {i, 0, Length[poly3coeffs] - 1}];

        # the following gives me earlier complex values
        #  x /. NSolve[
        #    Sum[polycoeffs[[i + 1]] x^i, {i, 0, Length[polycoeffs] - 1}] == 0,
        #    x,
        #    WorkingPrecision -> 100
        # ];
        #
        # trying the following to get rid of degeneracies, doesn't work, it just gives me
        # complex values when using Solve (NSolve has them anyways)
        # coeffs = SetPrecision[coeffs / Max[Abs[coeffs]], 100];

        # <|"Roots" -> roots, "EpsilonSquared" -> eps2, "Epsilons" -> eps,
        #  "PMFactors" -> pmfactors|>
        # <|"Roots" -> N[roots, 50], "EpsilonSquared" -> N[eps2, 50],
        #  "Epsilons" -> N[eps, 50], "PMFactors" -> N[pmfactors, 50]|>

    def close_session(self):
        if self.is_running:
            print("Terminating Wolfram session...")
            self.session.terminate()
            self.is_running = False
            print("Wolfram session terminated.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pyright: ignore
        self.close_session()
        return False  # re-raise exceptions

    # seems to not be called on exceptions, e.g., assertion failure
    def __del__(self):
        self.close_session()

    def compute_epsilons(self, num_triangles: int, a: float, b: float, c: float):
        result = self.session.evaluate(
            f"ComputeEpsilons[{num_triangles}, {a}, {b}, {c}]"
        )
        return result


import time


class Calculation:
    def __init__(
        self,
        num_triangles: int,
        alpha: float,
        beta: float,
        gamma: float,
        wolfram: Wolfram,
    ):
        self.num_triangles = num_triangles
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.wolfram_results = wolfram.compute_epsilons(
            num_triangles, alpha, beta, gamma
        )
        self.roots = np.array(
            [np.float128(r) for r in list(self.wolfram_results["Roots"])]
        )
        self.negroots = np.array(
            [np.float128(r) for r in list(self.wolfram_results["NegRoots"])]
        )
        self.eps2 = np.array(
            [np.float128(r) for r in list(self.wolfram_results["EpsilonSquared"])]
        )
        self.eps = np.array(
            [np.float128(r) for r in list(self.wolfram_results["Epsilons"])]
        )
        self.pm_factors = np.array(
            [np.float128(r) for r in list(self.wolfram_results["PMFactors"])]
        )
        self.poly = np.array(
            [np.float128(r) for r in list(self.wolfram_results["Poly"])]
        )
        self.poly3 = np.array(
            [np.float128(r) for r in list(self.wolfram_results["Poly3"])]
        )

        # print(self.poly)
        # print(self.poly3)
        # print(self.roots)
        # print(self.negroots)
        # print(self.eps2)
        # print(self.eps)
        # print(self.pm_factors)
        # print(self.wolfram_results["PMFactors"])

        self.fendley_gap_direct = float(self.wolfram_results["Epsilons"][0])
        self.num_majoranas = 2 * self.num_triangles

        self.fendley_h_matrix = np.zeros(
            (self.num_majoranas, self.num_majoranas), dtype=np.float128
        )
        self.fendley_norm = 0.0
        for i, eps in enumerate(self.eps):
            self.fendley_h_matrix[2 * i, 2 * i + 1] = -eps
            self.fendley_h_matrix[2 * i + 1, 2 * i] = eps
            self.fendley_norm += 2 * abs(eps)
        self.h_matrix = self.fendley_h_matrix.copy()

    def calculate_hl_matrices(self):
        if self.num_triangles % 2 == 0:
            self.num_currents = self.num_triangles // 2
        else:
            self.num_currents = (self.num_triangles + 1) // 2

        self.lagrange = []
        self.effective_norm = []
        for i in range(self.num_triangles):
            lagrange = 1.0
            for j in range(self.num_triangles):
                if i == j:
                    continue
                lagrange *= self.roots[j] / (self.roots[j] - self.roots[i])
                if self.roots[i] == self.roots[j]:
                    print(
                        "Warning: roots[i] == roots[j], this may cause numerical instability."
                    )
                    print(self.roots[i], self.roots[j])
                    print(self.wolfram_results["Roots"])
                    assert False
            self.lagrange.append(lagrange)
            # print(self.pm_factors[i] * lagrange)
            # print(np.sign(self.pm_factors[i] * lagrange), "should be +1")
            # print(i)
            # print(self.pm_factors[i], lagrange)
            assert np.sign(self.pm_factors[i] * lagrange) == 1
            effective_norm = (
                4 * np.sqrt(np.abs(self.pm_factors[i])) * np.sqrt(np.abs(lagrange))
            )
            # effective_norm = (
            #     16 * np.abs(self.pm_factors[i])
            # )
            # effective_norm = (
            #     16 * self.pm_factors[i]
            # )
            self.effective_norm.append(effective_norm)
        print("Effective norms:", self.effective_norm)

        # print(self.lagrange)
        # print(self.norm)

        mus = []
        for l in range(self.num_currents):
            l = 1 + 2 * l
            mul = np.zeros((self.num_triangles, self.num_triangles), dtype=np.float128)
            for m in range(self.num_triangles):
                for n in range(self.num_triangles):
                    mu = (
                        (-1) ** (((l - 1) // 2) % 2)
                        / 8
                        * self.effective_norm[m]
                        * self.effective_norm[n]
                    )
                    eps_sum = 0
                    for i in range(l):
                        if i % 2 == 0:
                            eps_sum += self.eps[m] ** i * self.eps[n] ** (l - i)
                        else:
                            eps_sum += self.eps[n] ** i * self.eps[m] ** (l - i)
                    mul[m, n] = mu * eps_sum
            mus.append(mul)

        self.hl_matrices = []
        self.hl_norms = []
        for l in range(self.num_currents):
            hl = np.zeros((self.num_majoranas, self.num_majoranas), dtype=np.float128)
            hl_norm = 0.0
            for m in range(1, self.num_triangles + 1):
                for n in range(1, self.num_triangles + 1):
                    mu = mus[l][m - 1, n - 1]
                    hl[(2 * m - 1) - 1, (2 * n) - 1] = mu
                    hl[(2 * n) - 1, (2 * m - 1) - 1] = -mu
                    hl_norm += 2 * np.abs(mu)
            self.hl_matrices.append(hl)
            self.hl_norms.append(hl_norm)

        # for h, norm in zip(self.hl_matrices, self.hl_norms):
        #     h = h / norm
        #     norm = 0.0
        #     for i in range(self.num_majoranas):
        #         for j in range(self.num_majoranas):
        #             norm += abs(h[i, j])
        #     print(norm)
        # h = self.fendley_h_matrix / self.fendley_norm
        # norm = 0.0
        # for i in range(self.num_majoranas):
        #     for j in range(self.num_majoranas):
        #         norm += abs(h[i, j])
        # print(norm)

    def extend_model(self, currents_weights: list[float], fendley_weight: float):
        """
        the original fendley h matrix and the currents hl matrices are both normalized;
        fendley_weight is None this normalization is automatically undone for the fendley
        h matrix (i.e., effectively it sets fendley_weight to self.fendley_h_norm)
        """
        assert self.num_currents == len(currents_weights)
        norm = self.fendley_norm
        if norm == 0:
            norm = 1.0
        self.h_matrix = fendley_weight * self.fendley_h_matrix / norm
        for l in range(self.num_currents):
            norm = self.hl_norms[l]
            if norm == 0:
                norm = 1.0
            self.h_matrix += currents_weights[l] * self.hl_matrices[l] / norm

    def compute_gap(self):
        norm = np.linalg.norm(self.h_matrix)
        if abs(norm) == 0:
            norm = 1.0
        majorana_matrix_normalized = self.h_matrix / norm
        lm, _ = skew_diagonalise(majorana_matrix_normalized)
        smoothen_lamda(lm)
        lm_pairs = get_lamda_pairs(lm)
        if len(lm_pairs) == 0:
            self.gap = 0.0
            return
        min_abs_lm = min(abs(x) for _, x in lm_pairs)
        self.gap = min_abs_lm * norm


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


# same as triangle_grid, i.e., alpha, beta, gamma add up to factor, but only include
# points where each of alpha, beta, gamma is at least minimum_bound. This is to avoid
# numerical instability when one of them is close to zero.
def triangle_grid_with_minimum_bound(
    num_samples: int, factor: float = 3, minimum_bound: float = 0.1
):
    points = []
    for i in range(num_samples + 1):
        for j in range(num_samples + 1 - i):
            k = num_samples - i - j
            alpha = i / num_samples * (factor - 3 * minimum_bound) + minimum_bound
            beta = j / num_samples * (factor - 3 * minimum_bound) + minimum_bound
            gamma = k / num_samples * (factor - 3 * minimum_bound) + minimum_bound
            points.append([alpha, beta, gamma])
    return np.array(points)


def triangle_grid_no_edge(num_samples: int, factor: float = 3):
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
