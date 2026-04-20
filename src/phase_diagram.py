import numpy as np
from scipy import linalg
from numpy.typing import NDArray
from sage.all import Graph


def skew_diagonalise(
    h: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """return (Lambda, K) such that h = K lambda K^dagger"""
    lm, km = linalg.schur(h, output="real")  # pyright: ignore
    assert np.allclose(h, km @ lm @ km.T)
    return lm, km  # pyright: ignore


def smoothen_lamda(lm: NDArray[np.float64], eps: float = 1e-13) -> None:
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


def triangle_grid(n, factor=3):
    points = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            alpha = i / n
            beta = j / n
            gamma = k / n
            points.append([alpha, beta, gamma])
    return np.array(points) * factor

def triangle_grid_inner(n, factor=3):
    points = triangle_grid(n, factor)
    to_remove = []
    for i, point in enumerate(points):
        if point[0] == 0 or point[1] == 0 or point[2] == 0:
            to_remove.append(i)
    return np.delete(points, to_remove, axis=0)


def points_to_plot_coordinates(points):
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.5, np.sqrt(3) / 2])
    xy = points[:, 0, None] * a + points[:, 1, None] * b + points[:, 2, None] * c
    return xy[:, 0], xy[:, 1]
