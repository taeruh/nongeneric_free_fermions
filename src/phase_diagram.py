import numpy as np
from numpy.typing import NDArray
from sage.all import Graph


def independence_polynomial(
    graph: Graph, weights: list[np.float64], max_alpha_plus_1: int
) -> NDArray[np.float64]:
    # recursive definition of the independence polynomial:
    # I(G, X) = I(G\{v}, X) + I(G\N[v], X) * X
    if graph.order() == 0:
        ret = np.zeros(max_alpha_plus_1, dtype=np.float64)
        ret[0] = np.float64(1.0)
        return ret
    else:
        v = graph.vertices()[0]
        g_v = graph.copy()
        g_v.delete_vertex(v)
        g_nv = g_v.copy()
        for neighbor in graph.neighbors(v):
            g_nv.delete_vertex(neighbor)
        poly_v = independence_polynomial(g_v, weights, max_alpha_plus_1)
        poly_nv = independence_polynomial(g_nv, weights, max_alpha_plus_1)
        ret = np.zeros(max_alpha_plus_1, dtype=np.float64)
        poly_nv = poly_nv * weights[v] ** 2
        ret[0] = poly_v[0]
        for i in range(1, max_alpha_plus_1):
            ret[i] = poly_v[i] + poly_nv[i - 1]
        return ret


def truncate_and_reverse_polynomial(poly: NDArray[np.float64]) -> NDArray[np.float64]:
    ret = poly.copy()
    while True:
        if ret[-1] == 0:
            ret = ret[:-1]
        else:
            break
    ret = ret[::-1]
    return ret


def multiply_polynomials(
    poly1: NDArray[np.float64], poly2: NDArray[np.float64]
) -> NDArray[np.float64]:
    deg1 = len(poly1) - 1
    deg2 = len(poly2) - 1
    result = np.zeros(deg1 + deg2 + 1, dtype=np.float64)
    for i in range(deg1 + 1):
        for j in range(deg2 + 1):
            result[i + j] += poly1[i] * poly2[j]
    return result


def roots(poly: NDArray[np.float64]) -> NDArray[np.float64]:
    roots = np.roots(poly)
    print(roots)
    return roots
