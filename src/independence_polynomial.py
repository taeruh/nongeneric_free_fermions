import numpy as np
from sage.all import Graph
from numpy.typing import NDArray


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


def truncate(poly: NDArray[np.float64]) -> NDArray[np.float64]:
    ret = poly.copy()
    while True:
        if ret[-1] == 0:
            ret = ret[:-1]
        else:
            break
    return ret


def reverse_polynomial(poly: NDArray[np.float64]) -> NDArray[np.float64]:
    return poly[::-1]
