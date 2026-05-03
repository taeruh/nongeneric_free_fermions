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
