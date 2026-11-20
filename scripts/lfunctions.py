"""Fast evaluation of zeta and of Dirichlet L-functions on the critical line, vectorised with numpy.

L(s, chi) = q^(-s) sum_a chi(a) zeta(s, a/q), with the Hurwitz zeta function summed by the Euler-Maclaurin
formula. Hardy's Z(t) = Re(e^(i theta(t)) L(1/2 + it, chi)) is real for a real primitive character, and its
sign changes are the zeros on the critical line. The scripts that import this module check it against mpmath.
"""
import math

import numpy as np

TWO_PI = 2 * math.pi
# B_2k / (2k)! for k = 1..6
_EM = [1 / 12, -1 / 720, 1 / 30240, -1 / 1209600, 1 / 47900160, -691 / 1307674368000]


def loggamma(z):
    """log Gamma(z) for a complex array with Re z > 0: eight steps of the recurrence, then Stirling's series."""
    z = np.asarray(z, dtype=complex)
    shift = np.zeros(z.shape, dtype=complex)
    w = z.copy()
    for _ in range(8):
        shift += np.log(w)
        w = w + 1
    stirling = ((w - 0.5) * np.log(w) - w + 0.5 * math.log(TWO_PI)
                + 1 / (12 * w) - 1 / (360 * w ** 3) + 1 / (1260 * w ** 5) - 1 / (1680 * w ** 7))
    return stirling - shift


def hurwitz_zeta(s, alpha, terms):
    """zeta(s, alpha) for a complex array s (Re s = 1/2) and 0 < alpha <= 1, by Euler-Maclaurin with `terms`
    leading terms; accurate to about 1e-10 when terms >= |s|."""
    s = np.asarray(s, dtype=complex)
    out = np.zeros(s.shape, dtype=complex)
    logs = np.log(np.arange(terms) + alpha)
    for i in range(0, len(s), 2000):
        out[i:i + 2000] = np.exp(-np.outer(s[i:i + 2000], logs)).sum(axis=1)
    m = terms + alpha
    lm = math.log(m)
    out += np.exp((1 - s) * lm) / (s - 1) + 0.5 * np.exp(-s * lm)
    rising = s.copy()                          # s (s+1) ... (s + 2k - 2)
    for k, b in enumerate(_EM, start=1):
        out += b * rising * np.exp(-(s + 2 * k - 1) * lm)
        rising = rising * (s + 2 * k - 1) * (s + 2 * k)
    return out


def dirichlet_L(t, q, chi):
    """L(1/2 + it, chi) for a real array t; chi is the list of values chi(0), ..., chi(q - 1)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    s = 0.5 + 1j * t
    terms = int(max(60, np.max(np.abs(t)) + 20))
    total = np.zeros(t.shape, dtype=complex)
    for a in range(1, q + 1):
        c = chi[a % q]
        if c:
            total += c * hurwitz_zeta(s, a / q, terms)
    return np.exp(-s * math.log(q)) * total


def theta(t, q, parity):
    """The phase of the gamma factor (q / pi)^((s + a)/2) Gamma((s + a)/2) at s = 1/2 + it."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return loggamma((0.5 + parity + 1j * t) / 2).imag + t / 2 * math.log(q / math.pi)


def hardy_Z(t, q, chi, parity):
    """Hardy's Z-function of L(s, chi); for q = 1 and chi = [1] it is that of zeta."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return (np.exp(1j * theta(t, q, parity)) * dirichlet_L(t, q, chi)).real


def mean_count(t, q, parity):
    """The smooth part of the number of zeros with 0 < gamma <= t: theta(t)/pi, plus 1 for zeta."""
    return theta(t, q, parity) / math.pi + (1.0 if q == 1 else 0.0)


def find_zeros(f, t_min, t_max, step, iterations=40):
    """The sign changes of f on a grid of the given step, refined by bisection."""
    grid = np.arange(t_min, t_max, step)
    values = f(grid)
    idx = np.nonzero(values[:-1] * values[1:] < 0)[0]
    lo, hi, flo = grid[idx].copy(), grid[idx + 1].copy(), values[idx].copy()
    for _ in range(iterations):
        mid = (lo + hi) / 2
        fm = f(mid)
        left = flo * fm <= 0
        hi = np.where(left, mid, hi)
        flo = np.where(left, flo, fm)
        lo = np.where(left, lo, mid)
    return (lo + hi) / 2
