"""Note 11: Milnor's two drums with the same tones.

What it computes
  (1) The theta series of the lattices E8 + E8 and D16+ (the integer vectors of R^16 with even coordinate sum, and
      their translate by (1/2, ..., 1/2)), up to squared length 2 * MMAX, from the one-variable theta functions:
      the even-sum vectors of Z^n are counted by (theta_3^n + theta_4^n)/2 and the translate by theta_2^n / 2.
      The counts for E8 to squared length 4 are confirmed by listing the vectors.
  (2) Jacobi's identity theta_3^4 = theta_2^4 + theta_4^4 on the same range, and the polynomial identity
      ((a^2 + (a + b)^2 + b^2)/2)^2 = (a^4 + (a + b)^4 + b^4)/2 on a 5 x 5 grid, which proves it.
  (3) The roots (vectors of squared length 2) of both lattices: those of E8 from the listing in (1), those of D16+
      as +-e_i +- e_j, each set checked to be distinct lattice vectors as many as the theta series counts; and the
      graph on the roots that joins two roots when their inner product is non-zero.

What it shows
  The two theta series agree (Witt): both are the Eisenstein series 1 + 480 sum sigma_7(m) q^m of weight 8, the
  only modular form of weight 8 for SL(2, Z) with constant term 1. The flat tori R^16 / (E8 + E8) and
  R^16 / D16+ therefore have the same Laplace eigenvalues with the same multiplicities (Milnor). Both lattices
  have 480 roots, but the root graph of E8 + E8 has two components and that of D16+ has one; an isometry maps
  roots to roots and keeps inner products, so the lattices, and the tori, are not isometric.
"""
import itertools

import numpy as np

from common import check

MMAX = 30                      # squared lengths up to 2 * MMAX
E = 8 * MMAX                   # series in x = q^(1/8), q = e^(2 pi i tau): a vector v has exponent 4|v|^2 = 8 (|v|^2/2)


def series(terms):
    s = [0] * (E + 1)
    for e, c in terms:
        if e <= E:
            s[e] += c
    return s


def mul(a, b):
    out = [0] * (E + 1)
    for i, ai in enumerate(a):
        if ai:
            for j in range(E + 1 - i):
                if b[j]:
                    out[i + j] += ai * b[j]
    return out


def power(a, n):
    out = series([(0, 1)])
    for _ in range(n):
        out = mul(out, a)
    return out


def add(*ss):
    return [sum(c) for c in zip(*ss)]


def half(s):
    assert all(c % 2 == 0 for c in s)
    return [c // 2 for c in s]


R = range(-MMAX, MMAX + 1)
th3 = series((4 * n * n, 1) for n in R)
th4 = series((4 * n * n, 1 - 2 * (n % 2)) for n in R)
th2 = series(((2 * n + 1) ** 2, 1) for n in R)

e8 = half(add(power(th2, 8), power(th3, 8), power(th4, 8)))       # D8 and its translate
d16p = half(add(power(th2, 16), power(th3, 16), power(th4, 16)))  # D16 and its translate
e8e8 = mul(e8, e8)


def coeffs(s):
    """Coefficient of q^m, m = |v|^2 / 2, for m = 0..MMAX."""
    assert all(c == 0 for e, c in enumerate(s) if e % 8)
    return [s[8 * m] for m in range(MMAX + 1)]


def sigma(k, m):
    return sum(d ** k for d in range(1, m + 1) if m % d == 0)


# (1) the counts, and a direct listing of E8 to squared length 4
ints = [v for v in itertools.product(range(-2, 3), repeat=8) if sum(v) % 2 == 0]
halves = [v for v in itertools.product((-1.5, -0.5, 0.5, 1.5), repeat=8) if (sum(v) - 4) % 2 == 0]
listed = [sum(1 for v in ints + halves if sum(x * x for x in v) == n) for n in (2, 4)]
check(coeffs(e8)[1:3] == listed == [240, 2160],
      "E8 has 240 vectors of squared length 2 and 2160 of squared length 4, by formula and by listing")
check(coeffs(e8) == [1] + [240 * sigma(3, m) for m in range(1, MMAX + 1)],
      f"theta of E8 = 1 + 240 sum sigma_3(m) q^m (Eisenstein series of weight 4) for m <= {MMAX}")
check(coeffs(e8e8) == coeffs(d16p),
      f"theta of E8 + E8 = theta of D16+ for every squared length up to {2 * MMAX}")
check(coeffs(d16p) == [1] + [480 * sigma(7, m) for m in range(1, MMAX + 1)],
      "both are 1 + 480 sum sigma_7(m) q^m: 480 roots, 61920 vectors of squared length 4, "
      f"{coeffs(d16p)[3]} of squared length 6")

# (2) why: Jacobi's identity and one line of algebra
check(power(th3, 4) == add(power(th2, 4), power(th4, 4)),
      f"Jacobi: theta_3^4 = theta_2^4 + theta_4^4 to x^{E}")
check(all((a * a + (a + b) ** 2 + b * b) ** 2 == 2 * (a ** 4 + (a + b) ** 4 + b ** 4)
          for a in range(5) for b in range(5)),
      "with a = theta_2^4, b = theta_4^4: theta_E8^2 = theta_D16+ = (a^2 + ab + b^2)^2 (degree 4, checked on 5 x 5)")

# (3) the roots tell the two lattices apart
e8_roots = [np.array(v, dtype=float) for v in ints + halves if sum(x * x for x in v) == 2]
z8 = np.zeros(8)
roots_e8e8 = np.array([np.concatenate([r, z8]) for r in e8_roots] + [np.concatenate([z8, r]) for r in e8_roots])
roots_d16 = np.array([np.eye(16)[i] * s + np.eye(16)[j] * t
                      for i, j in itertools.combinations(range(16), 2) for s in (1, -1) for t in (1, -1)])


def components(roots):
    joined = np.abs(roots @ roots.T) > 1e-9
    seen, count = np.zeros(len(roots), bool), 0
    for start in range(len(roots)):
        if seen[start]:
            continue
        count += 1
        stack = [start]
        seen[start] = True
        while stack:
            for k in np.flatnonzero(joined[stack.pop()] & ~seen):
                seen[k] = True
                stack.append(k)
    return count


def in_dn_plus(v):
    """D_n+ (n = 8: E8; n = 16: D16+): all coordinates integers or all in Z + 1/2, with even sum."""
    w = np.round(2 * v)
    return np.allclose(2 * v, w) and len(set(w % 2)) == 1 and round(v.sum()) % 2 == 0


def in_e8e8(v):
    return in_dn_plus(v[:8]) and in_dn_plus(v[8:])


for name, roots, member in (("E8 + E8", roots_e8e8, in_e8e8), ("D16+", roots_d16, in_dn_plus)):
    check(len({tuple(r) for r in roots}) == len(roots) == coeffs(d16p)[1] == 480
          and np.allclose((roots ** 2).sum(1), 2) and all(member(r) for r in roots),
          f"{name}: 480 distinct lattice vectors of squared length 2, all the roots the theta series counts")
check(components(roots_e8e8) == 2 and components(roots_d16) == 1,
      "root graph: two components for E8 + E8, one for D16+, so the lattices are not isometric")
