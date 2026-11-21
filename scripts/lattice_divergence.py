"""Note 12: the log-determinant divergence of two lattice metrics, the cosmological-term functional of the
theory of gravity from entropy evaluated on lattices.

What it computes
  (1) The functional. For positive definite n x n matrices A, B, S(A, B) = tr(G - 1 - log G) = tr G - n - log det G,
      G = A B^(-1), is twice the Kullback-Leibler divergence KL(N(0, A) || N(0, B)) (n = 2 and n = 3, exact
      Gauss-Hermite quadrature). The function x - 1 - log x is non-negative and vanishes only at x = 1. For a
      harmonic trap with thermal covariance A switched instantly to covariance B, the mean dissipated work
      <W> - Delta F is kT S(A, B)/2.
  (2) Lattice metrics. M(z) = (1/y) [[x^2 + y^2, x], [x, 1]] is the Gram matrix of the basis (z, 1) of Z + Z z divided
      by the covolume y; a change of basis by gamma in SL(2, Z) gives gamma M(z) gamma^T = M(gamma z). For
      G = M(z) M(w)^(-1): det G = 1, tr G = 2 cosh d(z, w), eigenvalues e^(+-d), S = 2 cosh d - 2 = 4 sinh^2(d/2)
      = d^2 + d^4/12 + O(d^6); the second-order term is (dx^2 + dy^2)/y^2, the Fisher-Rao metric of N(0, M(z)).
      For s M(z): S = 2 s cosh d - 2 - 2 log s, convex in s with minimum 2 log cosh d at s = 1/cosh d.
      The square and hexagonal bases: S(M(i), M(rho)) = 2(2/sqrt 3 - 1), the least value of S(M(i), M(gamma rho)).
  (3) A change of basis. S(M(z), M(gamma z)) = t^2 - 4 + (c(x^2 + y^2) + (d - a)x - b)^2 / y^2 for
      gamma = (a b; c d) of trace t: at least t^2 - 4, with equality on the axis when |t| > 2; least hyperbolic
      value 5, at (2 1; 1 1) and z = i. For (0 -1; 1 0): (x^2 + y^2 + 1)^2 / y^2 - 4; for (1 1; 0 1): 1/y^2.
  (4) Dirichlet's class number formula. For t = 3, ..., 11: t^2 - 4 = D f^2 with D fundamental, the fundamental unit
      eps_D (from the least solution of a^2 - D b^2 = +-4), k with (t + sqrt(t^2 - 4))/2 = eps_D^k, and the class
      number h(D) from the cycles of reduced indefinite forms; then L(1, chi_D) sqrt(S_gamma) / l = h(D) f / k,
      with S_gamma = t^2 - 4 and l = 2 arccosh(t/2) = 2 log eps_t.

What it shows
  On flat lattice metrics the functional tr(G - 1 - log G) of the emergent cosmological term is a function of
  hyperbolic distance; under the modular group its minimum is a discriminant, and Dirichlet's formula expresses
  the ratio of the geodesic length to the square root of that minimum through L(1, chi_D).

Scope
  Two flat metrics on a two-dimensional lattice; the functional is evaluated, not varied.
"""
import math
from fractions import Fraction as F

import matplotlib.pyplot as plt
import numpy as np

from common import BLUE, GRID, INK, INK_2, ORANGE, check, save

rng = np.random.default_rng(20261121)
PHI = (1 + math.sqrt(5)) / 2
RHO = complex(-0.5, math.sqrt(3) / 2)


def M(z):
    """the Gram matrix of the basis (z, 1) of Z + Z z divided by the covolume Im z: determinant one."""
    x, y = z.real, z.imag
    return np.array([[x * x + y * y, x], [x, 1.0]]) / y


def hyp(z, w):
    return math.acosh(1 + abs(z - w) ** 2 / (2 * z.imag * w.imag))


def S(A, B):
    """tr(G - 1 - log G) for G = A B^(-1), from the eigenvalues of G."""
    lam = np.linalg.eigvals(A @ np.linalg.inv(B)).real
    return float(np.sum(lam - 1 - np.log(lam)))


def act(g, z):
    return (g[0][0] * z + g[0][1]) / (g[1][0] * z + g[1][1])


# ---------------------------------------------------------------------------------------------------
print("(1) the functional")
worst = 0.0
for n in (2, 3):
    nodes, weights = np.polynomial.hermite_e.hermegauss(4)     # exact for polynomials of degree <= 7
    weights = weights / weights.sum()
    grid = np.array(np.meshgrid(*([nodes] * n), indexing="ij")).reshape(n, -1).T
    wgrid = np.prod(np.array(np.meshgrid(*([weights] * n), indexing="ij")).reshape(n, -1).T, axis=1)
    for _ in range(100):
        P, Q = rng.normal(size=(n, n)), rng.normal(size=(n, n))
        A, B = P @ P.T + 0.3 * np.eye(n), Q @ Q.T + 0.3 * np.eye(n)
        L, Ai, Bi = np.linalg.cholesky(A), np.linalg.inv(A), np.linalg.inv(B)
        xs_ = grid @ L.T
        logratio = (0.5 * math.log(np.linalg.det(B) / np.linalg.det(A))
                    - 0.5 * np.einsum("ij,jk,ik->i", xs_, Ai, xs_) + 0.5 * np.einsum("ij,jk,ik->i", xs_, Bi, xs_))
        kl = float(wgrid @ logratio)
        worst = max(worst, abs(2 * kl - S(A, B)) / (1 + S(A, B)))
check(worst < 1e-10, "tr(G - 1 - log G), G = A B^(-1), is 2 KL(N(0, A) || N(0, B)) for n = 2 and n = 3 "
                     f"(100 random pairs each, quadrature exact; relative error {worst:.0e})")
xs = np.concatenate([np.linspace(0.01, 0.99, 500), np.linspace(1.01, 20, 500)])
check(np.all(xs - 1 - np.log(xs) > 0), "x - 1 - log x > 0 for x != 1 (1000 points in (0, 20)), and = 0 at x = 1")

kT = 1.0
worst = 0.0
for _ in range(200):
    P, Q = rng.normal(size=(2, 2)), rng.normal(size=(2, 2))
    A, B = P @ P.T + 0.3 * np.eye(2), Q @ Q.T + 0.3 * np.eye(2)       # thermal covariances of two traps
    KA, KB = kT * np.linalg.inv(A), kT * np.linalg.inv(B)               # stiffness matrices
    work = 0.5 * np.trace((KB - KA) @ A)                                 # <H_B - H_A> in equilibrium of A
    dF = 0.5 * kT * math.log(np.linalg.det(KB) / np.linalg.det(KA))
    worst = max(worst, abs((work - dF) - kT * S(A, B) / 2) / (1 + S(A, B)))
check(worst < 1e-12, "a harmonic trap with thermal covariance A switched instantly to covariance B: mean work minus "
                     f"free-energy change = kT S(A, B)/2 (200 random pairs; {worst:.0e})")

# ---------------------------------------------------------------------------------------------------
print("(2) two lattice metrics")
worst = 0.0
for _ in range(2000):
    z = complex(rng.normal(), abs(rng.normal()) + 0.05)
    w = complex(rng.normal(), abs(rng.normal()) + 0.05)
    G = M(z) @ np.linalg.inv(M(w))
    d = hyp(z, w)
    lam = np.sort(np.linalg.eigvals(G).real)
    worst = max(worst, abs(np.linalg.det(G) - 1), abs(np.trace(G) - 2 * math.cosh(d)) / math.cosh(d),
                abs(lam[1] - math.exp(d)) / math.exp(d), abs(lam[0] - math.exp(-d)) / math.exp(-d),
                abs(S(M(z), M(w)) - (2 * math.cosh(d) - 2)) / (1 + math.cosh(d)))
check(worst < 1e-8, "G = M(z) M(w)^(-1): det G = 1, tr G = 2 cosh d, eigenvalues e^d and e^(-d), "
                    f"S = 2 cosh d - 2 (2000 random pairs, relative error {worst:.0e})")
GEN = {"T": ((1, 1), (0, 1)), "t": ((1, -1), (0, 1)), "S": ((0, -1), (1, 0))}


def word(n):
    g = ((1, 0), (0, 1))
    for ch in rng.choice(list("TtS"), size=n):
        h = GEN[ch]
        g = tuple(tuple(sum(g[i][k] * h[k][j] for k in range(2)) for j in range(2)) for i in range(2))
    return g


worst = 0.0
for _ in range(300):
    g = word(int(rng.integers(1, 9)))
    z = complex(rng.normal(), abs(rng.normal()) + 0.2)
    gm = np.array(g, dtype=float)
    worst = max(worst, np.abs(gm @ M(z) @ gm.T - M(act(g, z))).max() / np.abs(M(z)).max())
check(worst < 1e-9, f"a change of basis gamma in SL(2, Z): gamma M(z) gamma^T = M(gamma z) (300 random words; {worst:.0e})")
check(all(abs((2 * math.cosh(d) - 2) - 4 * math.sinh(d / 2) ** 2) < 1e-12 for d in (0.1, 1.0, 3.0))
      and all(abs((2 * math.cosh(d) - 2 - d * d - d ** 4 / 12) / d ** 6 - 1 / 360) < 1e-3 for d in (0.01, 0.02)),
      "2 cosh d - 2 = 4 sinh^2(d/2) = d^2 + d^4/12 + d^6/360 + ...")
worst = 0.0
for _ in range(200):
    z = complex(rng.normal(), abs(rng.normal()) + 0.3)
    dz = complex(*rng.normal(size=2)) * 1e-5
    Mi = np.linalg.inv(M(z))
    dM = M(z + dz) - M(z)
    fisher = 0.5 * np.trace(Mi @ dM @ Mi @ dM)
    worst = max(worst, abs(fisher / (abs(dz) ** 2 / z.imag ** 2) - 1))
check(worst < 1e-3, "Fisher-Rao: (1/2) tr((M^(-1) dM)^2) = (dx^2 + dy^2)/y^2, the hyperbolic metric (200 random points)")
z, w = complex(0.3, 1.1), complex(-0.7, 0.4)
d = hyp(z, w)
ss = np.linspace(0.05, 3, 600)
vals = [S(s * M(z), M(w)) for s in ss]
check(max(abs(S(s * M(z), M(w)) - (2 * s * math.cosh(d) - 2 - 2 * math.log(s))) for s in (0.2, 1.0, 2.7)) < 1e-10
      and abs(min(vals) - 2 * math.log(math.cosh(d))) < 1e-4 and abs(ss[int(np.argmin(vals))] - 1 / math.cosh(d)) < 0.01,
      "areas in the ratio s: S = 2 s cosh d - 2 - 2 log s, minimum 2 log cosh d at s = 1/cosh d")
orbit = set()
for _ in range(4000):
    g = word(int(rng.integers(0, 12)))
    p = act(g, RHO)
    orbit.add((round(p.real, 9), round(p.imag, 9)))
least = min(S(M(1j), M(complex(*p))) for p in orbit)
check(abs(S(M(1j), M(RHO)) - 2 * (2 / math.sqrt(3) - 1)) < 1e-12 and abs(least - S(M(1j), M(RHO))) < 1e-7,
      f"S(M(i), M(rho)) = 2(2/sqrt 3 - 1) = {S(M(1j), M(RHO)):.10f}, the least value of S(M(i), M(gamma rho)) "
      f"over {len(orbit)} points of the orbit of rho")

# ---------------------------------------------------------------------------------------------------
print("(3) a change of basis")
worst, count = 0.0, 0
for _ in range(600):
    (a, b), (c, dd) = g = word(int(rng.integers(1, 9)))
    z = complex(rng.normal(), abs(rng.normal()) + 0.2)
    x, y = z.real, z.imag
    t = a + dd
    formula = t * t - 4 + (c * (x * x + y * y) + (dd - a) * x - b) ** 2 / y ** 2
    e = S(M(z), M(act(g, z)))
    worst = max(worst, abs(e - formula) / (1 + abs(formula)))
    count += e >= t * t - 4 - 1e-9
check(worst < 1e-8 and count == 600,
      "S(M(z), M(gamma z)) = t^2 - 4 + (c(x^2 + y^2) + (d - a)x - b)^2 / y^2 >= t^2 - 4 "
      f"(600 random words in S, T; relative error {worst:.0e})")
GOLD = ((2, 1), (1, 1))
axis = [complex(0.5 + math.sqrt(5) / 2 * math.cos(th), math.sqrt(5) / 2 * math.sin(th)) for th in (0.4, 1.1, 2.0, 2.9)]
check(all(abs(S(M(p), M(act(GOLD, p))) - 5) < 1e-9 for p in axis) and abs(S(M(1j), M(act(GOLD, 1j))) - 5) < 1e-12
      and abs(abs(1j - 0.5) - math.sqrt(5) / 2) < 1e-12,
      "(2 1; 1 1): S = 5 = t^2 - 4 on its axis, the semicircle from -1/phi to phi, which passes through i")
pts = [complex(rng.normal(), abs(rng.normal()) + 0.2) for _ in range(200)]
check(all(abs(S(M(p), M(act(GEN["S"], p))) - ((p.real ** 2 + p.imag ** 2 + 1) ** 2 / p.imag ** 2 - 4))
          < 1e-8 * (1 + S(M(p), M(act(GEN["S"], p)))) for p in pts)
      and all(abs(S(M(p), M(act(GEN["T"], p))) - 1 / p.imag ** 2) < 1e-8 * (1 + 1 / p.imag ** 2) for p in pts)
      and abs(S(M(1j), M(act(GEN["S"], 1j)))) < 1e-12 and abs(S(M(RHO), M(act(((0, -1), (1, 1)), RHO)))) < 1e-12,
      "(0 -1; 1 0): S = (x^2 + y^2 + 1)^2/y^2 - 4, zero only at i; (1 1; 0 1): S = 1/y^2; (0 -1; 1 1) at rho: 0 "
      "(200 random points)")

# ---------------------------------------------------------------------------------------------------
print("(4) Dirichlet's class number formula")


def kronecker_pos(D, n):
    """the Kronecker symbol (D/n) for a positive fundamental discriminant D and n >= 1."""
    r = 1
    while n % 2 == 0:
        n //= 2
        r *= 0 if D % 2 == 0 else (1 if D % 8 in (1, 7) else -1)
    a, m = D % n, n
    while a:
        while a % 2 == 0:
            a //= 2
            if m % 8 in (3, 5):
                r = -r
        a, m = m, a
        if a % 4 == 3 and m % 4 == 3:
            r = -r
        a %= m
    return r if m == 1 else 0


def L1_real(D):
    """L(1, chi_D) = -D^(-1/2) sum_{a < D} chi_D(a) log sin(pi a / D), for a positive fundamental discriminant D."""
    return -sum(kronecker_pos(D, a) * math.log(math.sin(math.pi * a / D)) for a in range(1, D)) / math.sqrt(D)


def is_fundamental(D):
    sqf = lambda m: all(m % (p * p) for p in range(2, math.isqrt(m) + 1))
    return (D % 4 == 1 and sqf(D)) or (D % 4 == 0 and (D // 4) % 4 in (2, 3) and sqf(D // 4))


def fundamental(Delta):
    """Delta = D f^2 with D a fundamental discriminant."""
    for f in range(math.isqrt(Delta), 0, -1):
        if Delta % (f * f) == 0 and is_fundamental(Delta // (f * f)):
            return Delta // (f * f), f
    raise ValueError(Delta)


def unit(D):
    """the fundamental unit (a + b sqrt D)/2 of discriminant D, from the least b > 0 with a^2 - D b^2 = +-4."""
    b = 1
    while True:
        for sgn in (-1, 1):
            a2 = D * b * b + 4 * sgn
            a = math.isqrt(a2) if a2 > 0 else -1
            if a > 0 and a * a == a2:
                return a, b, sgn
        b += 1


def unit_power(D, a, b, target):
    """the least k with ((a + b sqrt D)/2)^k = target, in exact arithmetic on pairs (A, B) = A/2 + (B/2) sqrt D."""
    A, B, k = a, b, 1
    while (A, B) != target:
        A, B = (A * a + D * B * b) // 2, (A * b + B * a) // 2
        k += 1
        if k > 50:
            raise ValueError(target)
    return k


def narrow_class_number(D):
    """the number of cycles of reduced primitive forms (a, b, c), b^2 - 4ac = D, 0 < b < sqrt D,
    sqrt D - b < 2|a| < sqrt D + b: the number of proper equivalence classes of forms of discriminant D."""
    r = math.sqrt(D)
    reduced = set()
    for b in range(1, math.isqrt(D) + 1):
        if (b * b - D) % 4 or b >= r:
            continue
        ac = (b * b - D) // 4
        for a in range(-D, D + 1):
            if a and ac % a == 0:
                c = ac // a
                if r - b < 2 * abs(a) < r + b and math.gcd(math.gcd(a, b), c) == 1:
                    reduced.add((a, b, c))

    def step(f):
        a, b, c = f
        m = 2 * abs(c)
        bn = (-b) % m
        while not (r - m < bn < r):
            bn += m
        return c, bn, (bn * bn - D) // (4 * c)

    seen, cycles = set(), 0
    for f in sorted(reduced):
        if f in seen:
            continue
        cycles += 1
        g = f
        while g not in seen:
            seen.add(g)
            g = step(g)
            if g not in reduced:
                raise ValueError(g)
    return cycles


rows = []
for t in range(3, 12):
    S_g = t * t - 4
    ell = 2 * math.acosh(t / 2)
    D, f = fundamental(S_g)
    a, b, sgn = unit(D)
    k = unit_power(D, a, b, (t, f))
    hplus = narrow_class_number(D)
    h = hplus if sgn == -1 else hplus // 2
    eps_t = (t + math.sqrt(S_g)) / 2
    ratio = L1_real(D) * math.sqrt(S_g) / ell
    rows.append((t, S_g, D, f, (a, b), k, h, F(h * f, k)))
    check(abs(ell - 2 * math.log(eps_t)) < 1e-12 and abs(S_g - (eps_t - 1 / eps_t) ** 2) < 1e-9
          and abs(ratio - h * f / k) < 1e-10,
          f"t = {t}: S = {S_g} = {D}*{f}^2, eps_D = ({a} + {b} sqrt {D})/2 (norm {sgn:+d}), k = {k}, h = {h}; "
          f"L(1, chi_{D}) sqrt(S)/l = {F(h * f, k)}")
check([r[-1] for r in rows] == [F(1, 2), F(1), F(1), F(1), F(3, 4), F(2), F(1), F(2), F(3, 2)]
      and abs(L1_real(5) - 2 * math.log(PHI) / math.sqrt(5)) < 1e-12,
      "the ratios h f / k for t = 3, ..., 11 are 1/2, 1, 1, 1, 3/4, 2, 1, 2, 3/2; L(1, chi_5) = 2 log(phi)/sqrt 5")

# ---------------------------------------------------------------------------------------------------
print("figures")
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.38})
ax = axes[0]
X, Y = np.meshgrid(np.linspace(-1.1, 2.1, 321), np.linspace(0.05, 1.6, 201))
Sg = 5 + ((X * X + Y * Y) - X - 1) ** 2 / Y ** 2           # (2 1; 1 1): c = 1, d - a = -1, b = 1
cs = ax.contourf(X, Y, np.minimum(Sg, 30), levels=[5, 5.5, 7, 10, 15, 22, 30], cmap="Blues", alpha=0.9)
th = np.linspace(0.0, math.pi, 200)
ax.plot(0.5 + math.sqrt(5) / 2 * np.cos(th), math.sqrt(5) / 2 * np.sin(th), color=ORANGE, lw=2)
ax.plot([0], [1], "o", color=INK, ms=4)
ax.annotate("i", (0, 1), textcoords="offset points", xytext=(-9, 2), fontsize=9, color=INK)
ax.text(0.5, 0.45, "S = 5\non the axis", ha="center", fontsize=8.5, color=INK)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("S(M(z), M(γz)) for γ = (2 1; 1 1)", fontsize=9.5)
cb = fig.colorbar(cs, ax=ax, shrink=0.8, pad=0.02)
cb.ax.set_title("S", fontsize=8)
ax = axes[1]
ds = np.linspace(0, 2.2, 200)
ax.plot(ds, 2 * np.cosh(ds) - 2, color=BLUE, lw=2)
ax.plot(ds, ds ** 2, color=INK_2, lw=1, ls="--")
for dval, lab, off, ha in ((hyp(1j, RHO), "i and ρ: 0.309", (10, -10), "left"),
                           (math.acosh(1.5), "i and i + 1: 1", (10, -6), "left"),
                           (4 * math.log(PHI), "(2 1; 1 1) at i: 5", (-10, 2), "right")):
    ax.plot([dval], [2 * math.cosh(dval) - 2], "o", color=ORANGE, ms=6, mec="white", mew=1.2)
    ax.annotate(lab, (dval, 2 * math.cosh(dval) - 2), textcoords="offset points", xytext=off, ha=ha, fontsize=8,
                color=INK)
ax.text(2.02, 3.45, "δ²", fontsize=8.5, color=INK_2)
ax.grid(color=GRID, lw=0.6)
ax.set_xlabel("hyperbolic distance δ")
ax.set_ylabel("S = 2 cosh δ − 2")
ax.set_title("S is a function of distance", fontsize=9.5)
save(fig, "divergence-lattices")
print("done")
