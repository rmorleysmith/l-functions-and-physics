"""Note 02, part 1: quadratic characters count the modes of a periodic membrane and sum the energy of a crystal.

What it computes
  (1) For the nine imaginary quadratic fields of class number one, discriminant D in
      {-3, -4, -7, -8, -11, -19, -43, -67, -163}, the number r_Q(n) of points of the lattice of integers
      with norm Q = n equals w * sum_{d | n} chi_D(d) for every n <= 3000 (w = number of units). On the flat
      torus R^2 / Lambda these are the multiplicities of the eigenvalues of the Laplacian: the number of
      plane waves with the same frequency.
  (2) The same count for a square membrane with fixed edges (modes sin(m pi x) sin(n pi y), m, n >= 1).
  (3) Class number two: for D = -20 the single form x^2 + 5y^2 fails, and the two reduced forms together
      give 2 * sum chi_-20(d).
  (4) Epstein zeta functions: sum over the lattice of Q^(-2) equals w zeta(2) L(2, chi_D), checked by
      summing lattice points with a disc cutoff and a tail correction, for D = -3, -4, -7, -8, -11.
  (5) The Madelung constant of the two-dimensional square ionic crystal: the alternating lattice sum
      sum' (-1)^(x+y) (x^2 + y^2)^(-s) = -4 (1 - 2^(1-s)) zeta(s) beta(s), checked at s = 2 and at
      s = 1/2, where it gives the electrostatic energy 1.6155... (Evjen's method for the direct sum).
  (6) The three-dimensional rock-salt constant 1.7476 by the same direct method; there the lattice-point counts
      r_3(n) are class numbers (note 08), and no closed form in L-values is known.

What it shows
  The quadratic Dirichlet characters of the class-number-one fields are the mode-counting functions of the
  corresponding periodic lattices, and their L-functions are the spectral zeta functions of those lattices.
  Classical theorems (Jacobi, Dirichlet, Epstein; see Borwein, Glasser, McPhedran, Wan and Zucker, Lattice
  Sums Then and Now, Cambridge University Press, 2013), checked and drawn.
"""
import math

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

from common import BLUE, GRID, INK, INK_2, ORANGE, check, save

mp.mp.dps = 30


def kronecker(D, n):
    """The Kronecker symbol (D/n) for a discriminant D and n >= 1."""
    result = 1
    m = n
    p = 2
    while p * p <= m:
        while m % p == 0:
            result *= legendre(D, p)
            m //= p
        p += 1
    if m > 1:
        result *= legendre(D, m)
    return result


def legendre(D, p):
    if p == 2:
        if D % 2 == 0:
            return 0
        return 1 if D % 8 in (1, 7) else -1
    r = pow(D % p, (p - 1) // 2, p)
    return 0 if r == 0 else (1 if r == 1 else -1)


def principal_form(D):
    """(a, b, c) of the principal form of discriminant D < 0."""
    return (1, 0, -D // 4) if D % 4 == 0 else (1, 1, (1 - D) // 4)


def representation_counts(form, N):
    a, b, c = form
    r = np.zeros(N + 1, dtype=np.int64)
    disc = 4 * a * c - b * b
    ymax = int(math.isqrt(4 * a * N // disc)) + 2
    for y in range(-ymax, ymax + 1):
        xmax = int(math.isqrt(4 * c * N // disc)) + 2 + abs(b * y)
        x = np.arange(-xmax, xmax + 1)
        q = a * x * x + b * x * y + c * y * y
        q = q[(q >= 1) & (q <= N)]
        np.add.at(r, q, 1)
    return r


UNITS = {-3: 6, -4: 4}
HEEGNER = [-3, -4, -7, -8, -11, -19, -43, -67, -163]
N = 3000

# ---------------------------------------------------------------------------------------------------
print("(1) modes on the nine class-number-one tori")
for D in HEEGNER:
    w = UNITS.get(D, 2)
    r = representation_counts(principal_form(D), N)
    chi = [0] + [kronecker(D, n) for n in range(1, N + 1)]
    divsum = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        divsum[d::d] += chi[d]
    check((r[1:] == w * divsum[1:]).all(),
          f"D = {D:5d}: {principal_form(D)} represents n exactly {w} * sum_(d|n) chi_D(d) times, n <= {N}")

# ---------------------------------------------------------------------------------------------------
print("(2) a square membrane with fixed edges")
r2 = representation_counts((1, 0, 1), N)
fixed = np.zeros(N + 1, dtype=np.int64)
for m in range(1, int(math.isqrt(N)) + 1):
    for n in range(1, int(math.isqrt(N - m * m)) + 1):
        fixed[m * m + n * n] += 1
is_square = np.array([int(math.isqrt(k)) ** 2 == k for k in range(N + 1)])
check((fixed[1:] == (r2[1:] - 4 * is_square[1:]) // 4).all(),
      "fixed-edge square membrane: multiplicity of m^2 + n^2 = N is (r_2(N) - 4 [N square]) / 4")
check(r2[25] == 12 and fixed[25] == 2 and r2[65] == 16,
      "r_2(25) = 12 (5^2 + 0^2 and 3^2 + 4^2 with signs and order), r_2(65) = 16 (65 = 1^2 + 8^2 = 4^2 + 7^2)")

# ---------------------------------------------------------------------------------------------------
print("(3) class number two")
ra = representation_counts((1, 0, 5), N)
rb = representation_counts((2, 2, 3), N)
chi20 = [0] + [kronecker(-20, n) for n in range(1, N + 1)]
div20 = np.zeros(N + 1, dtype=np.int64)
for d in range(1, N + 1):
    div20[d::d] += chi20[d]
check(not (ra[1:] == 2 * div20[1:]).all() and ra[2] == 0 and div20[2] == 1,
      "x^2 + 5y^2 alone misses n = 2, which the character sum counts")
check((ra[1:] + rb[1:] == 2 * div20[1:]).all(),
      "x^2 + 5y^2 and 2x^2 + 2xy + 3y^2 together: 2 * sum chi_-20(d), n <= 3000")

# ---------------------------------------------------------------------------------------------------
print("(4) spectral zeta functions of the lattices")


def L(s, D):
    q = abs(D)
    return mp.dirichlet(s, [kronecker(D, a) if a else 0 for a in range(q)])


X = 400000
for D in (-3, -4, -7, -8, -11):
    w = UNITS.get(D, 2)
    r = representation_counts(principal_form(D), X).astype(float)
    n = np.arange(1, X + 1, dtype=float)
    direct = float(np.sum(r[1:] / n ** 2)) + 2 * math.pi / math.sqrt(-D) / X
    closed = w * mp.zeta(2) * L(2, D)
    check(abs(direct - float(closed)) < 1e-6 * float(closed),
          f"D = {D:3d}: sum' Q^-2 = {direct:.9f}, {w} zeta(2) L(2, chi_D) = {mp.nstr(closed, 12)}")
G = mp.catalan
check(abs(4 * mp.zeta(2) * L(2, -4) - 2 * mp.pi ** 2 / 3 * G) < 1e-25, "square lattice: (2 pi^2 / 3) * Catalan")

# ---------------------------------------------------------------------------------------------------
print("(5) the Madelung constant of the square ionic crystal")


def evjen_2d(M, s):
    x = np.arange(-M, M + 1)
    X2, Y2 = np.meshgrid(x, x)
    wgt = np.ones_like(X2, dtype=float)
    wgt[np.abs(X2) == M] *= 0.5
    wgt[np.abs(Y2) == M] *= 0.5
    r2_ = (X2 * X2 + Y2 * Y2).astype(float)
    r2_[M, M] = np.inf
    sign = np.where((X2 + Y2) % 2 == 0, 1.0, -1.0)
    return float(np.sum(wgt * sign * r2_ ** (-s)))


def naive_2d(M, s):
    x = np.arange(-M, M + 1)
    X2, Y2 = np.meshgrid(x, x)
    r2_ = (X2 * X2 + Y2 * Y2).astype(float)
    r2_[M, M] = np.inf
    sign = np.where((X2 + Y2) % 2 == 0, 1.0, -1.0)
    return float(np.sum(sign * r2_ ** (-s)))


def alt(s):
    return -4 * (1 - mp.power(2, 1 - s)) * mp.zeta(s) * L(s, -4)


check(abs(evjen_2d(400, 2) - float(alt(2))) < 1e-8, f"s = 2: alternating sum = {mp.nstr(alt(2), 12)}")
madelung = -alt(mp.mpf(1) / 2)
e = evjen_2d(200, 0.5)
check(abs(-e - float(madelung)) < 1e-5,
      f"s = 1/2: Evjen sum (M = 200) = {-e:.7f}; 4 (sqrt 2 - 1) |zeta(1/2)| beta(1/2) = {mp.nstr(madelung, 12)}")
print(f"     zeta(1/2) = {mp.nstr(mp.zeta(0.5), 12)}, beta(1/2) = L(1/2, chi_-4) = {mp.nstr(L(0.5, -4), 12)}")

# ---------------------------------------------------------------------------------------------------
print("(6) the three-dimensional rock-salt crystal, for contrast")


def evjen_3d(M):
    x = np.arange(-M, M + 1)
    X3, Y3, Z3 = np.meshgrid(x, x, x, indexing="ij")
    wgt = np.ones(X3.shape)
    for A in (X3, Y3, Z3):
        wgt[np.abs(A) == M] *= 0.5
    r = np.sqrt((X3 * X3 + Y3 * Y3 + Z3 * Z3).astype(float))
    r[M, M, M] = np.inf
    sign = np.where((X3 + Y3 + Z3) % 2 == 0, 1.0, -1.0)
    return -float(np.sum(wgt * sign / r))


m3 = evjen_3d(40)
check(abs(m3 - 1.747565) < 2e-4, f"rock salt (Evjen, M = 40): {m3:.6f} (the accepted value is 1.7475646)")

# ---------------------------------------------------------------------------------------------------
print("figures")

# (a) degenerate plane waves on the square and hexagonal tori
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
omega = complex(-0.5, math.sqrt(3) / 2)
for ax, (name, basis, norm, nval, formula) in zip(axes, (
        ("square torus ℝ²/ℤ[i]", (1, 1j), lambda a, b: a * a + b * b, 25,
         "$r(25) = 4\\,(\\chi_{-4}(1) + \\chi_{-4}(5) + \\chi_{-4}(25)) = 12$"),
        ("hexagonal torus ℝ²/ℤ[ω]", (1, omega), lambda a, b: a * a - a * b + b * b, 49,
         "$r(49) = 6\\,(\\chi_{-3}(1) + \\chi_{-3}(7) + \\chi_{-3}(49)) = 18$"))):
    R = math.sqrt(nval)
    t = np.linspace(0, 2 * math.pi, 400)
    for k in range(1, 81):
        if any(norm(a, b) == k for a in range(-10, 11) for b in range(-10, 11)):
            ax.plot(math.sqrt(k) * np.cos(t), math.sqrt(k) * np.sin(t), color=GRID, lw=0.6, zorder=0)
    pts = [(a * basis[0] + b * basis[1]) for a in range(-12, 13) for b in range(-12, 13)]
    pts = [p for p in pts if abs(p) <= 8.6]
    on = [p for p, (a, b) in zip(pts, [(0, 0)] * len(pts)) if abs(abs(p) ** 2 - nval) < 1e-9]
    ax.plot([p.real for p in pts], [p.imag for p in pts], "o", ms=2.5, color=INK_2, zorder=1)
    ax.plot(R * np.cos(t), R * np.sin(t), color="#86b6ef", lw=1.5, zorder=2)
    ax.plot([p.real for p in on], [p.imag for p in on], "o", ms=8, color=BLUE, mec="white", mew=1.5, zorder=3)
    check(len(on) == (12 if nval == 25 else 18), f"{name}: {len(on)} wave vectors with |k|^2 = {nval}")
    ax.set_title(f"{name}:\n{len(on)} plane waves of the same frequency", fontsize=10)
    ax.text(0, -9.9, formula, ha="center", fontsize=9, color=INK)
    ax.set_xlim(-9, 9)
    ax.set_ylim(-10.6, 9)
    ax.set_aspect("equal")
    ax.axis("off")
save(fig, "waves-tori")

# (b) the Madelung sum: naive squares oscillate, Evjen's weighting converges
Ms = np.arange(1, 41)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.plot(Ms, [-naive_2d(M, 0.5) for M in Ms], "o-", ms=4, lw=1.2, color=ORANGE, label="plain squares")
ax.plot(Ms, [-evjen_2d(M, 0.5) for M in Ms], "o-", ms=4, lw=1.2, color=BLUE, label="Evjen's weighting")
ax.axhline(float(madelung), color=INK_2, lw=1, ls="--")
ax.text(40.5, float(madelung), "$4(\\sqrt{2}-1)\\,|\\zeta(\\frac{1}{2})|\\,L(\\frac{1}{2},\\chi_{-4})$\n= 1.61554...",
        va="center", fontsize=9, color=INK)
ax.set_xlabel("half-width of the square of ions summed, M")
ax.set_ylabel("energy per ion (units $e^2/a$)")
ax.set_ylim(1.1, 1.75)
ax.set_xlim(0, 40)
ax.legend(loc="lower right")
ax.set_title("The square ionic crystal: its energy is a product of two L-values at s = ½")
save(fig, "waves-madelung")
print("done")
