"""Note 08: the speed of light, the light cone and spheres.

What it computes
  (1) c is a unit. 299 792 458 m/s is exact by the definition of the metre (SI, 1983); in other units the digits
      change (km/s, miles/s, feet per nanosecond), so the digits carry no arithmetic. With c = 1, space and time
      are measured in the same unit.
  (2) c turns lattices into frequencies. Light in a periodic box of side L has modes exp(i k.x),
      k = (2 pi/L) n with n in Z^3, two polarisations, frequency omega = c|k|. The number of modes below omega is
      2 #{n : |n|^2 <= N} ~ (8 pi/3) N^(3/2) = V omega^3 / (3 pi^2 c^3) (Planck's mode count); the multiplicity of
      the frequency (2 pi c/L) sqrt N is 2 r_3(N).
  (3) Spheres and class numbers (Gauss). For squarefree n > 3: r_3(n) = 12 h(-4n) if n = 1, 2 (mod 4),
      24 h(-n) if n = 3 (mod 8), and 0 if n = 7 (mod 8); with the class number formula,
      r_3(n) = (24 sqrt n / pi) L(1, chi_{-4n}) for n = 1, 2 (mod 4). Checked for all squarefree n <= 3000 and,
      for the L-value form, n <= 300.
  (4) Gauss's binary forms are integer relativity in 2 + 1 dimensions. The discriminant b^2 - 4ac is a Minkowski
      form of signature (2, 1) in (T, X, Y) = (a + c, b, a - c); SL(2, Z) acts on (a, b, c) by integral Lorentz
      transformations. Definite forms are timelike (massive particles; their roots tau are points of the upper
      half-plane, which is the space of velocities); degenerate forms are lightlike (the cusps); indefinite forms
      are spacelike (the closed geodesics). The Gaussian form x^2 + y^2 and the Eisenstein form x^2 + xy + y^2,
      normalised to the mass shell, move relative to each other at exactly c/2: their relative rapidity
      artanh(1/2) is the hyperbolic distance from i to rho. The class number h(D) counts integer particles of
      "mass" sqrt|D| up to integral Lorentz transformations, and equals (w sqrt|D| / 2 pi) L(1, chi_D).
  (5) Even spreading (Linnik, Duke). Integer points on the sphere x^2 + y^2 + z^2 = n, and integer particles on the
      mass shell b^2 - 4ac = D (Heegner points), spread evenly as n and |D| grow. On the sphere this is checked
      with Archimedes' hat-box law (heights uniform on [-1, 1]); on the modular surface, by the share of points
      above a given height.
  (6) The 3 + 1 light cone. Hermitian 2 x 2 matrices X have det X = t^2 - x^2 - y^2 - z^2 (c = 1); A in SL(2, C)
      acts by X -> A X A* preserving det (Lorentz), and on light rays (det X = 0) as a Moebius map of the
      celestial sphere. The Picard group SL(2, Z[i]) preserves the integer Hermitian matrices: an arithmetic
      Lorentz group built on the Gaussian integers.
  (7) Spheres: harmonics of degree l on the sphere have dimension 2l + 1, and sum_{l < n} (2l + 1) = n^2, the
      degeneracy of hydrogen's level n (without spin).
  (8) The fine-structure constant alpha = e^2/(4 pi eps0 hbar c), the dimensionless number in which c appears:
      CODATA 2018 gives 1/alpha = 137.035999084(21). Eddington's 137 and Wyler's 1969 formula
      (9/(8 pi^4)) (pi^5/(2^4 5!))^(1/4) miss by about 1.7 million and 4000 standard deviations.

What it shows
  The speed of light cannot be derived from number theory, because its value is a choice of units. What c does
  is set the light cone; with c = 1 the light cone is a quadratic form, and the arithmetic of quadratic forms
  (Gauss's classes, Dirichlet's L-values, the modular group) becomes the arithmetic of integer relativity.

What it does not claim
  That spacetime is a lattice, or that any physical constant follows from these identities.
"""
import cmath
import itertools
import math
from fractions import Fraction

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

from common import AQUA, BLUE, GRID, INK, INK_2, ORANGE, check, save

mp.mp.dps = 15
rng = np.random.default_rng(1983)

# ---------------------------------------------------------------------------------------------------
print("(1) c is a unit")
c = 299792458
check(Fraction(c, 1000) == Fraction(299792458, 1000), "c = 299 792 458 m/s exactly = 299 792.458 km/s")
mi = Fraction(1609344, 1000)
check(abs(float(c / mi) - 186282.397) < 1e-3, f"c = {float(c / mi):.3f} miles per second (1 mile = 1609.344 m)")
check(abs(float(Fraction(c) * Fraction(1, 10 ** 9) / Fraction(3048, 10000)) - 0.98357) < 1e-5,
      "c = 0.98357 feet per nanosecond: the digits depend only on the units")


# ---------------------------------------------------------------------------------------------------
def r3_table(N):
    r1 = np.zeros(N + 1, dtype=np.int64)
    for x in range(0, math.isqrt(N) + 1):
        r1[x * x] += 1 if x == 0 else 2
    r2 = np.convolve(r1, r1)[:N + 1]
    return np.convolve(r2, r1)[:N + 1]


print("(2) light in a periodic box")
NB = 40000
r3 = r3_table(NB)
modes = 2 * np.cumsum(r3)
for N in (100, 10000, 40000):
    ratio = modes[N] / (8 * math.pi / 3 * N ** 1.5)
    check(abs(ratio - 1) < 0.08 if N == 100 else abs(ratio - 1) < 0.01,
          f"modes with |n|^2 <= {N}: {modes[N]}, Planck/Weyl (8 pi/3) N^(3/2) = {8 * math.pi / 3 * N ** 1.5:.0f} (ratio {ratio:.4f})")


# ---------------------------------------------------------------------------------------------------
def legendre(D, p):
    if p == 2:
        return 0 if D % 2 == 0 else (1 if D % 8 in (1, 7) else -1)
    r = pow(D % p, (p - 1) // 2, p)
    return 0 if r == 0 else (1 if r == 1 else -1)


def kronecker(D, n):
    res, m, p = 1, n, 2
    while p * p <= m:
        while m % p == 0:
            res *= legendre(D, p)
            m //= p
        p += 1
    if m > 1:
        res *= legendre(D, m)
    return res


def reduced_forms(D):
    """primitive reduced positive definite forms (a, b, c) of discriminant D < 0."""
    out = []
    a = 1
    while 3 * a * a <= -D:
        for b in range(-a + 1, a + 1):
            if (b * b - D) % (4 * a) == 0:
                cc = (b * b - D) // (4 * a)
                if cc >= a and (b >= 0 or cc > a) and math.gcd(math.gcd(a, abs(b)), cc) == 1:
                    out.append((a, b, cc))
        a += 1
    return out


def squarefree(n):
    return all(n % (p * p) for p in range(2, math.isqrt(n) + 1))


print("(3) spheres and class numbers")
ok = True
for n in range(4, 3001):
    if not squarefree(n):
        continue
    if n % 4 in (1, 2):
        ok &= r3[n] == 12 * len(reduced_forms(-4 * n))
    elif n % 8 == 3:
        ok &= r3[n] == 24 * len(reduced_forms(-n))
    else:
        ok &= r3[n] == 0
check(ok, "Gauss: r_3(n) = 12 h(-4n), 24 h(-n) or 0 for every squarefree 3 < n <= 3000")


def L1(D):
    q = abs(D)
    return float(-sum(kronecker(D, a) * mp.digamma(mp.mpf(a) / q) for a in range(1, q + 1) if math.gcd(a, q) == 1) / q)


sample = [n for n in range(5, 301) if squarefree(n) and n % 4 in (1, 2)]
lvals = {n: 24 * math.sqrt(n) / math.pi * L1(-4 * n) for n in sample}
check(all(abs(lvals[n] - r3[n]) < 1e-6 for n in sample),
      f"r_3(n) = (24 sqrt n / pi) L(1, chi_-4n) for all {len(sample)} squarefree n = 1, 2 mod 4 below 300")

# ---------------------------------------------------------------------------------------------------
print("(4) binary forms as integer relativity in 2 + 1 dimensions")


def act(f, g):
    """(a, b, c) of f(alpha x + beta y, gamma x + delta y)."""
    a, b, cc = f
    (al, be), (ga, de) = g
    return (a * al * al + b * al * ga + cc * ga * ga,
            2 * a * al * be + b * (al * de + be * ga) + 2 * cc * ga * de,
            a * be * be + b * be * de + cc * de * de)


def disc(f):
    return f[1] * f[1] - 4 * f[0] * f[2]


S = ((0, -1), (1, 0))
T = ((1, 1), (0, 1))


def mul(g, h):
    return tuple(tuple(sum(g[i][k] * h[k][j] for k in range(2)) for j in range(2)) for i in range(2))


words = []
for _ in range(300):
    g = ((1, 0), (0, 1))
    for ch in rng.choice(["S", "T", "t"], size=rng.integers(1, 9)):
        g = mul(g, {"S": S, "T": T, "t": ((1, -1), (0, 1))}[ch])
    words.append(g)
forms = [tuple(int(v) for v in rng.integers(-9, 10, size=3)) for _ in range(300)]
check(all(disc(act(f, g)) == disc(f) for f, g in zip(forms, words)),
      "SL(2, Z) preserves the discriminant b^2 - 4ac: 300 random forms and matrices")
tc = lambda f: (f[0] + f[2], f[1], f[0] - f[2])
check(all(tc(f)[1] ** 2 + tc(f)[2] ** 2 - tc(f)[0] ** 2 == disc(f) for f in forms),
      "in (T, X, Y) = (a + c, b, a - c) the discriminant is X^2 + Y^2 - T^2: signature (2, 1)")


def root(f):
    a, b, cc = f
    return (-b + 1j * math.sqrt(-disc(f))) / (2 * a)


def mob(g, z):
    (al, be), (ga, de) = g
    return (al * z + be) / (ga * z + de)


def inv(g):
    (al, be), (ga, de) = g
    return ((de, -be), (-ga, al))


okr = True
for g in words[:100]:
    for f in ((1, 0, 1), (1, 1, 1), (2, 1, 3), (3, 2, 7)):
        okr &= abs(root(act(f, g)) - mob(inv(g), root(f))) < 1e-9
check(okr, "the root tau of a definite form moves by the Moebius action: tau(f o g) = g^-1 tau(f)")


def unit_velocity(f):
    T_, X_, Y_ = tc(f)
    return np.array([X_ / T_, Y_ / T_])


vG, vE = unit_velocity((1, 0, 1)), unit_velocity((1, 1, 1))
check(np.allclose(vG, 0) and np.allclose(vE, [0.5, 0]), "x^2 + y^2 is at rest; x^2 + xy + y^2 moves at velocity c/2")
pair = lambda f, g: f[1] * g[1] - 2 * f[0] * g[2] - 2 * g[0] * f[2]      # the bilinear form of b^2 - 4ac
rel = math.acosh(-pair((1, 0, 1), (1, 1, 1)) / math.sqrt(disc((1, 0, 1)) * disc((1, 1, 1))))   # relative rapidity
rho = cmath.exp(2j * math.pi / 3)
d_hyp = math.acosh(1 + abs(1j - rho) ** 2 / (2 * 1 * rho.imag))
check(abs(d_hyp - math.atanh(0.5)) < 1e-12 and abs(rel - math.atanh(0.5)) < 1e-12,
      f"hyperbolic distance from i to rho = artanh(1/2) = {math.atanh(0.5):.6f}: relative speed c/2")
check(disc((1, -1, -1)) == 5 and disc((1, 0, 0)) == 0,
      "x^2 - xy - y^2 (the golden geodesic) is spacelike, x^2 (a cusp) is lightlike")
fund = [D for D in range(-3, -400, -1) if (D % 4 == 1 and squarefree(-D)) or (D % 16 in (8, 12) and squarefree(-D // 4))]
okh = True
for D in fund:
    w = 6 if D == -3 else (4 if D == -4 else 2)
    okh &= abs(len(reduced_forms(D)) - w * math.sqrt(-D) / (2 * math.pi) * L1(D)) < 1e-6
check(okh, f"class number formula h(D) = (w sqrt|D| / 2 pi) L(1, chi_D) for all {len(fund)} fundamental D > -400")

# ---------------------------------------------------------------------------------------------------
print("(5) even spreading on spheres and on mass shells")
n_sph = 1000005                                        # squarefree, = 1 mod 4


def sphere_points(n):
    R = math.isqrt(n)
    out = []
    ys = np.arange(-R, R + 1)
    for x in range(-R, R + 1):
        z2 = n - x * x - ys * ys
        good = z2 >= 0
        zr = np.zeros_like(z2)
        zr[good] = np.round(np.sqrt(z2[good])).astype(np.int64)
        hit = good & (zr * zr == z2)
        for y, z in zip(ys[hit], zr[hit]):
            out.append((x, y, z))
            if z:
                out.append((x, y, -z))
    return np.array(out, dtype=float)


check(squarefree(n_sph) and n_sph % 4 == 1, f"n = {n_sph} is squarefree and 1 mod 4")
raw = sphere_points(n_sph)
check(len(raw) == 12 * len(reduced_forms(-4 * n_sph)), f"r_3({n_sph}) = {len(raw)} = 12 h(-4n) (Gauss)")
pts = raw / math.sqrt(n_sph)
zs = np.sort(pts[:, 2])
ks = float(np.max(np.abs(np.arange(1, len(zs) + 1) / len(zs) - (zs + 1) / 2)))
check(ks < 0.02, f"heights z/r of the {len(pts)} points are uniform on [-1, 1] (Archimedes' hat-box law): "
                 f"Kolmogorov-Smirnov distance {ks:.4f}")
heeg = []
for D in range(-20000, -20600, -1):
    if (D % 4 == 1 and squarefree(-D)) or (D % 16 in (8, 12) and squarefree(-D // 4)):
        heeg += [root(f) for f in reduced_forms(D)]
heeg = np.array(heeg)
for Y in (1.5, 3.0):
    frac = float(np.mean(heeg.imag > Y))
    check(abs(frac - 3 / (math.pi * Y)) < 0.03,
          f"{len(heeg)} Heegner points (|D| in [20000, 20600)): fraction above height {Y} is {frac:.3f}, "
          f"hyperbolic area fraction 3/(pi Y) = {3 / (math.pi * Y):.3f}")

# ---------------------------------------------------------------------------------------------------
print("(6) the 3 + 1 light cone and the celestial sphere")


def herm(t, x, y, z):
    return np.array([[t + z, x - 1j * y], [x + 1j * y, t - z]])


okl = True
for _ in range(200):
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    A = A / np.sqrt(np.linalg.det(A))
    Xm = herm(*rng.normal(size=4))
    okl &= abs(np.linalg.det(A @ Xm @ A.conj().T) - np.linalg.det(Xm)) < 1e-9
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    Av = A @ v
    zeta = v[0] / v[1]
    okl &= abs(Av[0] / Av[1] - (A[0, 0] * zeta + A[0, 1]) / (A[1, 0] * zeta + A[1, 1])) < 1e-9
    okl &= abs(np.linalg.det(np.outer(v, v.conj()))) < 1e-9
check(okl, "SL(2, C) preserves t^2 - x^2 - y^2 - z^2, and moves light rays v v* by Moebius maps of the sphere")
picard = [np.array([[1, 1], [0, 1]]), np.array([[1, 1j], [0, 1]]), np.array([[0, -1], [1, 0]]),
          np.array([[1j, 0], [0, -1j]])]
okp = True
for _ in range(200):
    d1, d2 = rng.integers(-9, 10, size=2)
    off = complex(*rng.integers(-9, 10, size=2))
    Xm = np.array([[d1, off], [off.conjugate(), d2]])
    for A in picard:
        Y_ = A @ Xm @ A.conj().T
        okp &= np.allclose(Y_, np.round(Y_.real) + 1j * np.round(Y_.imag)) and abs(Y_[0, 0].imag) < 1e-12 \
            and abs(np.linalg.det(Y_) - np.linalg.det(Xm)) < 1e-9
check(okp, "the Picard group SL(2, Z[i]) preserves the integer Hermitian matrices and their determinant")

# ---------------------------------------------------------------------------------------------------
print("(7) spheres and hydrogen")
check(all(math.comb(l + 2, 2) - math.comb(l, 2) == 2 * l + 1 for l in range(2, 50)),
      "harmonic polynomials of degree l in three variables: dimension 2l + 1")
check(all(sum(2 * l + 1 for l in range(n)) == n * n for n in range(1, 50)), "sum_{l < n} (2l + 1) = n^2")

# ---------------------------------------------------------------------------------------------------
print("(8) the fine-structure constant")
inv_alpha, sig = 137.035999084, 0.000000021
wyler = 1 / (9 / (8 * math.pi ** 4) * (math.pi ** 5 / (2 ** 4 * math.factorial(5))) ** 0.25)
print(f"     CODATA 2018 1/alpha = {inv_alpha} +- {sig}; Wyler {wyler:.7f}; Eddington 137")
check(abs(wyler - 137.0360824) < 1e-6 and abs(wyler - inv_alpha) / sig > 3000 and (inv_alpha - 137) / sig > 1e6,
      f"Wyler misses by {abs(wyler - inv_alpha) / sig:.0f} standard deviations, Eddington by {(inv_alpha - 137) / sig:.2e}")

# ---------------------------------------------------------------------------------------------------
print("figures")

# (a) binary forms in velocity space
fig, ax = plt.subplots(figsize=(5.6, 5.4))
th = np.linspace(0, 2 * np.pi, 400)
ax.fill(np.cos(th), np.sin(th), color="#f4f8fe", lw=0)
ax.plot(np.cos(th), np.sin(th), color=INK_2, lw=1.2)
for D, col, lab in ((-4, BLUE, "b² − 4ac = −4 (the Gaussian class)"), (-3, ORANGE, "b² − 4ac = −3 (the Eisenstein class)")):
    vs = []
    for a in range(1, 60):
        for b in range(-60, 61):
            if (b * b - D) % (4 * a) == 0:
                cc = (b * b - D) // (4 * a)
                if a + cc <= 70:
                    vs.append(unit_velocity((a, b, cc)))
    vs = np.array(vs)
    ax.plot(vs[:, 0], vs[:, 1], "o", ms=4, color=col, label=lab, mec="white", mew=0.4)
gold = np.array([0.0, -1.0, 2.0])                       # (T, X, Y) of x^2 - xy - y^2
s_ = np.linspace(-1, 1, 200)
cx, cy = s_ * 2 / math.sqrt(5), s_ * 1 / math.sqrt(5)   # timelike v with v.(X,Y) = T * 0, i.e. -vx + 2 vy = 0
ax.plot(cx, cy, color=AQUA, lw=2, label="x² − xy − y² (spacelike): a geodesic")
for f in ((1, 0, 0), (0, 0, 1), (1, 2, 1), (1, -2, 1), (4, 4, 1), (1, 4, 4)):
    T_, X_, Y_ = tc(f)
    ax.plot([X_ / T_], [Y_ / T_], "s", ms=6, color=INK, zorder=4)
ax.plot([], [], "s", ms=6, color=INK, label="squares such as x² (lightlike): cusps")
ax.annotate("x² + y² at rest", xy=(0, 0), xytext=(-0.75, 0.3), fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
ax.annotate("x² + xy + y² at c/2", xy=(0.5, 0), xytext=(0.22, -0.35), fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
ax.text(0.72, 0.86, "light\n|v| = c", fontsize=9, color=INK_2)
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.55, 1.15)
ax.set_aspect("equal")
ax.axis("off")
ax.legend(loc="lower center", fontsize=8, ncol=1)
ax.set_title("Gauss's binary forms as velocities in 2 + 1 relativity (c = 1)", fontsize=10)
save(fig, "light-forms")

# (b) spheres and mass shells, both in equal-area coordinates
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
ax = axes[0]
lon = np.arctan2(pts[:, 1], pts[:, 0])
ax.plot(np.degrees(lon), pts[:, 2], "o", ms=0.8, color=BLUE)
ax.set_xlim(-180, 180)
ax.set_ylim(-1, 1)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_xlabel("longitude (degrees)")
ax.set_ylabel("height z/r")
ax.set_title(f"sphere x² + y² + z² = {n_sph}:\n{len(pts)} integer points", fontsize=9)
ax = axes[1]
ax.plot(heeg.real, 1 / heeg.imag, "o", ms=0.8, color=ORANGE)
xb_ = np.linspace(-0.5, 0.5, 200)
ax.plot(xb_, 1 / np.sqrt(1 - xb_ ** 2), color=INK_2, lw=1)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(0, 1.18)
ax.set_xlabel("Re τ")
ax.set_ylabel("1 / Im τ")
ax.set_title(f"mass shells |D| ∈ [20000, 20600):\n{len(heeg)} Heegner points", fontsize=9)
fig.suptitle("Integer points on spheres and on mass shells spread evenly (Linnik; Duke 1988);\n"
             "both panels use coordinates in which area is uniform", fontsize=10, y=1.13)
save(fig, "light-spheres-shells")

# (c) light in a box
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
ax = axes[0]
Ns = np.arange(1, 2001)
ax.step(np.sqrt(Ns), modes[Ns], where="post", color=BLUE, lw=1.2, label="modes counted")
ax.plot(np.sqrt(Ns), 8 * math.pi / 3 * Ns ** 1.5, color=INK_2, lw=1, ls="--", label="V ω³ / (3π² c³)")
ax.set_xlabel("frequency, in units of 2πc/L")
ax.set_ylabel("number of modes")
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("Planck's mode count:\nc turns the lattice into frequencies", fontsize=9)
ax = axes[1]
xs_ = np.array(sample)
ax.plot(xs_, [r3[n] for n in sample], "o", ms=5, color=BLUE, label="r₃(n), counted")
ax.plot(xs_, [lvals[n] for n in sample], "x", ms=5, color=ORANGE, label="(24√n/π) L(1, χ₋₄ₙ)")
ax.set_xlabel("n (squarefree, ≡ 1, 2 mod 4)")
ax.set_ylabel("modes at frequency ∝ √n")
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("each degeneracy is an L-value\n(Gauss, Dirichlet)", fontsize=9)
save(fig, "light-modes")
print("done")
