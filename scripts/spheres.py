"""Note 10: spheres. Echoes, multipoles and rotations.

What it computes
  (1) Harmonic packets. The harmonic polynomials of degree l in d variables have dimension 2 (d = 2, l >= 1),
      2l + 1 (d = 3) and (l + 1)^2 (d = 4); the last is the degeneracy n^2 of the hydrogen level n = l + 1 (Fock).
  (2) Echoes of a cubic lattice. A flash at every point of Z^3, heard at a lattice point (wave equation, c = 1):
      Kirchhoff's formula gives echoes at the times sqrt N with strengths r_3(N)/(4 pi sqrt N). For squarefree N
      not 7 mod 8 this strength is L(1, chi_D)/zeta(2), D the discriminant of Q(sqrt(-N)); the echoes at
      N = 4^a (8b + 7) are silent. The same signal is t + sum_N r_3(N) sin(2 pi sqrt N t)/(2 pi sqrt N), a sum
      over the normal modes of the torus: Poisson summation for radial functions, checked with Gaussian smoothing.
  (3) Multipoles of a shell. With K_4(x) = sum x_i^4 - (3/(d + 2))|x|^4 (the cubic harmonic of degree 4),
      sum_{x in Z^d} K_4(x) q^(|x|^2) = (2d(d - 1)/(d + 2)) theta^d psi, psi = eta(z)^8 eta(4z)^8 / eta(2z)^8
      = sum (-1)^(n-1) sigma_3^odd(n) q^n. For d = 2, 3, 4 these are the eta products eta(z)^4 eta(2z)^2 eta(4z)^4,
      eta(z)^2 eta(2z)^7 eta(4z)^2 and eta(2z)^12, of weights 5, 11/2 and 6. The crystal field of a simple cubic
      lattice of point charges is the nearest-neighbour field times L(g, 9/2), g = eta(z)^2 eta(2z)^7 eta(4z)^2,
      evaluated by Ewald's method, which is the functional equation Lambda(11/2 - s) = Lambda(s).
  (4) Roundness. The normalised hexadecapole mu_d(N) = c_d(N)/(r_d(N) N^2) of the shell |x|^2 = N of Z^d:
      d = 2, mu(p) = cos(4 theta_p)/4 with Hecke's angles theta_p equidistributed; d = 3, |mu| about
      0.37/sqrt(r_3(N)); d = 4, Deligne's bound |a(p)| <= 2 p^(5/2) and the Sato-Tate law for eta(2z)^12.
      For d = 3 the square of the hexadecapole is a central L-value (Waldspurger): with F the newform of weight
      10 and level 2 and D the discriminant of Q(sqrt(-N)), a(N)^2 / N^(9/2) = L(5, F x chi_D)/L(5, F x chi_-4)
      for squarefree N not 7 mod 8; for N = 7 mod 8 the twisted L-function has root number -1.
  (5) Tetrahedral spin states. The 24 Hurwitz units are SL(2, F_3), with abelianisation Z/3, and form a root
      system of type D_4. Klein's invariant forms Phi (octahedron, degree 6), Psi (cube, degree 8), Phi^2 and X
      (degree 12) are spin states fixed by the units; their matrix coefficients, summed over the Hurwitz
      quaternions of norm n, are the coefficients of the newforms of level 2 and weights 8, 10, 14, 14. The
      number of invariant states of degree l (Molien) equals the number of newforms of weight l + 2 and level 2.
      The weight-10 form is the Shimura lift of g.
  (6) Rotations from quaternions. The six rotations of the sphere given by the quaternions 1 +- 2i, 1 +- 2j,
      1 +- 2k generate a free group; their sum, on harmonics of degree 1 <= l <= 40, has all eigenvalues within
      2 sqrt 5 (Lubotzky, Phillips and Sarnak, from Deligne's bound), distributed by the Kesten-McKay law of the
      6-regular tree.

What it shows
  A lattice of unit impulses, analysed by radius (shells) and by angle (spherical harmonics), is described packet
  by packet by modular forms: the size of a shell by Dirichlet L-values, its shape by cusp forms of weight
  l + d/2, and the evenness of its points by the Ramanujan bound.
"""
import itertools
import math

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import AQUA, BLUE, BLUE_RAMP, GRID, INK, INK_2, ORANGE, check, save

mp.mp.dps = 30
rng = np.random.default_rng(1935)


# ---------------------------------------------------------------------------------------------------
# exact power series with Python integers, truncated at q^M
def mul(a, b, m):
    out = [0] * (m + 1)
    for i, x in enumerate(a[:m + 1]):
        if x:
            for j, y in enumerate(b[:m + 1 - i]):
                if y:
                    out[i + j] += x * y
    return out


def power(a, e, m):
    out = [1] + [0] * m
    for _ in range(e):
        out = mul(out, a, m)
    return out


def inverse(a, m):
    out = [1] + [0] * m
    for n in range(1, m + 1):
        out[n] = -sum(a[k] * out[n - k] for k in range(1, n + 1))
    return out


def euler(d, m):
    """prod_n (1 - q^(dn))"""
    out = [1] + [0] * m
    for n in range(d, m + 1, d):
        for i in range(m, n - 1, -1):
            out[i] -= out[i - n]
    return out


def eta_quotient(exps, m):
    """prod_d eta(dz)^(r_d) as a list of coefficients of q^0, ..., q^m (the total q-power must be an integer)."""
    out = [1] + [0] * m
    for d, r in exps.items():
        e = euler(d, m)
        if r < 0:
            e, r = inverse(e, m), -r
        out = mul(out, power(e, r, m), m)
    shift, rem = divmod(sum(d * r for d, r in exps.items()), 24)
    assert rem == 0
    return ([0] * shift + out)[:m + 1]


def theta_series(m):
    out = [0] * (m + 1)
    for n in range(-math.isqrt(m), math.isqrt(m) + 1):
        out[n * n] += 1
    return out


def r_table(d, m):
    one = np.array(theta_series(m), dtype=np.int64)
    out = one.copy()
    for _ in range(d - 1):
        out = np.convolve(out, one)[:m + 1]
    return out


def squarefree(n):
    return all(n % (p * p) for p in range(2, math.isqrt(n) + 1))


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


def sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, math.isqrt(n) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


# ---------------------------------------------------------------------------------------------------
print("(1) harmonic packets")


def harmonic_dimension(d, l):
    """nullity of the Laplacian from polynomials of degree l to degree l - 2 in d variables."""
    mons = [a for a in itertools.product(range(l + 1), repeat=d) if sum(a) == l]
    if l < 2:
        return len(mons)
    lower = {a: i for i, a in enumerate(a for a in itertools.product(range(l - 1), repeat=d) if sum(a) == l - 2)}
    A = np.zeros((len(lower), len(mons)))
    for j, a in enumerate(mons):
        for i in range(d):
            if a[i] >= 2:
                b = list(a)
                b[i] -= 2
                A[lower[tuple(b)], j] += a[i] * (a[i] - 1)
    return len(mons) - np.linalg.matrix_rank(A)


formula = {2: lambda l: 1 if l == 0 else 2, 3: lambda l: 2 * l + 1, 4: lambda l: (l + 1) ** 2}
check(all(harmonic_dimension(d, l) == formula[d](l) for d in (2, 3, 4) for l in range(9)),
      "harmonic polynomials of degree l <= 8: dimension 2 (d = 2, l >= 1), 2l + 1 (d = 3), (l + 1)^2 (d = 4)")
check(all(sum(2 * l + 1 for l in range(n)) == n * n == formula[4](n - 1) for n in range(1, 30)),
      "hydrogen level n: n^2 = sum_{l < n} (2l + 1) = dimension of the harmonics of degree n - 1 on the 3-sphere")

# ---------------------------------------------------------------------------------------------------
print("(2) echoes of a cubic lattice")
NE = 3000
r3 = r_table(3, NE)
check(list(r3[1:10]) == [6, 12, 8, 6, 24, 24, 0, 12, 30],
      "shells of Z^3 (numbers of N-th neighbours in a simple cubic crystal): 6, 12, 8, 6, 24, 24, 0, 12, 30")


def legendre_silent(n):
    while n % 4 == 0:
        n //= 4
    return n % 8 == 7


check(all((r3[n] == 0) == legendre_silent(n) for n in range(1, NE + 1)),
      f"silent echoes: r_3(N) = 0 exactly for N = 4^a (8b + 7), N <= {NE} (Legendre); first 7, 15, 23, 28, 31")


def L1(D):
    """L(1, chi_D) for a negative discriminant, by the finite formula -(pi/|D|^(3/2)) sum k chi_D(k)."""
    q = -D
    return -math.pi / q ** 1.5 * sum(k * kronecker(D, k) for k in range(1, q))


zeta2 = math.pi ** 2 / 6
sample = [n for n in range(1, 301) if squarefree(n) and n % 8 != 7]
worst = max(abs(r3[n] / (4 * math.pi * math.sqrt(n)) - L1(-4 * n if n % 4 in (1, 2) else -n) / zeta2) for n in sample)
check(worst < 1e-9, f"echo strength r_3(N)/(4 pi sqrt N) = L(1, chi_D)/zeta(2) for the {len(sample)} squarefree "
      f"N <= 300 not 7 mod 8 (error {worst:.0e})")
worst = max(abs(r3[n] - 24 / math.pi * math.sqrt(n) * L1(-4 * n if n % 4 in (1, 2) else -n)) for n in sample)
check(worst < 1e-8 and int(r3[1]) == 6 and [n for n in range(1, 41) if r3[n] == 0] == [7, 15, 23, 28, 31, 39],
      "powder rings of a primitive cubic lattice: multiplicity m_N = r_3(N) = (24/pi) sqrt(N) L(1, chi_D); "
      "rings 7, 15, 23, 28, 31, 39 absent")

SIG = 0.05
Ns = np.arange(1, NE + 1)
sq = np.sqrt(Ns)
w3 = r3[1:].astype(float)


def gauss(x):
    return np.exp(-x ** 2 / (2 * SIG ** 2)) / (SIG * math.sqrt(2 * math.pi))


def echoes_smoothed(t):
    """the echo train, smoothed by a Gaussian of width SIG (with the image term and the source's own term)."""
    return t * gauss(t) / (2 * math.pi * SIG ** 2) + (w3 * (gauss(sq - t) - gauss(sq + t)) / (4 * math.pi * sq)).sum()


def modes_smoothed(t, nmax=NE):
    damp = np.exp(-2 * math.pi ** 2 * SIG ** 2 * Ns[:nmax])
    return t + (w3[:nmax] * damp * np.sin(2 * math.pi * sq[:nmax] * t) / (2 * math.pi * sq[:nmax])).sum()


ts = np.linspace(0.3, 6.0, 115)
worst = max(abs(echoes_smoothed(t) - modes_smoothed(t)) for t in ts)
check(worst < 1e-11, f"echoes = mean growth + normal modes, with r_3(N) on both sides (Poisson summation in three "
      f"dimensions): agreement {worst:.0e} for 0.3 <= t <= 6")

# ---------------------------------------------------------------------------------------------------
print("(3) multipoles of a shell")
M = 400
theta = theta_series(M)
psi = eta_quotient({1: 8, 2: -8, 4: 8}, M)


def sigma3_odd(n):
    return sum(d ** 3 for d in range(1, n + 1) if n % d == 0 and (n // d) % 2 == 1)


check(all(psi[n] == (-1) ** (n - 1) * sigma3_odd(n) for n in range(1, M + 1)),
      "psi = eta(z)^8 eta(4z)^8 / eta(2z)^8 = sum (-1)^(n-1) sigma_3^odd(n) q^n = q - 8q^2 + 28q^3 - 64q^4 + ...")


def hexadecapole(d, m):
    """(d + 2) sum_{|x|^2 = N} K_4(x) for N <= m, as exact integers, and the shell sizes r_d(N)."""
    R = math.isqrt(m)
    grids = np.meshgrid(*([np.arange(-R, R + 1)] * d), indexing="ij")
    n = sum(g * g for g in grids)
    mask = n <= m
    quartic = sum(g ** 4 for g in grids)
    w = ((d + 2) * quartic - 3 * n * n)[mask]
    c = np.bincount(n[mask], weights=w.astype(float), minlength=m + 1)
    return np.rint(c).astype(np.int64), np.bincount(n[mask], minlength=m + 1)


ETA = {2: {1: 4, 2: 2, 4: 4}, 3: {1: 2, 2: 7, 4: 2}, 4: {2: 12}}
for d, m in ((2, 400), (3, 400), (4, 150)):
    c, _ = hexadecapole(d, m)
    rhs = mul(power(theta, d, m), psi, m)
    check(all(int(c[n]) == 2 * d * (d - 1) * rhs[n] for n in range(m + 1)) and rhs == eta_quotient(ETA[d], m),
          f"d = {d}: sum K_4(x) q^(|x|^2) = ({2 * d * (d - 1)}/{d + 2}) theta^{d} psi = ({2 * d * (d - 1)}/{d + 2}) "
          + " ".join(f"eta({k}z)^{r}" for k, r in ETA[d].items()) + f", weight {4 + d / 2:g} (N <= {m})")

Dq = lambda f: [n * cf for n, cf in enumerate(f)]          # q d/dq
bracket = [x - 3 * y for x, y in zip(mul(theta, Dq(Dq(theta)), M), mul(Dq(theta), Dq(theta), M))]
check(bracket == [2 * v for v in mul(power(theta, 2, M), psi, M)],
      "theta D^2 theta - 3 (D theta)^2 = 2 theta^2 psi, D = q d/dq (a Rankin-Cohen bracket of weight 5)")
g = mul(power(theta, 3, M), psi, M)
check(g[1:10] == [1, -2, -8, 16, 20, -36, 0, -32, -75],
      "g = eta(z)^2 eta(2z)^7 eta(4z)^2 = q - 2q^2 - 8q^3 + 16q^4 + 20q^5 - 36q^6 + 0q^7 - 32q^8 - 75q^9 + ...")

# the crystal field: a(N) for N <= 3000, the Dirichlet series at s = 9/2
MA = 3000
psi_long = [0] + [(-1) ** (n - 1) * sigma3_odd(n) for n in range(1, MA + 1)]
r3_list = [int(v) for v in r3]
a = [0] * (MA + 1)
for n in range(1, MA + 1):
    a[n] = sum(psi_long[k] * r3_list[n - k] for k in range(1, n + 1))
check(a[:M + 1] == g, "a(N) = sum_m psi(m) r_3(N - m): the hexadecapole of a shell from the sizes of the smaller shells")
check(max(abs(a[n]) / n ** 2.5 for n in range(1, MA + 1)) <= 1.0, "|a(N)| <= N^(5/2) for N <= 3000")


def G(t, terms=120):
    return sum(a[n] * mp.e ** (-mp.pi * n * t) for n in range(1, terms))


t0 = mp.mpf("1.3")
check(abs(G(1 / t0) / (t0 ** mp.mpf(5.5) * G(t0)) - 1) < mp.mpf(10) ** -20,
      "G(t) = sum a(N) exp(-pi N t) satisfies G(1/t) = t^(11/2) G(t): g has weight 11/2")


def completed(s, terms=60):
    """Lambda(s) = pi^(-s) Gamma(s) sum a(N) N^(-s), from G(1/t) = t^(11/2) G(t) (Riemann's and Ewald's method)."""
    s = mp.mpf(s)
    s2 = mp.mpf(11) / 2 - s
    return sum(a[n] * (mp.gammainc(s, mp.pi * n) / (mp.pi * n) ** s + mp.gammainc(s2, mp.pi * n) / (mp.pi * n) ** s2)
               for n in range(1, terms))


L92 = completed(4.5) * mp.pi ** mp.mpf(4.5) / mp.gamma(4.5)
direct = sum(a[n] * n ** -4.5 for n in range(1, MA + 1))
check(abs(L92 - mp.mpf("0.888064766485554")) < mp.mpf(10) ** -14 and abs(direct - float(L92)) < 1e-7,
      f"simple cubic lattice of like charges: L(g, 9/2) = sum a(N) N^(-9/2) = {mp.nstr(L92, 15)} "
      f"(Ewald; direct sum to N = 3000: {direct:.8f})")
alt = [sum((-1) ** (n - 1) * a[n] * n ** -4.5 for n in range(1, top + 1)) for top in (1500, 3000)]
check(abs(alt[1] - 1.0224521) < 2e-7 and abs(alt[1] - alt[0]) < 1e-6,
      f"rock salt (alternating charges): sum (-1)^(N-1) a(N) N^(-9/2) = {alt[1]:.7f}")


def octahedral_residual(scale):
    """potential of six unit charges at distance 1 on the axes, minus 6 + (35/4)(x^4 + y^4 + z^4 - 3r^4/5)."""
    pt = [mp.mpf(v) * scale for v in ("0.31", "0.17", "0.23")]
    sites = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    V = sum(1 / mp.sqrt(sum((pc - sc) ** 2 for pc, sc in zip(pt, site))) for site in sites)
    r2 = sum(v * v for v in pt)
    return V - 6 - mp.mpf(35) / 4 * (sum(v ** 4 for v in pt) - mp.mpf(3) / 5 * r2 ** 2)


ratio = octahedral_residual(mp.mpf(1) / 10) / octahedral_residual(mp.mpf(1) / 20)
check(abs(ratio - 64) < 1, "six charges at distance a on the axes: V = 6q/a + (35q/4a^5)(x^4 + y^4 + z^4 - 3r^4/5) + O(r^6), "
      f"the octahedral crystal field (the remainder scales as r^6: ratio {mp.nstr(ratio, 5)} on halving r)")
check(abs(175 / 48 * 12 / 5 - 35 / 4) < 1e-12,
      "a shell of charges at radius a sqrt N adds the same term times a(N) N^(-9/2): (175/48)(12/5) = 35/4")

# ---------------------------------------------------------------------------------------------------
print("(4) how round is a shell")
# d = 2: Hecke's angles
PM = 10 ** 6
is_prime = sieve(PM)
A, B = np.meshgrid(np.arange(1, 1001), np.arange(1, 1001), indexing="ij")
keep = (A > B) & (A * A + B * B <= PM)
A, B = A[keep], B[keep]
pr = is_prime[A * A + B * B]
p2, ang = (A * A + B * B)[pr], np.arctan2(B[pr], A[pr])
check(len(p2) == np.count_nonzero(is_prime[np.arange(PM + 1) % 4 == 1]),
      f"every prime p = 1 mod 4 below 10^6 is a^2 + b^2 with a > b > 0 once: {len(p2)} primes, angles theta_p")
h5 = eta_quotient(ETA[2], 400)
small = p2 < 400
check(all(h5[int(p)] == round(2 * p * p * math.cos(4 * th)) for p, th in zip(p2[small], ang[small])),
      "d = 2: the coefficient of eta(z)^4 eta(2z)^2 eta(4z)^4 at p = 1 mod 4 is 2 p^2 cos(4 theta_p)")
u = np.sort(ang / (math.pi / 4))
ks2 = np.max(np.abs(u - (np.arange(1, len(u) + 1) - 0.5) / len(u)))
means = [abs(np.cos(4 * k * ang).mean()) for k in range(1, 6)]
check(ks2 < 0.005 and max(means) < 1 / math.sqrt(len(u)),
      f"Hecke: the angles are equidistributed (KS distance {ks2:.4f}; |mean cos(4k theta)| <= {max(means):.4f}, k <= 5)")

# d = 3: hexadecapoles for N <= 20000
N3 = 20000
R = math.isqrt(N3)
X, Y = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1), indexing="ij")
r3_big = np.zeros(N3 + 1)
c3_big = np.zeros(N3 + 1)
for z in range(-R, R + 1):
    n = X * X + Y * Y + z * z
    mask = n <= N3
    nn = n[mask]
    r3_big += np.bincount(nn, minlength=N3 + 1)
    wgt = 5 * (X[mask] ** 4 + Y[mask] ** 4 + z ** 4) - 3 * nn.astype(np.int64) ** 2
    c3_big += np.bincount(nn, weights=wgt.astype(float), minlength=N3 + 1)
check(all(c3_big[n] == 12 * a[n] for n in range(1, MA + 1)), "d = 3: enumeration agrees with the series for N <= 3000")
idx3 = np.nonzero(r3_big[1:] > 0)[0] + 1
mu3 = np.abs(c3_big[idx3]) / 5 / (r3_big[idx3] * idx3.astype(float) ** 2)
scaled3 = mu3 * np.sqrt(r3_big[idx3])
bands = [float(np.median(scaled3[(idx3 > lo) & (idx3 <= hi)])) for lo, hi in ((0, 2000), (2000, 8000), (8000, 20000))]
check(all(0.33 < b < 0.42 for b in bands),
      f"d = 3: |mu| sqrt(r_3) has median {bands[0]:.3f}, {bands[1]:.3f}, {bands[2]:.3f} for N <= 2000, 8000, 20000")

# d = 4: eta(2z)^12 and Deligne's bound
XM = 10000
jac = np.zeros(XM + 1, dtype=np.int64)
k = 0
while k * (k + 1) // 2 <= XM:
    jac[k * (k + 1) // 2] = (-1) ** k * (2 * k + 1)        # Jacobi: prod (1 - x^n)^3
    k += 1
e6 = np.convolve(jac, jac)[:XM + 1]
e12 = np.convolve(e6, e6)[:XM + 1]                          # eta(2z)^12 = q e12(q^2)
check([int(e12[m]) for m in range(7)] == eta_quotient({2: 12}, 13)[1::2],
      "eta(2z)^12 = q - 12q^3 + 54q^5 - 88q^7 - 99q^9 + 540q^11 - 418q^13 + ...")
primes4 = np.nonzero(sieve(2 * XM))[0]
primes4 = primes4[primes4 > 2]
a4 = e12[(primes4 - 1) // 2].astype(float)
st = a4 / (2 * primes4 ** 2.5)
check(np.abs(st).max() < 1, f"Deligne: |a(p)| <= 2 p^(5/2) for the {len(primes4)} odd primes below 20000 "
      f"(largest ratio {np.abs(st).max():.4f})")
xs = np.sort(st)
semi = 0.5 + (xs * np.sqrt(1 - xs ** 2) + np.arcsin(xs)) / math.pi
ks4 = np.max(np.abs(semi - (np.arange(1, len(xs) + 1) - 0.5) / len(xs)))
check(ks4 < 0.03, f"Sato-Tate: a(p)/(2 p^(5/2)) follows the semicircle law (KS distance {ks4:.4f})")
_, r4 = hexadecapole(4, 150)
check(all(r4[n] == 8 * sum(d for d in range(1, n + 1) if n % d == 0 and d % 4) for n in range(1, 151)),
      "Jacobi: r_4(N) = 8 sum of the divisors of N not divisible by 4")
mu4 = 4 * np.abs(a4) / (8 * (primes4 + 1) * primes4.astype(float) ** 2)
check(np.all(mu4 <= np.sqrt(primes4) / (primes4 + 1)), "d = 4: |mu_4(p)| <= sqrt p/(p + 1), about 1/sqrt(8 r_4(p))")

# d = 3: the square of the hexadecapole is a central L-value
MF = 900
p1 = np.array(euler(1, MF), dtype=np.int64)
p2 = np.array(euler(2, MF), dtype=np.int64)


def npower(arr, e):
    out = np.zeros(MF + 1, dtype=np.int64)
    out[0] = 1
    for _ in range(e):
        out = np.convolve(out, arr)[:MF + 1]
    return out


f8_long = [0] + [int(v) for v in np.convolve(npower(p1, 8), npower(p2, 8))[:MF]]            # eta(z)^8 eta(2z)^8
E2_long = [1] + [-24 * sum(d for d in range(1, n + 1) if n % d == 0) for n in range(1, MF + 1)]
F10 = mul(f8_long, [2 * (E2_long[n // 2] if n % 2 == 0 else 0) - E2_long[n] for n in range(MF + 1)], MF)
check(F10[1:8] == [1, 16, -156, 256, 870, -2496, -952],
      "F = eta(z)^8 eta(2z)^8 (2E_2(2z) - E_2(z)) = q + 16q^2 - 156q^3 + 256q^4 + 870q^5 - ..., weight 10, level 2")
F10f = np.array(F10[1:], dtype=float)
nF = np.arange(1, MF + 1, dtype=float)


def twisted_central_value(D):
    """L(5, F x chi_D) for a fundamental discriminant D < 0 with root number +1: conductor 2 D^2 (D odd) or D^2."""
    cond = 2 * D * D if D % 2 else D * D
    y = 2 * math.pi * nF / math.sqrt(cond)
    chi = np.array([kronecker(D, n) for n in range(1, MF + 1)], dtype=float)
    tail = np.exp(-y) * (1 + y + y ** 2 / 2 + y ** 3 / 6 + y ** 4 / 24)       # Gamma(5, y)/Gamma(5)
    return 2 * float((F10f * chi * tail / nF ** 5).sum())


L_first = twisted_central_value(-4)
shells = [n for n in range(1, 31) if squarefree(n) and n % 8 != 7]
worst = max(abs(a[n] ** 2 / n ** 4.5 * L_first / twisted_central_value(-4 * n if n % 4 in (1, 2) else -n) - 1)
            for n in shells)
check(worst < 1e-9, f"Waldspurger: a(N)^2 / N^(9/2) = L(5, F x chi_D)/L(5, F x chi_-4) for the {len(shells)} squarefree "
      f"N <= 30 not 7 mod 8 (relative error {worst:.0e}); 1/L(5, F x chi_-4) = {1 / L_first:.9f}")
worst = max(abs((12 / 5 * a[n] / (int(r3[n]) * n ** 2)) ** 2
                - math.pi ** 2 / 100 * twisted_central_value(-4 * n if n % 4 in (1, 2) else -n)
                / (L_first * L1(-4 * n if n % 4 in (1, 2) else -n) ** 2 * math.sqrt(n))) for n in shells)
check(worst < 1e-10, "roundness: mu_3(N)^2 = (pi^2/100) L(5, F x chi_D) / (L(5, F x chi_-4) L(1, chi_D)^2 sqrt N)")


def twist_root_number(D):
    """the root number of F x chi_D for odd D, from f(-1/(Mz)) = w M^5 z^10 f(z), M = 2 D^2, at z = i t/sqrt M."""
    cond = 2 * D * D
    chi = [kronecker(D, n) for n in range(1, MF + 1)]
    f = lambda t: sum(F10[n] * chi[n - 1] * mp.e ** (-2 * mp.pi * n * t / mp.sqrt(cond)) for n in range(1, 400))
    t = mp.mpf("1.1")
    w = f(1 / t) / ((1j * t) ** 10 * f(t))       # (sqrt M z)^10 with z = i t/sqrt M
    return -w.real                                # root number = (-1)^(k/2) w, k = 10


eps3, eps7 = twist_root_number(-3), twist_root_number(-7)
check(abs(eps3 - 1) < 1e-12 and abs(eps7 + 1) < 1e-12 and all(a[n] == 0 for n in range(7, MA + 1, 8)),
      "the root number of F x chi_D is +1 for D = -3 and -1 for D = -7: on the empty shells N = 7 mod 8 the central "
      "value vanishes by symmetry")

# ---------------------------------------------------------------------------------------------------
print("(5) tetrahedral spin states and newforms of level 2")
MH = 48
RH = 2 * math.isqrt(MH) + 1
grid = np.arange(-RH, RH + 1)
G4 = np.stack(np.meshgrid(grid, grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 4)
par = G4 % 2
G4 = G4[(par == par[:, :1]).all(axis=1)]
n4 = (G4 * G4).sum(axis=1)
G4 = G4[(n4 % 4 == 0) & (n4 > 0) & (n4 <= 4 * MH)]
hurwitz = G4 / 2.0
hnorm = (G4 * G4).sum(axis=1) // 4
counts = np.bincount(hnorm, minlength=MH + 1)
check(all(counts[n] == 24 * sum(d for d in range(1, n + 1, 2) if n % d == 0) for n in range(1, MH + 1)),
      "Hurwitz quaternions of norm n: 24 times the sum of the odd divisors of n")
units = hurwitz[hnorm == 1]


def qmul(p, q):
    a1, b1, c1, d1 = p
    a2, b2, c2, d2 = q
    return (a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2)


def mmul3(p, q):
    return tuple((p[i] * q[j] + p[i + 1] * q[j + 2]) % 3 for i in (0, 2) for j in (0, 1))


# SL(2, F_3) -> Hurwitz units, S -> i, T -> (-1 - i + j + k)/2
gens = [((0, 2, 1, 0), (0.0, 1.0, 0.0, 0.0)), ((1, 1, 0, 1), (-0.5, -0.5, 0.5, 0.5))]
graph = {((1, 0, 0, 1), (1.0, 0.0, 0.0, 0.0))}
frontier = list(graph)
while frontier:
    new = []
    for m, q in frontier:
        for gm, gq in gens:
            e = (mmul3(m, gm), qmul(q, gq))
            if e not in graph:
                graph.add(e)
                new.append(e)
    frontier = new
check(len(graph) == 24 and len({m for m, _ in graph}) == 24 and len({q for _, q in graph}) == 24
      and {q for _, q in graph} == {tuple(u) for u in units},
      "the Hurwitz units are SL(2, F_3): S -> i, T -> (-1 - i + j + k)/2 extends to an isomorphism of groups of order 24")
sl2f3 = [m for m, _ in graph]
inv3 = {m: next(x for x in sl2f3 if mmul3(m, x) == (1, 0, 0, 1)) for m in sl2f3}
comm = {mmul3(mmul3(x, y), mmul3(inv3[x], inv3[y])) for x in sl2f3 for y in sl2f3}
while True:
    bigger = comm | {mmul3(x, y) for x in comm for y in comm}
    if bigger == comm:
        break
    comm = bigger
check(len(comm) == 8 and (1, 1, 0, 1) not in comm,
      "its commutator subgroup has order 8 (the quaternion group) and T generates the quotient Z/3: "
      "the characters of order 3 of SL(2, Z) factor through the Hurwitz units")
gram = units @ units.T
reflections_close = all(any(np.allclose(b - 2 * (alpha @ b) * alpha, u) for u in units) for alpha in units for b in units)
check(reflections_close and set(np.rint(2 * gram).astype(int).ravel()) == {-2, -1, 0, 1, 2}
      and np.linalg.matrix_rank(units) == 4,
      "the 24 units form a root system of rank 4 with one root length: type D_4 (the vertices of the 24-cell)")


def su2(q):
    """the matrices [[a + bi, c + di], [-c + di, a - bi]] of an array of quaternions a + bi + cj + dk."""
    q = np.atleast_2d(q)
    a_, b_, c_, d_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([np.stack([a_ + 1j * b_, c_ + 1j * d_], axis=-1), np.stack([-c_ + 1j * d_, a_ - 1j * b_], axis=-1)], axis=-2)


def pmul(Pa, Pb):
    """product of polynomials in u, coefficients along the last axis, vectorised over the first."""
    out = np.zeros((Pa.shape[0], Pa.shape[1] + Pb.shape[1] - 1), dtype=complex)
    for j in range(Pb.shape[1]):
        out[:, j:j + Pa.shape[1]] += Pa * Pb[:, j:j + 1]
    return out


def act(F, mats):
    """(rho(m) F)(u, v) = F((u, v) m) for a binary form F = sum F[k] u^(l-k) v^k and an array of 2 x 2 matrices."""
    l = len(F) - 1
    U = np.stack([mats[:, 1, 0], mats[:, 0, 0]], axis=-1)        # m00 u + m10 v, ascending powers of u (v = 1)
    V = np.stack([mats[:, 1, 1], mats[:, 0, 1]], axis=-1)        # m01 u + m11 v
    one = np.ones((mats.shape[0], 1), dtype=complex)
    pu, pv = [one], [one]
    for _ in range(l):
        pu.append(pmul(pu[-1], U))
        pv.append(pmul(pv[-1], V))
    out = np.zeros((mats.shape[0], l + 1), dtype=complex)
    for k in range(l + 1):
        if F[k] != 0:
            out += F[k] * pmul(pu[l - k], pv[k])[:, ::-1]
    return out


def inner(F, Gs):
    """the SU(2)-invariant inner product on binary forms of degree l: monomials orthogonal, |u^(l-k) v^k|^2 = 1/C(l, k)."""
    l = len(F) - 1
    wts = np.array([1 / math.comb(l, k) for k in range(l + 1)])
    return (np.conj(F) * wts * Gs).sum(axis=-1)


def form(l, terms):
    F = np.zeros(l + 1, dtype=complex)
    for k, cf in terms.items():
        F[k] = cf
    return F


PHI = form(6, {1: 1, 5: -1})                    # u^5 v - u v^5: the vertices of the octahedron
PSI = form(8, {0: 1, 4: 14, 8: 1})              # u^8 + 14 u^4 v^4 + v^8: the vertices of the cube
XI = form(12, {0: 1, 4: -33, 8: -33, 12: 1})    # u^12 - 33 u^8 v^4 - 33 u^4 v^8 + v^12: the edge midpoints
PHI2 = np.convolve(PHI, PHI)
KLEIN = {"Phi": PHI, "Psi": PSI, "X": XI, "Phi^2": PHI2}
klein_relation = np.convolve(XI, XI) - np.convolve(np.convolve(PSI, PSI), PSI) + 108 * np.convolve(PHI2, PHI2)
check(np.abs(klein_relation).max() == 0, "Klein: X^2 = Psi^3 - 108 Phi^4")
edge_midpoints = {tuple(np.round(np.array(v) / math.sqrt(2), 9)) for v in itertools.product((1, -1, 0), repeat=3)
                  if sum(abs(t) for t in v) == 2}
stars_X = set()
for zr in np.roots(XI):
    den = 1 + abs(zr) ** 2
    stars_X.add(tuple(np.round([2 * zr.real / den, 2 * zr.imag / den, (abs(zr) ** 2 - 1) / den], 9) + 0.0))
check(stars_X == edge_midpoints, "the 12 stars of X are the midpoints of the edges of the octahedron")
twist = su2(np.array([1.0, 1.0, 0.0, 0.0]) / math.sqrt(2))
unit_mats = su2(units)
signs = {}
for name, F in KLEIN.items():
    fixed = np.abs(act(F, unit_mats) - F).max() < 1e-12
    s = (inner(F, act(F, twist)) / inner(F, F[None, :]))[0]
    signs[name] = round(s.real)
    check(fixed and abs(s - signs[name]) < 1e-12,
          f"Klein's form {name} (degree {len(F) - 1}, spin {(len(F) - 1) // 2}) is fixed by the 24 units; "
          f"the twist (1 + i)/sqrt 2 acts by {signs[name]:+d}")

hmats = su2(hurwitz)


def theta_of(F, Fp=None):
    """sum over Hurwitz quaternions x of <F', rho(x) F> q^N(x), normalised by its first coefficient when F' = F."""
    vals = inner(F if Fp is None else Fp, act(F, hmats))
    return np.bincount(hnorm, weights=vals.real, minlength=MH + 1), np.abs(vals.imag).max()


def close(series, exact):
    exact = np.array(exact[:MH + 1], dtype=float)
    return np.abs(series / series[1] - exact).max() < 1e-9 * np.abs(exact).max()


f8 = eta_quotient({1: 8, 2: 8}, MH)
E2 = [1] + [-24 * sum(d for d in range(1, n + 1) if n % d == 0) for n in range(1, MH + 1)]
E2_level2 = [2 * (E2[n // 2] if n % 2 == 0 else 0) - E2[n] for n in range(MH + 1)]
f10 = mul(f8, E2_level2, MH)
th_phi, _ = theta_of(PHI)
th_psi, _ = theta_of(PSI)
check(close(th_phi, f8), "spin 3 (octahedron): sum <Phi, x Phi> q^N(x) is eta(z)^8 eta(2z)^8 = q - 8q^2 + 12q^3 + 64q^4 "
      "- 210q^5 - ..., the newform of weight 8 and level 2")
check(close(th_psi, f10) and f10[1:6] == [1, 16, -156, 256, 870],
      "spin 4 (cube): sum <Psi, x Psi> q^N(x) is q + 16q^2 - 156q^3 + 256q^4 + 870q^5 + ..., the newform of weight 10 "
      "and level 2 (= eta(z)^8 eta(2z)^8 (2E_2(2z) - E_2(z)))")
th_x, _ = theta_of(XI)
th_p2, _ = theta_of(PHI2)
cross, _ = theta_of(XI, PHI2)
f14 = {"X": np.rint(th_x / th_x[1]).astype(np.int64), "Phi^2": np.rint(th_p2 / th_p2[1]).astype(np.int64)}
check(list(f14["X"][1:6]) == [1, -64, -1836, 4096, 3990] and list(f14["Phi^2"][1:6]) == [1, 64, 1236, 4096, -57450]
      and np.abs(cross).max() < 1e-6 * np.abs(th_x).max(),
      "spin 6: X gives q - 64q^2 - 1836q^3 + 4096q^4 + 3990q^5 + ..., Phi^2 gives q + 64q^2 + 1236q^3 + 4096q^4 "
      "- 57450q^5 + ..., and the mixed series vanishes")


def hecke_ok(c, k):
    """multiplicativity and the Hecke relation at 3 and 5 for a normalised series of weight k and level 2."""
    c = [int(v) for v in c]
    mult = all(c[m * n] == c[m] * c[n] for m in range(2, 8) for n in range(2, 8) if math.gcd(m, n) == 1 and m * n <= MH)
    return mult and c[9] == c[3] ** 2 - 3 ** (k - 1) and c[25] == c[5] ** 2 - 5 ** (k - 1) and c[4] == c[2] ** 2


check(hecke_ok(f8, 8) and hecke_ok(f10, 10) and hecke_ok(f14["X"], 14) and hecke_ok(f14["Phi^2"], 14),
      "all four series are Hecke eigenforms: a(mn) = a(m) a(n) for coprime m, n; a(p^2) = a(p)^2 - p^(k-1), p = 3, 5; "
      "a(4) = a(2)^2")
table = {"Phi": (f8, 8), "Psi": (f10, 10), "X": (f14["X"], 14), "Phi^2": (f14["Phi^2"], 14)}
check(all(int(c[2]) == signs[name] * 2 ** (k // 2 - 1) for name, (c, k) in table.items()),
      "a(2) = (sign under the twist) 2^(l/2): -8, +16, -64, +64")


def fricke_sign(c, k):
    """w with f(-1/(2z)) = w 2^(k/2) z^k f(z), evaluated at z = 0.8 i/sqrt 2 from the q-series."""
    z = mp.mpc(0, 0.8 / mp.sqrt(2))
    f = lambda w_: sum(int(c[n]) * mp.e ** (2j * mp.pi * n * w_) for n in range(1, MH + 1))
    return f(-1 / (2 * z)) / (2 ** (k / 2) * z ** k * f(z))


roots = {}
for name, (c, k) in table.items():
    w = fricke_sign(c, k)
    roots[name] = round(((-1) ** (k // 2) * w).real)
    check(abs(w - round(w.real)) < 1e-8,
          f"{name}: Fricke eigenvalue {round(w.real):+d}, root number (-1)^(k/2) w = {roots[name]:+d}")
check(all(roots[name] == (-1) ** ((k - 2) // 2) * signs[name] for name, (c, k) in table.items()),
      "root number = (-1)^j (sign of the spin-j state under the twist): +1, +1, -1, +1")


def chi4(n):
    return 0 if n % 2 == 0 else (1 if n % 4 == 1 else -1)


check(all(f10[p] == a[p * p] + chi4(p) * p ** 4 for p in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)),
      "Shimura: the weight-10 form has a(p) = a_g(p^2) + chi_-4(p) p^4 for 3 <= p <= 47: it is the lift of g")


def molien(l):
    """dimension of the forms of degree l fixed by the binary tetrahedral group: (1 + t^12)/((1 - t^6)(1 - t^8))."""
    count = lambda n: sum(1 for i in range(n // 6 + 1) if n >= 0 and (n - 6 * i) % 8 == 0)
    return count(l) + count(l - 12)


def dim_new_level2(k):
    """dim S_k^new(Gamma_0(2)) for even k >= 2 from the dimension formulas at levels 1 and 2."""
    if k < 4:
        return 0
    level1 = k // 12 - 1 if k % 12 == 2 else k // 12
    return (k // 4 - 1) - 2 * level1


check(all(molien(l) == dim_new_level2(l + 2) for l in range(2, 601, 2)) and all(molien(l) == 0 for l in range(1, 601, 2)),
      "Molien = Eichler: the number of spin states of degree l fixed by the Hurwitz units equals the number of "
      "newforms of weight l + 2 and level 2, for 1 <= l <= 600")
first = [l for l in range(1, 40) if molien(l)]
check(first[:8] == [6, 8, 12, 14, 16, 18, 20, 22] and molien(12) == 2,
      "first invariant states: spin 3, 4, 6 (two), 7, 8, 9 (two), ...; hydrogen shells n = 7, 9, 13, ...")

# ---------------------------------------------------------------------------------------------------
print("(6) rotations from quaternions")
V_GENS = np.array([(1, 2, 0, 0), (1, -2, 0, 0), (1, 0, 2, 0), (1, 0, -2, 0), (1, 0, 0, 2), (1, 0, 0, -2)], dtype=float)


def spin_matrices(j):
    """J_x, J_y, J_z on the representation of spin j (dimension 2j + 1), in the basis m = j, j - 1, ..., -j."""
    m = np.arange(j, -j - 1, -1.0)
    up = np.sqrt(j * (j + 1) - m[1:] * (m[1:] + 1))          # <m + 1| J_+ |m>
    Jp = np.diag(up, 1)
    return (Jp + Jp.T) / 2, (Jp - Jp.T) / 2j, np.diag(m)


def cos_of(J, angle):
    """cos(angle J) for a Hermitian matrix J."""
    vals, vecs = np.linalg.eigh(J)
    return (vecs * np.cos(angle * vals)) @ vecs.conj().T


LMAX = 40
ANGLE = math.acos(-3 / 5)          # the quaternions 1 +- 2i, 1 +- 2j, 1 +- 2k rotate by this angle about the axes
eigs = []
for l in range(1, LMAX + 1):
    # a rotation by ANGLE about an axis and its inverse add up to 2 cos(ANGLE J_axis)
    T = 2 * sum(cos_of(J, ANGLE) for J in spin_matrices(l))
    eigs.extend(np.linalg.eigvalsh(T))
eigs = np.array(eigs)
bound = 2 * math.sqrt(5)
check(len(eigs) == (LMAX + 1) ** 2 - 1 and abs(2 * math.acos(1 / math.sqrt(5)) - ANGLE) < 1e-12,
      f"the sum of the six rotations is T = 2 (cos(t J_x) + cos(t J_y) + cos(t J_z)), cos t = -3/5; on the harmonics "
      f"of degree 1 <= l <= {LMAX} it has {len(eigs)} eigenvalues")
check(abs(eigs[:3] + 0.4).max() < 1e-12, "l = 1: T = 2 + 4 cos(theta) = -2/5 on vectors, for rotations by arccos(-3/5)")
check(np.abs(eigs).max() < bound,
      f"Lubotzky-Phillips-Sarnak: every eigenvalue has absolute value at most 2 sqrt 5 = {bound:.4f} "
      f"(largest {np.abs(eigs).max():.4f}); the trivial bound is 6")


def kesten_mckay_pdf(x, q=5):
    return (q + 1) * np.sqrt(np.maximum(4 * q - x ** 2, 0)) / (2 * math.pi * ((q + 1) ** 2 - x ** 2))


xg = np.linspace(-bound, bound, 20001)
cdf = np.concatenate([[0], np.cumsum((kesten_mckay_pdf(xg[1:]) + kesten_mckay_pdf(xg[:-1])) / 2 * np.diff(xg))])
check(abs(cdf[-1] - 1) < 1e-6, "the Kesten-McKay density of the 6-regular tree integrates to 1 on [-2 sqrt 5, 2 sqrt 5]")
es = np.sort(eigs)
ks6 = np.max(np.abs(np.interp(es, xg, cdf) - (np.arange(1, len(es) + 1) - 0.5) / len(es)))
check(ks6 < 0.03, f"the eigenvalues follow the Kesten-McKay law (KS distance {ks6:.4f})")

# words in the six generators: a free group
words = {tuple(int(v) for v in q): i for i, q in enumerate(V_GENS)}
level = dict(words)
sizes = [len(level)]
all_levels = [dict(level)]
for _ in range(5):
    nxt = {}
    for q, last in level.items():
        for i, gq in enumerate(V_GENS):
            if i == (last ^ 1):                 # no generator next to its inverse
                continue
            prod = tuple(int(v) for v in qmul(q, gq))
            if next(v for v in prod if v) < 0:      # q and -q are the same rotation
                prod = tuple(-v for v in prod)
            nxt[prod] = i
    level = nxt
    sizes.append(len(level))
    all_levels.append(dict(level))
check(sizes == [6 * 5 ** t for t in range(6)] and all(sum(v * v for v in q) == 5 ** 6 for q in level),
      "reduced words of length t <= 6 give 6 * 5^(t-1) distinct quaternions of norm 5^t: the six rotations generate a free group")


def rotate(qs, v):
    """the rotations x -> q x q^(-1) applied to a vector v."""
    qs = np.asarray(qs, dtype=float)
    qs = qs / np.linalg.norm(qs, axis=1, keepdims=True)
    w, xyz = qs[:, 0], qs[:, 1:]
    t = 2 * np.cross(xyz, v)
    return v + w[:, None] * t + np.cross(xyz, t)


start = np.array([0.36, 0.48, 0.8])
orbit = rotate(list(all_levels[4].keys()), start)
heights = np.sort(orbit[:, 2])
ks_orbit = np.max(np.abs((heights + 1) / 2 - (np.arange(1, len(heights) + 1) - 0.5) / len(heights)))
check(len(orbit) == 3750 and ks_orbit < 0.02,
      f"the 3750 images of a point under the words of length 5 are evenly spread: heights uniform to KS {ks_orbit:.4f}")

# ---------------------------------------------------------------------------------------------------
print("figures")
YELLOW = "#eda100"

# echoes
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"wspace": 0.3})
ax = axes[0]
top = 20
for n in range(1, top + 1):
    if r3[n]:
        ax.plot([math.sqrt(n)] * 2, [0, r3[n] / (4 * math.pi * math.sqrt(n))], color=BLUE, lw=2, solid_capstyle="round")
    else:
        ax.plot([math.sqrt(n)], [0], marker="x", color=ORANGE, ms=7, mew=2, clip_on=False, zorder=5)
for n in (7, 15):
    ax.text(math.sqrt(n), 0.045, str(n), ha="center", fontsize=8, color=INK_2)
ax.plot([0.86], [1.115], marker="x", color=ORANGE, ms=6, mew=1.8)
ax.text(0.96, 1.115, "no echo: $N=7,\\,15$", va="center", fontsize=8.5, color=INK)
for n in (1, 2, 3, 5, 6, 9, 14):
    ax.text(math.sqrt(n), r3[n] / (4 * math.pi * math.sqrt(n)) + 0.03, str(n), ha="center", fontsize=8, color=INK_2)
ax.set_xlim(0.7, 4.6)
ax.set_ylim(0, 1.2)
ax.set_xlabel("arrival time  $t=\\sqrt{N}$  (units of $a/c$)")
ax.set_ylabel("echo strength  $r_3(N)/4\\pi\\sqrt{N}$")
ax.set_title("Echoes of a cubic lattice", loc="left")
ax.grid(axis="y")
ax = axes[1]
tt = np.linspace(0.75, 4.25, 900)
exact = np.array([echoes_smoothed(t) for t in tt])
ax.plot(tt, exact, color=BLUE, lw=2.6, alpha=0.55, label="echoes, smoothed")
ax.plot(tt, [modes_smoothed(t, 40) for t in tt], color=ORANGE, lw=1.1, label="40 lowest tones")
ax.plot(tt, tt, color=INK_2, lw=0.8, ls=(0, (4, 3)), label="mean growth $t$")
ax.set_xlim(0.75, 4.25)
ax.set_xlabel("time  $t$")
ax.set_ylabel("signal at a lattice point")
ax.set_title("The same signal from the normal modes", loc="left")
ax.legend(loc="upper left", fontsize=8, handlelength=1.6)
ax.grid(axis="y")
save(fig, "spheres-echoes")

# roundness
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.3})
ax = axes[0]
c2, r2 = hexadecapole(2, 20000)
i2 = np.nonzero((r2[1:] > 0) & (c2[1:] != 0))[0] + 1
mu2 = np.abs(c2[i2]) / 4 / (r2[i2] * i2.astype(float) ** 2)
keep3 = (c3_big[idx3] != 0) & (idx3 > 200)
sub3 = rng.choice(np.nonzero(keep3)[0], size=2500, replace=False)
ax.scatter(r2[i2], mu2, s=5, color=AQUA, alpha=0.5, lw=0, label="circles, $d=2$", rasterized=True)
ax.scatter(r3_big[idx3][sub3], mu3[sub3], s=5, color=BLUE, alpha=0.5, lw=0, label="spheres, $d=3$", rasterized=True)
ax.scatter(8 * (primes4 + 1), mu4, s=5, color=ORANGE, alpha=0.5, lw=0, label="3-spheres, $d=4$", rasterized=True)
xr = np.array([4.0, 2e5])
ax.plot(xr, 0.37 / np.sqrt(xr), color=INK, lw=1, ls=(0, (4, 3)))
ax.text(2.5e5, 0.3, "dashed: $0.37/\\sqrt{\\mathrm{number\\ of\\ points}}$", fontsize=8.5, color=INK, ha="right")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(3, 3e5)
ax.set_ylim(2e-5, 0.6)
ax.set_xlabel("number of lattice points on the shell")
ax.set_ylabel("anisotropy  $|\\mu_d(N)|$")
ax.set_title("How round is a shell?", loc="left")
ax.legend(loc="lower left", fontsize=8, markerscale=2.5, handletextpad=0.2)
ax.grid(True, which="major")
ax = axes[1]
ax.hist(st, bins=24, range=(-1, 1), density=True, color=ORANGE, alpha=0.55, edgecolor="white", lw=0.6)
xx = np.linspace(-1, 1, 400)
ax.plot(xx, 2 / math.pi * np.sqrt(1 - xx ** 2), color=INK, lw=1.2)
ax.text(0.0, 0.3, "semicircle\n(Sato–Tate)", ha="center", fontsize=8.5, color=INK)
ax.set_xlabel("$a(p)/2p^{5/2}$ for $\\eta(2z)^{12}$")
ax.set_ylabel("density")
ax.set_title("Within Deligne's bound", loc="left")
ax.set_xlim(-1.05, 1.05)
fig.set_dpi(200)
save(fig, "spheres-roundness")


# constellations: Husimi densities of Klein's forms on the sphere
def husimi(F, zeta):
    l = len(F) - 1
    val = sum(F[k] * zeta ** (l - k) for k in range(l + 1))
    return np.abs(val) ** 2 / (1 + np.abs(zeta) ** 2) ** l


view = np.array([[0.85, 0.38, 0.27]])
view = view / np.linalg.norm(view)
ex = np.cross([0, 0, 1.0], view[0])
ex /= np.linalg.norm(ex)
ey = np.cross(view[0], ex)
px = np.linspace(-1, 1, 700)
PX, PY = np.meshgrid(px, px)
inside = PX ** 2 + PY ** 2 <= 1
PZ = np.sqrt(np.where(inside, 1 - PX ** 2 - PY ** 2, 0))
pts3 = PX[..., None] * ex + PY[..., None] * ey + PZ[..., None] * view[0]
zeta = (pts3[..., 0] + 1j * pts3[..., 1]) / np.where(np.abs(1 - pts3[..., 2]) < 1e-9, 1e-9, 1 - pts3[..., 2])
cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
octahedron = [np.array(v, dtype=float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
cube = [np.array(v) / math.sqrt(3) for v in itertools.product((1, -1), repeat=3)]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3))
for ax, F, stars, title in ((axes[0], PHI, octahedron, "Spin 3: the octahedron $\\Phi$"),
                            (axes[1], PSI, cube, "Spin 4: the cube $\\Psi$")):
    Q = np.where(inside, husimi(F, zeta), np.nan)
    ax.imshow(Q / np.nanmax(Q), extent=(-1, 1, -1, 1), origin="lower", cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=INK_2, lw=0.8))
    shown = 0
    for s in stars:
        if s @ view[0] > 0.02:
            ax.plot([s @ ex], [s @ ey], marker="o", ms=5.5, mfc="white", mec=INK, mew=1.0)
            shown += 1
    ax.set_title(title, loc="left")
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-1.06, 1.06)
    ax.set_aspect("equal")
    ax.axis("off")
    # the zeros of the form are its stars
    for s in stars:
        zs = (s[0] + 1j * s[1]) / (1 - s[2]) if abs(1 - s[2]) > 1e-9 else None
        if zs is not None:
            assert husimi(F, np.array(zs)) < 1e-18
save(fig, "spheres-constellations")
check(True, "the stars of Phi are the six vertices of the octahedron, those of Psi the eight vertices of the cube")

# rotations
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.26})
ax = axes[0]
ax.hist(eigs, bins=36, range=(-bound, bound), density=True, color=BLUE, alpha=0.55, edgecolor="white", lw=0.6)
ax.plot(xg, kesten_mckay_pdf(xg), color=INK, lw=1.2)
ax.axvline(-bound, color=ORANGE, lw=1)
ax.axvline(bound, color=ORANGE, lw=1)
ax.text(bound + 0.2, 0.09, "$2\\sqrt{5}$", ha="left", va="center", fontsize=9, color=INK)
ax.text(-bound - 0.2, 0.09, "$-2\\sqrt{5}$", ha="right", va="center", fontsize=9, color=INK)
ax.annotate("Kesten–McKay law", xy=(-3.6, kesten_mckay_pdf(np.array([-3.6]))[0]), xytext=(-2.6, 0.162), fontsize=8.5,
            color=INK, arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.6))
ax.set_xlim(-7.4, 7.4)
ax.set_ylim(0, 0.18)
ax.set_xlabel("eigenvalue of the six rotations")
ax.set_ylabel("density")
ax.set_title("Spectrum on the sphere, $l\\leq40$", loc="left")
ax = axes[1]
lon = np.degrees(np.arctan2(orbit[:, 1], orbit[:, 0]))
ax.scatter(lon, orbit[:, 2], s=2.2, color=BLUE, lw=0, rasterized=True)
ax.set_xlim(-180, 180)
ax.set_ylim(-1, 1)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_xlabel("longitude (degrees)")
ax.set_ylabel("height")
ax.set_title("3750 images of one point (words of length 5)", loc="left")
fig.set_dpi(200)
save(fig, "spheres-rotations")

print("all checks passed")
