"""Note 06: towards a number-theoretic standard model. The exact facts the questions rest on.

What it computes (exact rational arithmetic, except the L-function checks of (3b), which are in floating point)
  (1) Counts. Twelve fermion flavours (six quarks, six leptons); twelve gauge bosons, the dimension of
      SU(3) x SU(2) x U(1) = 8 + 3 + 1; fifteen Weyl fermions per generation (sixteen with a right-handed
      neutrino). The Higgs is the one scalar; the gauge bosons are vectors.
  (2) Balance. The six anomaly-cancellation conditions of one generation: SU(3)^3, SU(3)^2 U(1), SU(2)^2 U(1),
      U(1)^3, gravitational U(1), and Witten's SU(2) condition (an even number of doublets). All hold exactly.
  (3) The balance fixes the charges of one generation of fifteen. With the Standard Model's representations and
      unknown rational hypercharges, the linear and cubic conditions have, up to scale and swapping u and d,
      the single solution with y_Q != 0: the observed hypercharges (1/6, -2/3, 1/3, -1/2, 1). Checked by an
      exhaustive integer search and by algebra. Without the gravitational condition the cubic condition is
      Fermat's equation y_e^3 + (y_u - 2 y_Q)^3 = (y_u + 4 y_Q)^3 (Lohitsiri and Tong), whose rational
      solutions are the trivial ones: y_Q = 0, or the same hypercharges; the gravitational condition follows.
      Scope, each point checked: over the reals the gauge conditions have solutions that violate the
      gravitational one; with a right-handed neutrino the conditions hold for Y + eps (B - L), every eps; the
      electric charges need the Higgs doublet as well (Q = T3 + Y/(2 y_H), y_H = -y_Q - y_u); with N colours
      the cubic condition is y_e^3 + (y_u + (1 - N) y_Q)^3 = (y_u + (N + 1) y_Q)^3, Fermat's cubic for every N.
  (3b) The hypercharge curve. The solutions, up to scale, are the three rational points of the elliptic curve
      X^3 + Y^3 = Z^3, a group of order 3 in which u <-> d is negation. The curve has complex multiplication by
      Z[omega]; its number of points modulo p is p + 1 - a(p) with sum a(n) q^n = eta(3z)^2 eta(9z)^2 (level 27),
      a(p) = 0 for p = 2 (mod 3) and Gauss's formula for p = 1 (mod 3). Its L-function has root number +1 and
      central value L(1) = Gamma(1/3)^3/(6 pi sqrt 3) = 0.5888795834..., which is not zero.
  (4) The hidden Z/6. The element (e^(2 pi i/3) I_3, -I_2, e^(2 pi i/6)) of SU(3) x SU(2) x U(1) acts
      trivially on every field, and the subgroup of the centre and U(1) acting trivially is exactly Z/6, so the
      group acting faithfully on the fields is (SU(3) x SU(2) x U(1))/Z6. (Which quotient by a subgroup of Z/6 is
      nature's gauge group is an open question: Tong, JHEP 07 (2017) 104.) For comparison: PSL(2, Z) = Z/2 * Z/3 has abelianisation Z/6
      (note 01: the characters of SL(2, Z) trivial on -1 are the six even powers of the eta^2 multiplier).
  (5) Colour lives on the Eisenstein integers. The roots of SU(3) are the six units of Z[omega], so its root
      lattice is Z[omega]; the colour triplet's weights are 1/(1 - omega) times the cube roots of unity; the
      weight lattice is (1 - omega)^(-1) Z[omega], and the centre Z/3 is the residue field Z[omega]/(1 - omega)
      at the ramified prime sqrt(-3). The six roots are one orbit of the stabiliser of the Eisenstein point rho
      in SL(2, Z): ST acts on Z + Z rho as multiplication by e^(2 pi i/6). The roots of SU(2) are the units of Z.
  (6) Chirality. One generation differs from its conjugate: the Standard Model is chiral.

What it shows
  Exact arithmetic facts about the Standard Model's representations: the conditions that make it consistent are
  Diophantine, its colour group is built on the Eisenstein integers, and its fields are blind to the same Z/6
  that abelianises PSL(2, Z). The counts, the shared lattice and the Z/6 are the subject of the questions in note 06.
"""
import cmath
import itertools
import math
from fractions import Fraction as F

import matplotlib.pyplot as plt
import numpy as np

from common import BLUE, GRID, INK, INK_2, ORANGE, check, save

# one generation, all fields left-handed Weyl: (name, colour: +1 triplet / -1 antitriplet / 0 singlet,
# SU(2) dimension, hypercharge)
GEN = [("Q", 1, 2, F(1, 6)), ("u^c", -1, 1, F(-2, 3)), ("d^c", -1, 1, F(1, 3)), ("L", 0, 2, F(-1, 2)),
       ("e^c", 0, 1, F(1))]
NU = ("nu^c", 0, 1, F(0))
HIGGS = ("H", 0, 2, F(1, 2))


def dim3(c):
    return 3 if c else 1


# ---------------------------------------------------------------------------------------------------
print("(1) counts")
check(6 + 6 == 12 and 8 + 3 + 1 == 12, "12 fermion flavours (u d c s t b; e mu tau and three neutrinos); "
                                      "12 gauge bosons = dim SU(3) + dim SU(2) + dim U(1) = 8 + 3 + 1")
weyl = sum(dim3(c) * d2 for _, c, d2, _ in GEN)
check(weyl == 15, f"{weyl} Weyl fermions per generation (16 with a right-handed neutrino)")


# ---------------------------------------------------------------------------------------------------
def anomalies(fields):
    return {
        "SU(3)^3": sum(c * d2 for _, c, d2, _ in fields),
        "SU(3)^2 U(1)": sum(d2 * y for _, c, d2, y in fields if c),
        "SU(2)^2 U(1)": sum(dim3(c) * y for _, c, d2, y in fields if d2 == 2),
        "U(1)^3": sum(dim3(c) * d2 * y ** 3 for _, c, d2, y in fields),
        "gravity-U(1)": sum(dim3(c) * d2 * y for _, c, d2, y in fields),
    }


print("(2) balance: anomaly cancellation")
for gen, label in ((GEN, "one generation"), (GEN + [NU], "with a right-handed neutrino")):
    an = anomalies(gen)
    check(all(v == 0 for v in an.values()), f"{label}: " + ", ".join(f"{k} = {v}" for k, v in an.items()))
doublets = sum(dim3(c) for _, c, d2, _ in GEN if d2 == 2)
check(doublets % 2 == 0, f"Witten's condition: {doublets} SU(2) doublets per generation, an even number")
charges = [(T3, y) for _, c, d2, y in GEN for T3 in ((F(1, 2), F(-1, 2)) if d2 == 2 else (F(0),))
           for _ in range(dim3(c))]
check(sum(T3 + y for T3, y in charges) == 0, "the electric charges Q = T3 + Y of a generation sum to zero")

# ---------------------------------------------------------------------------------------------------
print("(3) the balance fixes the charges")
sols = []
R = range(-40, 41)
for u, d, L, e in itertools.product(R, R, R, R):
    q = 1                               # hypercharges in units of y_Q (so the Standard Model is 6 x Y)
    if 2 * q + u + d == 0 and 3 * q + L == 0 and 6 * q + 3 * u + 3 * d + 2 * L + e == 0 \
            and 6 * q ** 3 + 3 * u ** 3 + 3 * d ** 3 + 2 * L ** 3 + e ** 3 == 0:
        sols.append((u, d, L, e))
check(sorted(sols) == [(-4, 2, -3, 6), (2, -4, -3, 6)],
      f"y_Q = 1: the only integer solutions with |y| <= 40 are {sorted(sols)}, the Standard Model's (x6) up to u <-> d")
check([6 * y for _, _, _, y in GEN] == [1, -4, 2, -3, 6], "6 Y = (1, -4, 2, -3, 6) for (Q, u^c, d^c, L, e^c)")

# without the gravitational condition: Fermat's cubic (Lohitsiri and Tong)
grid = range(-3, 4)
check(all(6 * q ** 3 + 3 * u ** 3 + 3 * (-2 * q - u) ** 3 + 2 * (-3 * q) ** 3 + e ** 3
          == e ** 3 + (u - 2 * q) ** 3 - (u + 4 * q) ** 3 for q in grid for u in grid for e in grid),
      "with y_L = -3 y_Q and y_d = -2 y_Q - y_u, the cubic condition is y_e^3 + (y_u - 2 y_Q)^3 = (y_u + 4 y_Q)^3")
gauge_only = []
for q in (0, 1):
    for u, e in itertools.product(R, R):
        d, L = -2 * q - u, -3 * q
        if 6 * q ** 3 + 3 * u ** 3 + 3 * d ** 3 + 2 * L ** 3 + e ** 3 == 0:
            gauge_only.append((q, u, d, L, e))
trivial = all(e == 0 or u == 2 * q or u == -4 * q for q, u, d, L, e in gauge_only)
gravity = all(6 * q + 3 * u + 3 * d + 2 * L + e == 0 for q, u, d, L, e in gauge_only)
check(trivial and gravity and sorted(t for t in gauge_only if t[0] == 1) == [(1, -4, 2, -3, 6), (1, 2, -4, -3, 6)]
      and all(e == 0 and d == -u for q, u, d, L, e in gauge_only if q == 0),
      f"gauge conditions alone, |y_u|, |y_e| <= 40: {len(gauge_only)} solutions, all with one cube zero (Fermat), "
      "all satisfying the gravitational condition; for y_Q = 1 they are the Standard Model's two")

# scope: quantisation, the right-handed neutrino, the Higgs, the number of colours
q_, u_ = 1.0, 0.0
d_, L_, e_ = -2 * q_ - u_, -3 * q_, 72 ** (1 / 3)
real_cubic = 6 * q_ ** 3 + 3 * u_ ** 3 + 3 * d_ ** 3 + 2 * L_ ** 3 + e_ ** 3
real_grav = 6 * q_ + 3 * u_ + 3 * d_ + 2 * L_ + e_
check(abs(real_cubic) < 1e-12 and abs(real_grav - (72 ** (1 / 3) - 6)) < 1e-12 and real_grav < -1.8,
      f"over the reals: (y_Q, y_u, y_d, y_L, y_e) = (1, 0, -2, -3, 72^(1/3)) satisfies the gauge conditions "
      f"and has gravitational sum {real_grav:.3f}: quantised hypercharge is a hypothesis")
B_MINUS_L = {"Q": F(1, 3), "u^c": F(-1, 3), "d^c": F(-1, 3), "L": F(-1), "e^c": F(1), "nu^c": F(1)}
check(all(all(v == 0 for v in anomalies([(n, c, d2, y + eps * B_MINUS_L[n]) for n, c, d2, y in GEN + [NU]]).values())
          for eps in (F(1, 7), F(-2, 3), F(5))),
      "with a right-handed neutrino every Y + eps (B - L) satisfies all the conditions: sixteen fermions do not "
      "fix their hypercharges")
for yq in (F(1, 6), F(5, 7)):
    yu, yd, yL, ye = -4 * yq, 2 * yq, -3 * yq, 6 * yq
    yH = -yq - yu                                   # the Yukawa coupling Q u^c H is neutral
    lam = 1 / (2 * yH)                              # the charge T3 + lam Y that leaves the Higgs vacuum neutral
    observed = [F(1, 2) + lam * yq, F(-1, 2) + lam * yq, F(1, 2) + lam * yL, F(-1, 2) + lam * yL,
                lam * yu, lam * yd, lam * ye]
    check(yH == 3 * yq and observed == [F(2, 3), F(-1, 3), F(0), F(-1), F(-2, 3), F(1, 3), F(1)]
          and F(1, 2) + 2 * lam * yq == F(5, 6),
          f"y_Q = {yq}: with the Higgs, y_H = -y_Q - y_u = 3 y_Q and Q = T3 + Y/(2 y_H) gives "
          "(u, d, nu, e, u^c, d^c, e^c) = (2/3, -1/3, 0, -1, -2/3, 1/3, 1); another multiple of Y does not")


def cubic_colours(N, q, u, e):
    """the U(1)^3 condition with N colours, after y_L = -N y_Q and y_d = -2 y_Q - y_u."""
    return 2 * N * q ** 3 + N * u ** 3 + N * (-2 * q - u) ** 3 + 2 * (-N * q) ** 3 + e ** 3


check(all(cubic_colours(N, q, u, e) == e ** 3 + (u + (1 - N) * q) ** 3 - (u + (N + 1) * q) ** 3
          for N in range(1, 13) for q in grid for u in grid for e in grid),
      "N colours, N = 1, ..., 12: the cubic condition is y_e^3 + (y_u + (1 - N) y_Q)^3 = (y_u + (N + 1) y_Q)^3 "
      "(an identity of cubic polynomials, checked on a 7 x 7 x 7 grid)")
check(all(cubic_colours(N, 1, (N - 1), 2 * N) == 0 and cubic_colours(N, 1, -(N + 1), 2 * N) == 0
          and 2 * N + N * (-2) + 2 * (-N) + 2 * N == 0
          and F(1, 2) + F(1, 2 * N) * (-N) == 0 and F(1, 2 * N) * 2 * N == 1 for N in range(1, 13)),
      "N colours: {y_u, y_d} = {(N - 1) y_Q, -(N + 1) y_Q}, y_L = -N y_Q, y_e = 2N y_Q solve all four conditions; "
      "with y_H = N y_Q the neutrino is neutral, e^c has charge 1 and the quarks have 1/2 + 1/(2N), -1/2 + 1/(2N)")

# ---------------------------------------------------------------------------------------------------
print("(3b) the hypercharge curve X^3 + Y^3 = Z^3 and its L-function")
# (X, Y, Z) = (y_e, y_u + (1 - N) y_Q, y_u + (N + 1) y_Q); for N = 3: (y_e, y_u - 2 y_Q, y_u + 4 y_Q)
points = {"y_Q = 0": (0, 1, 0), "observed": (1, -4, 6), "observed, u <-> d": (1, 2, 6)}      # (y_Q, y_u, y_e)
XYZ = {k: (e, u - 2 * q, u + 4 * q) for k, (q, u, e) in points.items()}
O_, P_, Q_ = XYZ["y_Q = 0"], XYZ["observed"], XYZ["observed, u <-> d"]
det3 = lambda r1, r2, r3: (r1[0] * (r2[1] * r3[2] - r2[2] * r3[1]) - r1[1] * (r2[0] * r3[2] - r2[2] * r3[0])
                           + r1[2] * (r2[0] * r3[1] - r2[1] * r3[0]))
sigma = lambda pt: (pt[0], -pt[2], -pt[1])                  # u <-> d
proj_eq = lambda s_, t_: all(s_[i] * t_[j] == s_[j] * t_[i] for i in range(3) for j in range(3))
check((O_, P_, Q_) == ((0, 1, 1), (6, -6, 0), (6, 0, 6)) and det3(O_, P_, Q_) == 0
      and all(X ** 3 + Y ** 3 == Z ** 3 and X + Y == Z and X * Y * Z == 0 for X, Y, Z in (O_, P_, Q_)),
      "the three solutions are the points (0:1:1), (1:-1:0) (observed), (1:0:1) of the cubic: flexes on the line X + Y = Z")
check(all((e, (-2 * q - u) + (1 - N) * q, (-2 * q - u) + (N + 1) * q) == sigma((e, u + (1 - N) * q, u + (N + 1) * q))
          for N in range(1, 13) for q in grid for u in grid for e in grid)
      and proj_eq(sigma(O_), O_) and proj_eq(sigma(P_), Q_)
      and all(det3(O_, (X, Y, Z), sigma((X, Y, Z))) == 0 for X in grid for Y in grid for Z in grid),
      "exchanging u and d is (X : Y : Z) -> (X : -Z : -Y) for N = 1, ..., 12; it fixes (0:1:1), swaps the other two, "
      "and O, T, sigma(T) are collinear for every T: negation in the group with origin O, of order 3 on these points")
check(all(1728 - 432 * (x + y) ** 3 - 1296 * (x - y) ** 2 * (x + y) == 1728 * (1 - x ** 3 - y ** 3)
          for x in grid for y in grid)
      and all((4 * (2 * y + 1)) ** 2 - ((4 * x) ** 3 - 432) == 64 * (y * y + y - x ** 3 + 7) for x in grid for y in grid),
      "x^3 + y^3 = 1 maps to Y^2 = X^3 - 432 by X = 12/(x + y), Y = 36 (x - y)/(x + y); with X = 4x', "
      "Y = 4(2y' + 1) this is y'^2 + y' = x'^3 - 7 (LMFDB 27.a3)")
MC = 400
eta27 = [1] + [0] * MC                                   # prod (1 - q^(3n))^2 (1 - q^(9n))^2
for step in (3, 3, 9, 9):
    for n_ in range(step, MC + 1, step):
        for i_ in range(MC, n_ - 1, -1):
            eta27[i_] -= eta27[i_ - n_]
a27 = [0] + eta27[:MC]                                   # eta(3z)^2 eta(9z)^2 = sum a27[n] q^n
check(a27[1:20] == [1, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0, 0, 5, 0, 0, 4, 0, 0, -7],
      "eta(3z)^2 eta(9z)^2 = q - 2q^4 - q^7 + 5q^13 + 4q^16 - 7q^19 + ...")
primes_c = [p for p in range(2, MC) if all(p % d for d in range(2, int(p ** 0.5) + 1)) and p != 3]
counts_ok, gauss_ok, inert_ok = True, True, True
for p in primes_c:
    mult = {}
    for x in range(p):
        mult[pow(x, 3, p)] = mult.get(pow(x, 3, p), 0) + 1
    affine = sum(mult[u_] * mult[v_] * mult.get((u_ + v_) % p, 0) for u_ in mult for v_ in mult)
    projective = (affine - 1) // (p - 1)
    counts_ok = counts_ok and projective == p + 1 - a27[p]
    if p % 3 == 2:
        inert_ok = inert_ok and affine == p * p and a27[p] == 0
    else:
        A = [A_ for A_ in range(-2 * p, 2 * p) if A_ % 3 == 1 and 4 * p - A_ * A_ > 0
             and (4 * p - A_ * A_) % 27 == 0 and math.isqrt((4 * p - A_ * A_) // 27) ** 2 == (4 * p - A_ * A_) // 27]
        gauss_ok = gauss_ok and len(A) == 1 and a27[p] == -A[0]
check(counts_ok, f"for the {len(primes_c)} primes p < {MC}, p != 3: the cubic has p + 1 - a(p) points modulo p, "
                 "a(p) the coefficient of eta(3z)^2 eta(9z)^2")
check(inert_ok, "p = 2 (mod 3): p^2 solutions (x, y, z) of x^3 + y^3 = z^3 modulo p, and a(p) = 0")
check(gauss_ok, "p = 1 (mod 3): a(p) = -A with 4p = A^2 + 27 B^2, A = 1 (mod 3) (Gauss)")
check((5 + 1 - a27[5], 7 + 1 - a27[7], math.gcd(5 + 1 - a27[5], 7 + 1 - a27[7])) == (6, 9, 3),
      "6 points modulo 5 and 9 modulo 7: a finite group of rational points has order dividing 3")
G27 = lambda y_: sum(a27[n_] * math.exp(-2 * math.pi * n_ * y_ / math.sqrt(27)) for n_ in range(1, MC))
check(all(abs(G27(1 / y_) / (y_ * y_ * G27(y_)) - 1) < 1e-12 for y_ in (0.7, 1.3, 2.1)),
      "f(i/(sqrt 27 y)) = y^2 f(i y/sqrt 27): the functional equation Lambda(2 - s) = Lambda(s), root number +1")
L27 = 2 * sum(a27[n_] / n_ * math.exp(-2 * math.pi * n_ / math.sqrt(27)) for n_ in range(1, MC))
check(abs(L27 - math.gamma(1 / 3) ** 3 / (6 * math.pi * math.sqrt(3))) < 1e-13 and abs(L27 - 0.5888795834284833) < 1e-13,
      f"the central value L(1) = {L27:.16f} = Omega/3, Omega = Gamma(1/3)^3/(2 pi sqrt 3) the real period; not zero")

# ---------------------------------------------------------------------------------------------------
print("(4) the hidden Z/6")
fields = GEN + [NU, HIGGS]


def acts_trivially(k, eps, theta):
    """(omega^k I_3, (-1)^eps I_2, e^(2 pi i theta) acting with charge 6Y) on every field."""
    for _, c, d2, y in fields:
        phase = Fraction_mod1(F(k * c, 3) + (F(eps, 2) if d2 == 2 else 0) + 6 * y * theta)
        if phase != 0:
            return False
    return True


def Fraction_mod1(x):
    return x - math.floor(x)


kernel = [(k, eps, F(j, 36)) for k in range(3) for eps in range(2) for j in range(36)
          if acts_trivially(k, eps, F(j, 36))]
check(len(kernel) == 6 and (1, 1, F(1, 6)) in kernel,
      f"the elements acting trivially form a group of order {len(kernel)} generated by "
      "(omega I_3, -I_2, e^(2 pi i/6)): Z/6")
check(all(acts_trivially(k, e_, F(j, 36 * 5)) is False or F(j, 180) * 6 == int(F(j, 180) * 6)
          for k in range(3) for e_ in range(2) for j in range(180)), "no finer rotation of U(1) acts trivially")
psl_chars = [k for k in range(12) if (6 * k) % 12 == 0]
check(len(psl_chars) == 6, "PSL(2, Z): the characters of SL(2, Z) trivial on -1 (pitch 6) are the 6 even powers: Z/6")

# ---------------------------------------------------------------------------------------------------
print("(5) colour lives on the Eisenstein integers")
w = cmath.exp(2j * math.pi / 3)
a1, a2 = 1 + 0j, w
roots = [a1, a2, a1 + a2, -a1, -a2, -(a1 + a2)]
sixth = [cmath.exp(1j * math.pi * k / 3) for k in range(6)]
check(all(min(abs(r - s) for s in sixth) < 1e-12 for r in roots), "the six roots of SU(3) are the six units of Z[omega]")
ip = lambda x, y: (x * y.conjugate()).real
cartan = [[round(2 * ip(x, y) / ip(y, y)) for y in (a1, a2)] for x in (a1, a2)]
check(cartan == [[2, -1], [-1, 2]], "their Cartan matrix is that of A2: [[2, -1], [-1, 2]]")
lam1 = (2 * a1 + a2) / 3
check(abs(lam1 - 1 / (1 - w)) < 1e-12, "the highest weight of the triplet is 1/(1 - omega)")
triplet = [lam1, lam1 - a1, lam1 - a1 - a2]
check(all(abs(t - lam1 * w ** k) < 1e-12 for t, k in zip(triplet, (0, 1, 2))),
      "the three colours are 1/(1 - omega) times the cube roots of unity 1, omega, omega^2")
check(abs(abs(1 - w) ** 2 - 3) < 1e-12 and round(np.linalg.det(np.array(cartan))) == 3,
      "N(1 - omega) = 3 = det(Cartan) = [weights : roots]: the centre Z/3 is Z[omega]/(1 - omega)")
ST = ((0, -1), (1, 1))                                   # S T, which fixes rho = omega
fixed = (ST[0][0] * w + ST[0][1]) / (ST[1][0] * w + ST[1][1])
mult = ST[1][0] * w + ST[1][1]                           # c rho + d: how ST acts on the lattice Z + Z rho
check(abs(fixed - w) < 1e-12 and abs(mult - cmath.exp(1j * math.pi / 3)) < 1e-12,
      "ST fixes rho and acts on Z + Z rho as multiplication by rho + 1 = e^(2 pi i/6)")
orbit = [mult ** k for k in range(6)]
check(all(min(abs(r - o) for o in orbit) < 1e-12 for r in roots),
      "the six roots of SU(3) are one orbit of the stabiliser of rho (the units of Z[omega])")
units_Z = [u for u in range(-3, 4) if u != 0 and 1 % u == 0]
check(units_Z == [-1, 1] and len(roots) + len(units_Z) == 8 and 8 + (2 + 1 + 1) == 12,
      "roots of SU(2) = units of Z; 6 + 2 = 8 gauge bosons of non-zero weight, plus rank 2 + 1 + 1 = 4: twelve")

# ---------------------------------------------------------------------------------------------------
print("(6) chirality")
reps = {(c, d2, y) for _, c, d2, y in GEN}
conj = {(-c, d2, -y) for c, d2, y in reps}
check(reps != conj, "one generation differs from its conjugate: the Standard Model is chiral")

# ---------------------------------------------------------------------------------------------------
print("figures")

# (a) colour on the Eisenstein lattice
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
lat = [a + b * w for a in range(-4, 5) for b in range(-4, 5)]
lat = [z for z in lat if abs(z) < 2.3]
ax.plot([z.real for z in lat], [z.imag for z in lat], "o", ms=3, color=INK_2)
hexagon = roots[0:1] + [roots[2], roots[1], roots[3], roots[5], roots[4], roots[0]]
hexagon = sorted(roots, key=lambda z: cmath.phase(z)) + [sorted(roots, key=lambda z: cmath.phase(z))[0]]
ax.plot([z.real for z in hexagon], [z.imag for z in hexagon], color=BLUE, lw=1.2)
ax.plot([z.real for z in roots], [z.imag for z in roots], "o", ms=8, color=BLUE, mec="white", mew=1.5)
tri = triplet + triplet[:1]
ax.plot([z.real for z in tri], [z.imag for z in tri], color=ORANGE, lw=1.2)
ax.plot([z.real for z in triplet], [z.imag for z in triplet], "^", ms=9, color=ORANGE, mec="white", mew=1.2)
ax.plot([-z.real for z in triplet], [-z.imag for z in triplet], "v", ms=9, color="white", mec=ORANGE, mew=1.5)
for z, lab in zip(triplet, ("r", "g", "b")):
    ax.text(z.real * 1.75, z.imag * 1.75, lab, ha="center", va="center", fontsize=9, color=INK)
ax.text(0, -2.55, "dots: Z[ω]   blue: the six roots (gluons)\norange: quark colours   hollow: antiquarks",
        ha="center", fontsize=8, color=INK)
ax.set_xlim(-2.4, 2.4)
ax.set_ylim(-2.9, 2.4)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("SU(3) colour on the Eisenstein integers", fontsize=10)

# (b) the balance
ax = axes[1]
names = ["Q", "uᶜ", "dᶜ", "L", "eᶜ"]
lin = [float(dim3(c) * d2 * y) for _, c, d2, y in GEN]
cub = [float(dim3(c) * d2 * y ** 3) for _, c, d2, y in GEN]
x = np.arange(5)
ax.bar(x - 0.2, lin, width=0.38, color=BLUE, label="multiplicity × Y   (sum 0)")
ax.bar(x + 0.2, cub, width=0.38, color=ORANGE, label="multiplicity × Y³  (sum 0)")
ax.axhline(0, color=INK_2, lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("contribution")
ax.set_ylim(-2.2, 1.75)
ax.legend(loc="upper center", fontsize=8.5)
ax.set_title("One generation balances exactly", fontsize=10)
save(fig, "sm-colour-balance")
print("done")
