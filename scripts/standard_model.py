"""Note 06: towards a number-theoretic standard model. The exact facts the conjectures rest on.

What it computes (exact rational arithmetic throughout)
  (1) Counts. Twelve fermion flavours (six quarks, six leptons); twelve gauge bosons, the dimension of
      SU(3) x SU(2) x U(1) = 8 + 3 + 1; fifteen Weyl fermions per generation (sixteen with a right-handed
      neutrino). The Higgs is the one scalar; the gauge bosons are vectors.
  (2) Balance. The six anomaly-cancellation conditions of one generation: SU(3)^3, SU(3)^2 U(1), SU(2)^2 U(1),
      U(1)^3, gravitational U(1), and Witten's SU(2) condition (an even number of doublets). All hold exactly.
  (3) The balance fixes the charges. With the Standard Model's representations and unknown hypercharges, the
      linear and cubic conditions have, up to scale and swapping u and d, the single solution with y_Q != 0:
      the observed hypercharges (1/6, -2/3, 1/3, -1/2, 1). Checked by an exhaustive integer search and by
      algebra.
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
  that abelianises PSL(2, Z). The counts and the Z/6 match are recorded as conjectures in note 06.
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
