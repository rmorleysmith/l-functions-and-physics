"""Note 01: the twelve characters of the modular group, and the twelve tones.

What it computes
  (1) S^2 = -1 and (ST)^3 = -1 in SL(2, Z).
  (2) The multiplier of eta^2: for 200 words gamma in S, T and their inverses, the ratio
      (c tau + d)^(-1) eta(gamma tau)^2 / eta(tau)^2 equals zeta_12^pitch(gamma), where the pitch is
      counted letter by letter (T -> 1, T^-1 -> 11, S -> 9, S^-1 -> 3).
  (3) The Klein four-group (Z/12)^x = {1, 5, 7, 11} as Gal(Q(zeta_12)/Q), and its three quadratic
      subfields Q(i), Q(sqrt -3), Q(sqrt 3).
  (4) The four Dirichlet characters mod 12, their orthogonality, and L(1, chi) from the digamma
      function against the class number formula: pi/4, pi/(3 sqrt 3), log(2 + sqrt 3)/sqrt 3.
  (5) The diatonic scale: seven consecutive fifths, maximally even, the Christoffel word LLLsLLs.
  (6) Why the ear chooses 12: the continued fraction of log2(3/2) and the error of the best fifth in
      every equal temperament with n <= 60 steps.

What it shows
  The group-theoretic statements (1)-(2) are proved in Lean in the companion repository
  morley-smith-formal (tau-track/TauTrack/Tones); here they are checked numerically. (3)-(6) are classical.

What it does not claim
  That the modular group explains musical tuning. Section (6) gives the acoustic reason for 12, which is
  a different reason from the one in (1)-(2).
"""
import cmath
import itertools
import math
import random
from fractions import Fraction

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

from common import BLUE, GRID, INK, INK_2, ORANGE, check, save

mp.mp.dps = 40
NAMES = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"]

# ---------------------------------------------------------------------------------------------------
print("(1) relations in SL(2, Z)")
S = np.array([[0, -1], [1, 0]], dtype=object)
T = np.array([[1, 1], [0, 1]], dtype=object)
I2 = np.array([[1, 0], [0, 1]], dtype=object)
check((S.dot(S) == -I2).all(), "S^2 = -1")
ST = S.dot(T)
check((ST.dot(ST).dot(ST) == -I2).all(), "(ST)^3 = -1")

# ---------------------------------------------------------------------------------------------------
print("(2) the multiplier of eta^2")


def eta(tau):
    q = mp.exp(2j * mp.pi * tau)
    return mp.exp(2j * mp.pi * tau / 24) * mp.qp(q, q)


def mobius(g, tau):
    return (g[0][0] * tau + g[0][1]) / (g[1][0] * tau + g[1][1])


S_inv = -S
T_inv = np.array([[1, -1], [0, 1]], dtype=object)
letters = {"T": (T, 1), "t": (T_inv, 11), "S": (S, 9), "s": (S_inv, 3)}
zeta12 = mp.exp(2j * mp.pi / 12)
rng = random.Random(20261120)
tau0 = mp.mpc(0.13, 1.07)
worst = 0
tested = 0
while tested < 200:
    word = "".join(rng.choice("TtSs") for _ in range(rng.randint(1, 7)))
    g = I2
    pitch = 0
    for ch in word:
        m, p = letters[ch]
        g = g.dot(m)
        pitch = (pitch + p) % 12
    gt = mobius(g, tau0)
    if mp.im(gt) < 0.15:      # keep the q-product fast; the identity holds everywhere
        continue
    ratio = eta(gt) ** 2 / eta(tau0) ** 2 / (g[1][0] * tau0 + g[1][1])
    worst = max(worst, abs(ratio - zeta12 ** pitch))
    tested += 1
check(worst < mp.mpf(10) ** -30,
      f"eta^2 |_1 gamma = zeta_12^pitch(gamma) eta^2 for 200 random words (max error {mp.nstr(worst, 3)})")
check(abs(eta(tau0 + 1) ** 2 - zeta12 * eta(tau0) ** 2) < 1e-30, "T: pitch 1, a semitone")
check(abs(eta(-1 / tau0) ** 2 / tau0 - (-1j) * eta(tau0) ** 2) < 1e-30, "S: multiplier -i = zeta_12^9")

# ---------------------------------------------------------------------------------------------------
print("(3) (Z/12)^x and the subfields of Q(zeta_12)")
units = [a for a in range(12) if math.gcd(a, 12) == 1]
check(units == [1, 5, 7, 11], "(Z/12)^x = {1, 5, 7, 11}")
check(all(a * a % 12 == 1 for a in units), "every unit squares to 1: the Klein four-group")
z = cmath.exp(2j * math.pi / 12)
check(abs(z ** 3 - 1j) < 1e-12, "i = zeta^3")
check(abs((z ** 4 - z ** 8) - 1j * math.sqrt(3)) < 1e-12, "sqrt(-3) = zeta^4 - zeta^8")
check(abs((z + z ** 11) - math.sqrt(3)) < 1e-12, "sqrt(3) = zeta + zeta^11")
fix = {name: [a for a in units if abs(f(z ** a) - f(z)) < 1e-12]
       for name, f in (("Q(i)", lambda w: w ** 3), ("Q(sqrt -3)", lambda w: w ** 4 - w ** 8),
                       ("Q(sqrt 3)", lambda w: w + w ** 11))}
check(fix == {"Q(i)": [1, 5], "Q(sqrt -3)": [1, 7], "Q(sqrt 3)": [1, 11]},
      f"fixing subgroups {fix}: M5 fixes Q(i), M7 (fifths) fixes Q(sqrt -3), M11 fixes Q(sqrt 3)")

# ---------------------------------------------------------------------------------------------------
print("(4) the four characters mod 12")
chars = {
    "trivial": lambda n: 1 if math.gcd(n, 12) == 1 else 0,
    "chi_-4": lambda n: 0 if n % 2 == 0 else (1 if n % 4 == 1 else -1),
    "chi_-3": lambda n: 0 if n % 3 == 0 else (1 if n % 3 == 1 else -1),
}
chars["chi_12"] = lambda n: chars["chi_-4"](n) * chars["chi_-3"](n)
for name, f in chars.items():
    print(f"     {name:8s} on 1, 5, 7, 11: {[f(a) for a in units]}")
gram = [[sum(f(a) * g(a) for a in units) for g in chars.values()] for f in chars.values()]
check(gram == [[4 if i == j else 0 for j in range(4)] for i in range(4)], "orthogonality: Gram matrix 4 I")


def L1(chi, q):
    """L(1, chi) = -(1/q) sum_a chi(a) psi(a/q) for a non-principal character mod q."""
    return -sum(chi(a) * mp.digamma(mp.mpf(a) / q) for a in range(1, q + 1)) / q


expected = {"chi_-4": (chars["chi_-4"], 4, mp.pi / 4, "pi/4 (Leibniz)"),
            "chi_-3": (chars["chi_-3"], 3, mp.pi / (3 * mp.sqrt(3)), "pi/(3 sqrt 3)"),
            "chi_12": (chars["chi_12"], 12, mp.log(2 + mp.sqrt(3)) / mp.sqrt(3), "log(2 + sqrt 3)/sqrt 3")}
for name, (f, q, val, txt) in expected.items():
    got = L1(f, q)
    check(abs(got - val) < mp.mpf(10) ** -35, f"L(1, {name}) = {mp.nstr(got, 15)} = {txt}")

# ---------------------------------------------------------------------------------------------------
print("(5) the diatonic scale")
diatonic = {0, 2, 4, 5, 7, 9, 11}
check({7 * (k - 1) % 12 for k in range(7)} == diatonic, "seven consecutive fifths F C G D A E B")
check({(12 * k // 7 + 11) % 12 for k in range(7)} == diatonic, "maximally even: floor(12k/7) + 11")
steps = "".join("L" if (b - a) % 12 == 2 else "s" for a, b in zip([0, 2, 4, 5, 7, 9, 11], [2, 4, 5, 7, 9, 11, 12]))
check(steps == "LLsLLLs", "major scale steps LLsLLLs")
christoffel = "".join("s" if (2 * i) // 7 - (2 * (i - 1)) // 7 else "L" for i in range(1, 8))
rotations = {christoffel[k:] + christoffel[:k] for k in range(7)}
check(christoffel == "LLLsLLs" and steps in rotations,
      "the lower Christoffel word of slope 2/5 is LLLsLLs (Lydian); its rotations are the seven modes")

# ---------------------------------------------------------------------------------------------------
print("(6) why the ear chooses 12")
L = math.log2(1.5)
cf, x = [], mp.log(1.5, 2)
for _ in range(9):
    a = int(mp.floor(x))
    cf.append(a)
    x = 1 / (x - a)
conv = []
for k in range(1, len(cf) + 1):
    f = Fraction(0)
    for a in reversed(cf[1:k]):
        f = 1 / (a + f)
    conv.append(cf[0] + f)
print(f"     log2(3/2) = {L:.10f} = [{cf[0]}; {', '.join(map(str, cf[1:]))}, ...]")
print(f"     convergents: {', '.join(str(c) for c in conv)}")
check(Fraction(7, 12) in conv and Fraction(24, 41) in conv and Fraction(31, 53) in conv,
      "7/12, 24/41 and 31/53 are convergents")
ns = np.arange(1, 61)
err = np.array([abs(1200 * (round(n * L) / n - L)) for n in ns])
check(abs(err[11] - 1.955) < 1e-3, f"12-TET fifth is {1200 * 7 / 12:.0f} cents, {err[11]:.3f} cents flat of 3/2")
better = [int(n) for n in ns if err[n - 1] < err[11]]
check(better[0] == 29, f"the first n > 12 with a better fifth is n = {better[0]}")

# ---------------------------------------------------------------------------------------------------
print("figures")

# (a) the pitch clock
fig, ax = plt.subplots(figsize=(4.6, 4.6))
ang = lambda k: math.pi / 2 - 2 * math.pi * k / 12
pts = {k: (math.cos(ang(k)), math.sin(ang(k))) for k in range(12)}
t = np.linspace(0, 2 * math.pi, 400)
ax.plot(np.cos(t), np.sin(t), color=GRID, lw=1, zorder=0)
for ks, col, lab in (([0, 3, 6, 9], BLUE, "powers of S (fixing i): diminished seventh"),
                     ([0, 2, 4, 6, 8, 10], ORANGE, "powers of ST (fixing ρ): whole-tone scale")):
    xs = [pts[k][0] for k in ks + ks[:1]]
    ys = [pts[k][1] for k in ks + ks[:1]]
    ax.plot(xs, ys, color=col, lw=2, label=lab, solid_joinstyle="round")
for k, (x0, y0) in pts.items():
    ax.plot(x0, y0, "o", ms=8, color="white", mec=INK_2, mew=1.2, zorder=3)
    ax.text(1.17 * x0, 1.17 * y0, f"{NAMES[k]}\n{k}", ha="center", va="center", fontsize=9, color=INK)
ax.annotate("", xy=(0.5 * pts[1][0], 0.5 * pts[1][1]), xytext=(0.5 * pts[0][0], 0.5 * pts[0][1]),
            arrowprops=dict(arrowstyle="->", color=INK_2, lw=1, connectionstyle="arc3,rad=-0.3"))
ax.text(0.2, 0.3, "T", color=INK, fontsize=11, style="italic")
ax.annotate("", xy=(0.5 * pts[9][0], 0.5 * pts[9][1]), xytext=(0.5 * pts[0][0], 0.5 * pts[0][1]),
            arrowprops=dict(arrowstyle="->", color=INK_2, lw=1, connectionstyle="arc3,rad=0.3"))
ax.text(-0.3, 0.12, "S", color=INK, fontsize=11, style="italic")
ax.set_xlim(-1.45, 1.45)
ax.set_ylim(-1.75, 1.45)
ax.set_aspect("equal")
ax.axis("off")
ax.legend(loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.02))
ax.set_title("The pitch of an element of SL(2, ℤ)\n(the multiplier of η², in steps of $e^{2\\pi i/12}$)")
save(fig, "tones-clock")

# (b) the fundamental domain and its images
fig, ax = plt.subplots(figsize=(6.4, 3.4))


def apply(g, z):
    return (g[0][0] * z + g[0][1]) / (g[1][0] * z + g[1][1])


rho = cmath.exp(2j * math.pi / 3)
top = 3.0
edge = ([complex(-0.5, y) for y in np.linspace(top, math.sqrt(3) / 2, 80)]
        + [cmath.exp(1j * th) for th in np.linspace(2 * math.pi / 3, math.pi / 3, 120)]
        + [complex(0.5, y) for y in np.linspace(math.sqrt(3) / 2, top, 80)])
mats = {"S": S, "T": T, "t": T_inv}
seen = set()
for length in range(0, 5):
    for w in itertools.product("STt", repeat=length):
        g = I2
        for ch in w:
            g = g.dot(mats[ch])
        key = tuple(int(v) for v in g.flatten())
        neg = tuple(-v for v in key)
        if key in seen or neg in seen:
            continue
        seen.add(key)
        img = [apply(g, z) for z in edge]
        ax.plot([z.real for z in img], [z.imag for z in img], color=GRID, lw=0.8, zorder=1)
fx = [z.real for z in edge]
fy = [z.imag for z in edge]
ax.fill(fx + [0.5, -0.5], fy + [top, top], color="#cde2fb", zorder=2, lw=0)
ax.plot(fx, fy, color=BLUE, lw=2, zorder=3)
ax.plot([0], [1], "o", ms=8, color=BLUE, mec="white", mew=1.5, zorder=4)
ax.plot([rho.real, -rho.conjugate().real], [rho.imag, rho.imag], "o", ms=8, color=ORANGE, mec="white",
        mew=1.5, zorder=4)
ax.annotate("i, fixed by S\n(order 4: four tones)", xy=(0, 1), xytext=(0.95, 1.55), fontsize=9,
            color=INK, arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
ax.annotate("ρ, fixed by ST\n(order 6: six tones)", xy=(rho.real, rho.imag), xytext=(-1.95, 1.55),
            fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
ax.axhline(0, color=INK_2, lw=0.8)
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(0, 2.1)
ax.set_aspect("equal")
ax.set_yticks([0, 1, 2])
ax.set_xticks([-2, -1, -0.5, 0, 0.5, 1, 2])
ax.set_xticklabels(["−2", "−1", "−½", "0", "½", "1", "2"])
ax.set_title("The fundamental domain of SL(2, ℤ) (blue) and some of its images")
save(fig, "tones-domain")

# (c) equal temperaments
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.vlines(ns, 1e-2, err, color="#9ec5f4", lw=2)
ax.plot(ns, err, "o", ms=4, color=BLUE)
for n in (12, 41, 53):
    ax.plot([n], [err[n - 1]], "o", ms=8, color=ORANGE, mec="white", mew=1.5, zorder=4)
    ax.annotate(f"{n}: {err[n - 1]:.2f}¢", xy=(n, err[n - 1]), xytext=(4, 6), textcoords="offset points",
                fontsize=9, color=INK)
ax.set_yscale("log")
ax.set_ylim(1e-2, 1000)
ax.set_xlim(0, 61)
ax.set_xlabel("number of equal steps in the octave, n")
ax.set_ylabel("error of the best fifth (cents)")
ax.grid(axis="y", which="major")
ax.set_title("Why the ear chooses 12: 7/12 is a convergent of log₂(3/2)")
save(fig, "tones-temperaments")
print("done")
