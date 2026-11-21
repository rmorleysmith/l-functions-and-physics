"""Note 07: impulses, packets and spectra. Primes as unit impulses, characters as orthogonal packets, zeros as
the continuous spectrum.

What it computes
  D1 (impulses and waves). Poisson summation. A comb of unit impulses diffused for a time t equals a sum of
     waves, the k-th damped by exp(-4 pi^2 k^2 t); at x = 0 for the Gaussian this is theta(1/t) = sqrt(t) theta(t),
     where theta(t) = sum_n exp(-pi n^2 t).
  D2 (orthogonal packets). The character table modulo 12 is a 4 x 4 Hadamard matrix (H H^T = 4 I). So the
     prime-power impulse train splits into four residue classes, and back, without loss:
     psi(x; 12, a) = (1/4) sum_chi chi(a) psi(x, chi), and the "energy" is preserved (Parseval):
     sum_a psi(x; 12, a)^2 = (1/4) sum_chi psi(x, chi)^2.
  D3 (the L-function as a transform of the impulses). -L'/L(s, chi) = sum chi(n) Lambda(n) n^(-s): the
     Laplace transform, in u = log x, of the impulse train with an impulse log p at each u = log p^k.
     Checked at s = 2 for the four characters.
  D4 (the explicit formula). Each packet is rebuilt from the zeros of its L-function:
     psi(x, chi) = -sum_rho x^rho / rho + (terms from the trivial zeros and a constant), and
     psi(x) = x - sum_rho x^rho / rho - log 2 pi - log(1 - x^-2)/2. With the zeros up to height T the four
     staircases psi(x; 12, a) are reproduced, and the error falls as T grows.
  D5 (the spectrum of the zeros). Landau: sum_{0 < gamma <= T} x^(i gamma) = -(T/2 pi) Lambda(x)/sqrt(x)
     + O(log T). As a function of u = log x, the zeros' "diffraction pattern" has lines exactly at the logs of
     the prime powers, with depths Lambda(n)/sqrt(n). Checked with the first 600 zeros of zeta.
  Chebyshev's bias. Counting primes (not prime powers), the class 1 mod 12, which receives every prime square,
     lags the classes 5, 7, 11 (Rubinstein and Sarnak). Checked at x = 10^6.

What it shows
  Unit impulses (the primes), orthogonal packets (the four characters mod 12) and a continuous spectrum
  (the zeros) are three views of one object, with exact, reusable transforms between them.

Assumption
  The reconstructions use the zeros on the critical line below height 150; under the generalised Riemann
  hypothesis these are all the non-trivial zeros below that height.
"""
import math

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

import lfunctions as lf
from common import AQUA, BLUE, GRID, INK, INK_2, ORANGE, check, save

mp.mp.dps = 15
YELLOW = "#eda100"
CLASS_COL = {1: BLUE, 5: ORANGE, 7: AQUA, 11: YELLOW}

chi4 = lambda n: 0 if n % 2 == 0 else (1 if n % 4 == 1 else -1)
chi3 = lambda n: 0 if n % 3 == 0 else (1 if n % 3 == 1 else -1)
PRIM = {"χ₋₄": chi4, "χ₋₃": chi3, "χ₁₂": lambda n: chi4(n) * chi3(n)}       # primitive characters
cop = lambda n: 1 if math.gcd(n, 12) == 1 else 0
CHARS = {"χ₀": cop, "χ₋₄": lambda n: cop(n) * chi4(n), "χ₋₃": lambda n: cop(n) * chi3(n),
         "χ₁₂": lambda n: chi4(n) * chi3(n)}                                    # the characters mod 12


def sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if s[p]:
            s[p * p::p] = False
    return np.nonzero(s)[0]


# ---------------------------------------------------------------------------------------------------
print("D1 impulses and waves")
theta = lambda t: mp.nsum(lambda n: mp.exp(-mp.pi * n * n * t), [-mp.inf, mp.inf])
for t in (0.37, 1.9):
    check(abs(theta(1 / t) - mp.sqrt(t) * theta(t)) < 1e-12, f"theta(1/t) = sqrt(t) theta(t) at t = {t}")
for x0, t0 in ((0.3, 0.01), (0.5, 0.05)):
    comb = mp.nsum(lambda n: mp.exp(-(x0 - n) ** 2 / (4 * t0)) / mp.sqrt(4 * mp.pi * t0), [-mp.inf, mp.inf])
    waves = mp.nsum(lambda k: mp.exp(-4 * mp.pi ** 2 * k * k * t0) * mp.cos(2 * mp.pi * k * x0), [-mp.inf, mp.inf])
    check(abs(comb - waves) < 1e-12,
          f"diffused comb = damped waves at x = {x0}, t = {t0}: {mp.nstr(comb, 10)}")

# ---------------------------------------------------------------------------------------------------
print("D2 orthogonal packets")
units = [1, 5, 7, 11]
H = np.array([[CHARS[c](a) for a in units] for c in CHARS])
check((H @ H.T == 4 * np.eye(4)).all(), "the character table mod 12 is a Hadamard matrix: H H^T = 4 I")
N = 10 ** 6
P = sieve(N)
Lam = np.zeros(N + 1)
for p in P:
    pk = p
    while pk <= N:
        Lam[pk] = math.log(p)
        pk *= p
n_all = np.arange(N + 1)
cls = {a: Lam * (n_all % 12 == a) for a in units}
psi_cls = {a: np.cumsum(v) for a, v in cls.items()}
psi_chi = {c: np.cumsum(Lam * np.array([f(k) for k in range(12)])[n_all % 12]) for c, f in CHARS.items()}
X = [100, 10 ** 4, 10 ** 6]
ok = all(abs(psi_cls[a][x] - sum(CHARS[c](a) * psi_chi[c][x] for c in CHARS) / 4) < 1e-6 * x
         for a in units for x in X)
check(ok, "psi(x; 12, a) = (1/4) sum_chi chi(a) psi(x, chi) at x = 10^2, 10^4, 10^6")
for x in X:
    e1 = sum(psi_cls[a][x] ** 2 for a in units)
    e2 = sum(psi_chi[c][x] ** 2 for c in CHARS) / 4
    check(abs(e1 - e2) < 1e-9 * e1, f"Parseval at x = {x}: sum_a psi(x;12,a)^2 = (1/4) sum_chi psi(x,chi)^2 = {e1:.6g}")

# ---------------------------------------------------------------------------------------------------
print("D3 the L-function as a transform of the impulses")
prim = {"χ₋₄": [0, 1, 0, -1], "χ₋₃": [0, 1, -1], "χ₁₂": [PRIM["χ₁₂"](k) for k in range(12)]}
for c, v in prim.items():
    direct = float(np.sum(Lam[1:] * np.array(v)[n_all[1:] % len(v)] / n_all[1:] ** 2.0))
    Lf = lambda s: mp.dirichlet(s, v)
    exact = -mp.diff(Lf, 2) / Lf(2)
    check(abs(direct - float(exact)) < 5e-5, f"{c}: sum chi(n) Lambda(n) n^-2 (n <= 10^6) = {direct:.6f}, -L'/L(2) = {mp.nstr(exact, 8)}")

# ---------------------------------------------------------------------------------------------------
print("D4 the explicit formula: packets rebuilt from zeros")


def zeros_of(vals, a, q, T, step=0.02):
    """Zeros of L(s, chi) on the critical line below height T: sign changes of Hardy's Z (lfunctions.py)."""
    found = lf.find_zeros(lambda t: lf.hardy_Z(t, q, vals, a), 0.05, T, step)
    for z in found[:3]:                                    # the fast evaluator against mpmath
        assert abs(mp.dirichlet(mp.mpc(0.5, float(z)), vals)) < 1e-9
    return [float(z) for z in found]


TMAX = 150
info = {"χ₋₄": (prim["χ₋₄"], 1, 4), "χ₋₃": (prim["χ₋₃"], 1, 3), "χ₁₂": (prim["χ₁₂"], 0, 12)}
zeros = {}
for c, (v, a, q) in info.items():
    zeros[c] = zeros_of(v, a, q, TMAX)
    main = TMAX / (2 * math.pi) * math.log(q * TMAX / (2 * math.pi * math.e))
    check(abs(len(zeros[c]) - main) < 2.5, f"{c}: {len(zeros[c])} zeros below {TMAX} (main term {main:.1f})")
zz = []
k = 1
while True:
    g = float(mp.zetazero(k).imag)
    if g > TMAX:
        break
    zz.append(g)
    k += 1
zeros["ζ"] = zz
print(f"     zeta: {len(zz)} zeros below {TMAX} (main term {TMAX / (2 * math.pi) * math.log(TMAX / (2 * math.pi * math.e)) + 7 / 8:.1f})")
check(abs(len(zz) - (TMAX / (2 * math.pi) * math.log(TMAX / (2 * math.pi * math.e)) + 7 / 8)) < 2.5, "zeta: zero count matches Riemann-von Mangoldt")
bconst = {}
for c, (v, a, q) in info.items():
    Lf = lambda s, v=v: mp.dirichlet(s, v)
    bconst[c] = float(mp.diff(Lf, 0) / Lf(0)) if a == 1 else float(mp.diff(Lf, 0, 2) / (2 * mp.diff(Lf, 0)))


def osc(x, gammas, T):
    """-sum over zeros 1/2 + i gamma, gamma in (-T, T), of x^rho / rho."""
    g = np.array([y for y in gammas if y <= T])
    rho = 0.5 + 1j * g
    return -2 * np.real(np.exp(np.outer(np.log(x), rho)) / rho).sum(axis=1)


def psi_formula(x, c, T):
    if c == "ζ":
        return x + osc(x, zeros["ζ"], T) - math.log(2 * math.pi) - 0.5 * np.log(1 - x ** -2.0)
    v, a, q = info[c]
    smooth = (-np.log(x) - 0.5 * np.log(1 - x ** -2.0)) if a == 0 else 0.5 * np.log((x + 1) / (x - 1))
    return osc(x, zeros[c], T) - bconst[c] + smooth


def prime_power_sum(x, p, sign):
    """sum over p^k <= x of sign^k log p (the impulses at the powers of p, weighted by chi(p)^k)."""
    K = np.floor(np.log(x) / math.log(p) + 1e-12).astype(int)
    if sign == 1:
        return math.log(p) * K
    return math.log(p) * np.where(K % 2 == 0, 0.0, -1.0)


def psi_class_formula(x, a, T):
    """psi(x; 12, a) from the zeros: the characters mod 12 are the primitive ones with the impulses at the
    powers of 2 and 3 removed."""
    psi0 = psi_formula(x, "ζ", T) - prime_power_sum(x, 2, 1) - prime_power_sum(x, 3, 1)
    psi4 = psi_formula(x, "χ₋₄", T) - prime_power_sum(x, 3, chi4(3))
    psi3 = psi_formula(x, "χ₋₃", T) - prime_power_sum(x, 2, chi3(2))
    psi12 = psi_formula(x, "χ₁₂", T)
    return (psi0 + chi4(a) * psi4 + chi3(a) * psi3 + chi4(a) * chi3(a) * psi12) / 4


xs_mid = np.array([n + 0.5 for n in range(3, 150)])
errs = {}
for T in (50, 100, TMAX):
    errs[T] = max(float(np.sqrt(np.mean((psi_class_formula(xs_mid, a, T) - psi_cls[a][xs_mid.astype(int)]) ** 2)))
                  for a in units)
print("     root-mean-square error of the four rebuilt staircases on 3 < x < 150: "
      + ", ".join(f"T = {T}: {e:.3f}" for T, e in errs.items()))
check([round(errs[T], 2) for T in (50, 100, TMAX)] == [0.83, 0.53, 0.42],
      "the error falls as more zeros are used: 0.83, 0.53, 0.42 at T = 50, 100, 150")

# ---------------------------------------------------------------------------------------------------
print("D5 the spectrum of the zeros (Landau's formula)")
NZ = 600
gam = lf.find_zeros(lambda t: lf.hardy_Z(t, 1, [1], 0), 0.05, 945.0, 0.02)[:NZ]
check(len(gam) == NZ and all(abs(gam[k - 1] - float(mp.zetazero(k).imag)) < 1e-8 for k in (1, 300, 600)),
      "the first 600 zeros of zeta from the fast evaluator; numbers 1, 300 and 600 agree with mpmath")
T = gam[-1]
print(f"     the 600th zero is at height {T:.2f}")
for n_, lab in ((2, "2"), (3, "3"), (4, "4"), (5, "5"), (7, "7")):
    val = np.cos(gam * math.log(n_)).sum()
    pred = -T / (2 * math.pi) * math.log(2 if n_ in (2, 4) else n_) / math.sqrt(n_)
    check(abs(val - pred) < 0.03 * abs(pred), f"x = {lab}: sum cos(gamma log x) = {val:.1f}, Landau -(T/2pi) Lambda(x)/sqrt x = {pred:.1f}")
u = np.linspace(0.3, 3.3, 3001)
sigma = 150.0
w = np.exp(-(gam / sigma) ** 2 / 2)
spec = np.cos(np.outer(u, gam)) @ w
mins = [u[i] for i in range(1, len(u) - 1) if spec[i] < spec[i - 1] and spec[i] < spec[i + 1] and spec[i] < -8]
pp = [n for n in range(2, 28) if Lam[n] > 0]
near = all(min(abs(m - math.log(n)) for n in pp) < 0.012 for m in mins)
check(near and len(mins) >= 9, f"the {len(mins)} deep dips in 0.3 < u < 3.3 all sit at log of prime powers "
                               f"({', '.join(str(round(math.exp(m))) for m in mins)})")

# ---------------------------------------------------------------------------------------------------
print("Chebyshev's bias")
pi_cls = {a: int(np.sum(P % 12 == a)) for a in units}
print(f"     pi(10^6; 12, a) for a = 1, 5, 7, 11: {[pi_cls[a] for a in units]}")
check(pi_cls[1] == min(pi_cls.values()), "the class 1 (which receives every prime square) has the fewest primes")
print("     psi(10^6; 12, a) - 10^6/4: " + ", ".join(f"{a}: {psi_cls[a][N] - N / 4:+.0f}" for a in units))

# ---------------------------------------------------------------------------------------------------
print("figures")

# (a) impulses and packets
nmax = 100
fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
ax = axes[0]
for a in units:
    idx = [n for n in range(2, nmax + 1) if Lam[n] > 0 and n % 12 == a]
    ax.vlines(idx, 0, Lam[idx], color=CLASS_COL[a], lw=2, label=f"{a} mod 12")
other = [n for n in range(2, nmax + 1) if Lam[n] > 0 and math.gcd(n, 12) > 1]
ax.vlines(other, 0, Lam[other], color=GRID, lw=2, label="2, 3 and their powers")
ax.set_ylabel("impulse log p")
ax.legend(loc="upper center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, 1.32))
ax.set_title("Unit impulses: one at each prime power, of size log p, sorted into four residue classes", fontsize=10, pad=26)
ax = axes[1]
offs = {"χ₀": 0, "χ₋₄": 1, "χ₋₃": 2, "χ₁₂": 3}
for c, f in CHARS.items():
    ns = np.array([n for n in range(2, nmax + 1) if Lam[n] > 0 and f(n) != 0])  # the characters mod 12
    vals = np.array([f(n) * Lam[n] for n in ns]) / 5.5
    base = -2.2 * offs[c]
    ax.axhline(base, color=GRID, lw=0.8)
    ax.vlines(ns, base, base + vals, color=INK_2, lw=1.6)
    ax.text(101, base, c, va="center", fontsize=9, color=INK)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("n")
ax.set_title("Four orthogonal packets: the same impulses weighted by the characters χ(n) = ±1", fontsize=10)
save(fig, "imp-packets")

# (b) packets rebuilt from zeros
xs = np.linspace(2.0, 150, 3000)
fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.8), sharex=True, sharey=True)
for ax, a in zip(axes.flat, units):
    ax.step(np.arange(2, 151), psi_cls[a][2:151], where="post", color=INK_2, lw=1.0, alpha=0.6, label="exact")
    ax.plot(xs, psi_class_formula(xs, a, TMAX), color=CLASS_COL[a], lw=1.3, label="from the zeros")
    ax.set_title(f"primes ≡ {a} mod 12", fontsize=9, loc="left")
for ax in axes[1]:
    ax.set_xlabel("x")
for ax in axes[:, 0]:
    ax.set_ylabel("ψ(x; 12, a)")
axes[0, 0].legend(loc="upper left", fontsize=8)
fig.suptitle(f"Each packet rebuilt from the zeros of the four L-functions below height {TMAX}", fontsize=10)
save(fig, "imp-explicit")

# (c) the spectrum of the zeros
fig, ax = plt.subplots(figsize=(7.4, 3.2))
ax.plot(u, spec, color=BLUE, lw=1.1)
scale = spec[np.argmin(np.abs(u - math.log(2)))] / (math.log(2) / math.sqrt(2))
for n in pp:
    if math.log(n) <= 3.3:
        ax.vlines(math.log(n), 0, scale * Lam[n] / math.sqrt(n), color=ORANGE, lw=2.2, alpha=0.8)
        ax.text(math.log(n), 4.6 if n in (16, 25) else 3.0, str(n), ha="center", va="bottom", fontsize=7.5,
                color=INK)
ax.axhline(0, color=GRID, lw=0.8)
ax.set_ylim(-23, 6.5)
ax.text(0.3, 3.0, "n =", fontsize=7.5, color=INK_2, va="bottom")
ax.set_xlabel("u = log x")
ax.set_ylabel("Σ w(γ) cos(γu)")
ax.set_title("The zeros of ζ, read as frequencies, ring exactly at the logarithms of the prime powers\n"
             "(blue: first 600 zeros; orange: impulses −Λ(n)/√n, scaled)", fontsize=10)
save(fig, "imp-landau")

# (d) Chebyshev's bias
xb = np.unique(np.logspace(2, 6, 400).astype(int))
pi_cum = {a: np.cumsum(np.isin(n_all, P) & (n_all % 12 == a)) for a in units}
tot = sum(pi_cum[a] for a in units)
fig, ax = plt.subplots(figsize=(7.4, 3.0))
for a in units:
    ax.plot(xb, pi_cum[a][xb] - tot[xb] / 4, color=CLASS_COL[a], lw=1.4, label=f"{a} mod 12")
ax.axhline(0, color=GRID, lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("x")
ax.set_ylabel("π(x; 12, a) − π(x)/4")
ax.legend(loc="lower left", ncol=4, fontsize=8.5)
ax.set_title("Chebyshev's bias: the class of the squares (1 mod 12) runs behind", fontsize=10)
save(fig, "imp-bias")
print("done")
