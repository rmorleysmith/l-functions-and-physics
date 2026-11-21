"""Note 09: the zeros as energy levels.

What it computes
  (1) A vectorised Riemann-Siegel evaluation of Hardy's Z(t), with the correction terms C0 and C1, checked
      against mpmath at random points near t = 10^6 (agreement to 10^-7).
  (2) Every zero of zeta with 995290 < t < 1000280: the sign changes of Z on a grid of step 0.02, refined by
      bisection. Their number, 9511, equals the exact count N(1000280) - N(995290) from mpmath, so every zero in
      the window is simple and on the critical line. A sample is checked with mpmath's own Z.
  (3) Unfolding by the mean counting function N(t) = theta(t)/pi + 1, and the statistics of the unfolded zeros:
      the nearest-neighbour spacings against the Wigner surmises of the Gaussian unitary and orthogonal
      ensembles and against a Poisson process (Kolmogorov-Smirnov distances); the pair correlation against
      Montgomery's 1 - (sin pi u / pi u)^2; the number variance against Dyson and Mehta's formula for the
      unitary ensemble and against Berry's semiclassical formula, a sum over the prime powers below E / 2 pi
      plus the random-matrix tail. The zeros follow the ensemble up to L of about 1 and Berry's formula, to
      0.002, up to L = 6.
  (4) The same statistics for the primes between 10^9 and 10^9 + 200000, which behave as a Poisson process.
  (5) The trace formula. The density of zeros smoothed by a Gaussian of width 1 equals
      (1/2 pi)(Re psi(1/4 + iE/2) - log pi) - (1/pi) sum_n Lambda(n) n^(-1/2) exp(-(log n)^2 / 2) cos(E log n),
      the explicit formula with a Gaussian test function. Checked at 400 energies to 10^-6.
  (6) Four packets, four spectra. The zeros below height 600 of the four L-functions of Q(zeta_12) (zeta,
      chi_-4, chi_-3, chi_12), found with the evaluator of lfunctions.py (checked against mpmath). Each spectrum
      repels itself; their union has the spacing law of four independent spectra superposed, and its close
      pairs join zeros of different L-functions.

What it shows
  The zeros of zeta near t = 10^6 have the statistics of the energy levels of a quantum system without
  time-reversal symmetry (the Gaussian unitary ensemble), while the primes are spaced like a Poisson process;
  and the density of zeros is a mean term plus one oscillation for each prime power, as the level density of a
  quantum system is a mean term plus one oscillation for each classical periodic orbit. The four orthogonal
  packets of note 07 have statistically independent spectra.

Assumption
  None for the window computed: the count in (2) shows that all its zeros are on the critical line.
"""
import math

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

import lfunctions as lf
from common import AQUA, BLUE, GRID, INK, INK_2, ORANGE, check, save

mp.mp.dps = 20
TWO_PI = 2 * math.pi
YELLOW = "#eda100"
rng = np.random.default_rng(1973)

# ---------------------------------------------------------------------------------------------------
print("(1) the Riemann-Siegel formula")


def theta(t):
    """The Riemann-Siegel theta function (asymptotic series; error below 1e-12 for t > 1000)."""
    return t / 2 * np.log(t / TWO_PI) - t / 2 - math.pi / 8 + 1 / (48 * t) + 7 / (5760 * t ** 3)


def psi0(p):
    return mp.cos(2 * mp.pi * (p * p - p - mp.mpf(1) / 16)) / mp.cos(2 * mp.pi * p)


P_GRID = (np.arange(1000) + 0.5) / 1000
C1_TABLE = np.array([float(-mp.diff(psi0, mp.mpf(float(x)), 3) / (96 * mp.pi ** 2)) for x in P_GRID])
T1, T2 = 995290.0, 1000280.0
NB = 398                                   # floor(sqrt(t / 2 pi)) throughout the window
check(math.floor(math.sqrt(T1 / TWO_PI)) == NB == math.floor(math.sqrt(T2 / TWO_PI)),
      f"the window {T1:.0f} < t < {T2:.0f} lies in the block floor(sqrt(t/2 pi)) = {NB}")
_n = np.arange(1, NB + 1)
LOGN, ISQ = np.log(_n), 1 / np.sqrt(_n)


def Z(t):
    """Hardy's Z on the window: main sum of 398 terms plus the corrections C0 and C1."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty_like(t)
    for i in range(0, len(t), 4000):
        tt = t[i:i + 4000]
        out[i:i + 4000] = 2 * (np.cos(theta(tt)[:, None] - tt[:, None] * LOGN[None, :]) * ISQ[None, :]).sum(axis=1)
    p = np.sqrt(t / TWO_PI) - NB
    pp = np.where(np.abs(np.cos(TWO_PI * p)) < 1e-6, p + 1e-7, p)
    c0 = np.cos(TWO_PI * (pp * pp - pp - 1 / 16)) / np.cos(TWO_PI * pp)
    c1 = np.interp(p, P_GRID, C1_TABLE)
    return out + (-1) ** (NB - 1) * (TWO_PI / t) ** 0.25 * (c0 + c1 * np.sqrt(TWO_PI / t))


sample_t = np.sort(rng.uniform(T1, T2, 12))
ref = np.array([float(mp.siegelz(mp.mpf(float(t)))) for t in sample_t])
err = float(np.max(np.abs(Z(sample_t) - ref)))
check(err < 3e-9, f"Z by Riemann-Siegel with C0 and C1 agrees with mpmath at 12 random points: max error {err:.1e}")

# ---------------------------------------------------------------------------------------------------
print("(2) every zero in the window")
exact_count = int(mp.nzeros(T2) - mp.nzeros(T1))
grid = np.arange(T1, T2, 0.02)
zg = Z(grid)
idx = np.nonzero(zg[:-1] * zg[1:] < 0)[0]
check(len(idx) == exact_count,
      f"{len(idx)} sign changes of Z on a grid of step 0.02; N({T2:.0f}) - N({T1:.0f}) = {exact_count}: "
      "every zero in the window is simple and on the critical line")
lo, hi, flo = grid[idx].copy(), grid[idx + 1].copy(), zg[idx].copy()
for _ in range(36):
    mid = (lo + hi) / 2
    fm = Z(mid)
    left = flo * fm <= 0
    hi = np.where(left, mid, hi)
    flo = np.where(left, flo, fm)
    lo = np.where(left, lo, mid)
gamma = (lo + hi) / 2
pick = rng.choice(len(gamma), 15, replace=False)
resid = max(abs(float(mp.siegelz(mp.mpf(float(gamma[k]))))) for k in pick)
check(resid < 1e-6, f"mpmath's Z at 15 of the located zeros: largest value {resid:.1e}")
first = float(mp.zetazero(int(mp.nzeros(T1)) + 1).imag)
check(abs(first - gamma[0]) < 1e-6, f"the first zero of the window, {gamma[0]:.6f}, is zero number "
                                    f"{int(mp.nzeros(T1)) + 1} of zeta ({first:.6f} by mpmath)")

# ---------------------------------------------------------------------------------------------------
print("(3) statistics of the unfolded zeros")
x = theta(gamma) / math.pi + 1
s = np.diff(x)
check(abs(s.mean() - 1) < 2e-3, f"unfolded spacings have mean {s.mean():.5f}")

erf = np.vectorize(math.erf)
CDF = {"GUE": lambda v: erf(2 * v / math.sqrt(math.pi)) - 4 * v / math.pi * np.exp(-4 * v * v / math.pi),
       "GOE": lambda v: 1 - np.exp(-math.pi * v * v / 4),
       "Poisson": lambda v: 1 - np.exp(-v)}
PDF = {"GUE": lambda v: 32 / math.pi ** 2 * v * v * np.exp(-4 * v * v / math.pi),
       "GOE": lambda v: math.pi / 2 * v * np.exp(-math.pi * v * v / 4),
       "Poisson": lambda v: np.exp(-v)}


def ks(sample, cdf):
    v = np.sort(sample)
    n = len(v)
    F = cdf(v)
    return float(max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n)))


ks_zero = {k: ks(s, f) for k, f in CDF.items()}
print("     Kolmogorov-Smirnov distance of the zero spacings: "
      + ", ".join(f"{k} {v:.4f}" for k, v in ks_zero.items()))
check(abs(ks_zero["GUE"] - 0.0148) < 5e-4 and abs(ks_zero["GOE"] - 0.0829) < 5e-4 and abs(ks_zero["Poisson"] - 0.2941) < 5e-4,
      "the spacings of the zeros follow the unitary ensemble, not the orthogonal one, and not Poisson")
below_half = float(np.mean(s < 0.5))
check(abs(below_half - float(CDF["GUE"](0.5))) < 0.015,
      f"share of zero spacings below half the mean: {below_half:.3f} (GUE {float(CDF['GUE'](0.5)):.3f}, "
      f"Poisson {float(CDF['Poisson'](0.5)):.3f})")

du = 0.05
edges = np.arange(0, 3 + du / 2, du)
counts = np.zeros(len(edges) - 1)
k = 1
while True:
    d = x[k:] - x[:-k]
    if d.min() > 3:
        break
    counts += np.histogram(d, bins=edges)[0]
    k += 1
u_mid = (edges[:-1] + edges[1:]) / 2
r2 = counts / (len(x) * du)
r2_gue = 1 - (np.sin(math.pi * u_mid) / (math.pi * u_mid)) ** 2
rms_gue = float(np.sqrt(np.mean((r2 - r2_gue) ** 2)))
rms_poi = float(np.sqrt(np.mean((r2 - 1) ** 2)))
check(rms_gue < 0.06 and rms_gue < rms_poi / 4,
      f"pair correlation on 0 < u < 3: rms distance {rms_gue:.3f} from 1 - (sin pi u / pi u)^2, "
      f"{rms_poi:.3f} from the Poisson value 1")


def number_variance(pos, L):
    starts = np.arange(pos[0], pos[-1] - L, 0.05)
    cnt = np.searchsorted(pos, starts + L) - np.searchsorted(pos, starts)
    return float(cnt.var())


def sigma2_gue(L):
    L = mp.mpf(L)
    return float((mp.log(2 * mp.pi * L) + mp.euler + 1 - mp.cos(2 * mp.pi * L) - mp.ci(2 * mp.pi * L)) / mp.pi ** 2
                 + L * (1 - 2 / mp.pi * mp.si(2 * mp.pi * L)))


def prime_powers_below(X):
    """(k, p, p^k) for the prime powers p^k < X."""
    sieve = np.ones(int(X) + 1, dtype=bool)
    sieve[:2] = False
    for q in range(2, math.isqrt(int(X)) + 1):
        if sieve[q]:
            sieve[q * q::q] = False
    ps = np.nonzero(sieve)[0]
    ks, pp, pk = [], [], []
    k = 1
    while 2 ** k < X:
        sel = ps[ps.astype(float) ** k < X]
        ks.append(np.full(len(sel), k))
        pp.append(sel)
        pk.append(sel.astype(float) ** k)
        k += 1
    return np.concatenate(ks), np.concatenate(pp), np.concatenate(pk)


E_MEAN = float(gamma.mean())
LOG_X = math.log(E_MEAN / TWO_PI)
PK_K, PK_P, PK_Q = prime_powers_below(E_MEAN / TWO_PI)


def sigma2_berry(L):
    """Berry's semiclassical number variance: the prime powers below E / 2 pi, plus the random-matrix tail."""
    orbits = float(np.sum(np.sin(math.pi * L * PK_K * np.log(PK_P) / LOG_X) ** 2 / (PK_K ** 2 * PK_Q)))
    a = 2 * math.pi * L
    tail = 0.5 - 0.5 * (math.cos(a) + a * float(mp.si(a)) - a * math.pi / 2)
    return 2 / math.pi ** 2 * (orbits + tail)


LS = np.round(np.arange(0.1, 6.01, 0.1), 2)
nv_zero = np.array([number_variance(x, L) for L in LS])
nv_gue = np.array([sigma2_gue(L) for L in LS])
nv_berry = np.array([sigma2_berry(L) for L in LS])
at = lambda L: int(round(L * 10)) - 1
for L in (0.5, 1.0):
    check(abs(nv_zero[at(L)] - nv_gue[at(L)]) < 0.02,
          f"number variance of the zeros at L = {L}: {nv_zero[at(L)]:.3f} (GUE {nv_gue[at(L)]:.3f}, "
          f"Berry {nv_berry[at(L)]:.3f}, Poisson {L:.1f})")
for L in (2.0, 4.0, 6.0):
    check(nv_zero[at(L)] < nv_gue[at(L)] - 0.02 and abs(nv_zero[at(L)] - nv_berry[at(L)]) < 0.03,
          f"at L = {L} the zeros are more rigid than the ensemble: {nv_zero[at(L)]:.3f} (GUE {nv_gue[at(L)]:.3f}; "
          f"Berry's sum over prime powers {nv_berry[at(L)]:.3f})")
dev_berry = float(np.max(np.abs(nv_zero - nv_berry)))
dev_gue = float(np.max(np.abs(nv_zero - nv_gue)))
check(dev_berry < 0.0025 and abs(dev_gue - 0.087) < 0.001,
      f"over 0.1 <= L <= 6 the zeros stay within {dev_berry:.3f} of Berry's formula and depart by {dev_gue:.3f} from GUE")

# ---------------------------------------------------------------------------------------------------
print("(4) the primes, for comparison")
A, B = 10 ** 9, 10 ** 9 + 200000
small = [q for q in range(2, math.isqrt(B) + 1) if all(q % r for r in range(2, math.isqrt(q) + 1))]
seg = np.ones(B - A + 1, dtype=bool)
for q in small:
    seg[(-A) % q::q] = False
primes = A + np.nonzero(seg)[0]
gaps = np.diff(primes) / np.log(primes[:-1])
y = np.concatenate([[0.0], np.cumsum(gaps)])
check(abs(gaps.mean() - 1) < 0.02, f"{len(primes)} primes in [10^9, 10^9 + 200000]; normalised gaps have mean {gaps.mean():.4f}")
cuts = [0.5, 1.0, 2.0]


def shares(sample=None, cdf=None):
    if sample is not None:
        c = [float(np.mean(sample < t)) for t in cuts]
    else:
        c = [float(cdf(t)) for t in cuts]
    return [c[0], c[1] - c[0], c[2] - c[1], 1 - c[2]]


table = {"zeros": shares(s), "GUE": shares(cdf=CDF["GUE"]), "primes": shares(gaps), "Poisson": shares(cdf=CDF["Poisson"])}
print("     share of spacings in [0, 1/2), [1/2, 1), [1, 2), [2, oo), in units of the mean:")
for name, row in table.items():
    print(f"       {name:8s} " + "  ".join(f"{v:.3f}" for v in row))
check(max(abs(a - b) for a, b in zip(table["zeros"], table["GUE"])) < 0.02,
      "the zeros' shares are within 0.02 of the unitary ensemble's")
check(max(abs(a - b) for a, b in zip(table["primes"], table["Poisson"])) < 0.05
      and table["primes"][0] > 3 * table["zeros"][0] and table["primes"][3] > 4 * table["zeros"][3],
      "the primes' shares are within 0.05 of a Poisson process's: small and large gaps are common, as they are "
      "rare for the zeros")
nv_prime = np.array([number_variance(y, L) for L in LS])
lx = math.log(A)
gm = [L * (1 - math.log(L * lx) / lx) for L in (1.0, 2.0)]          # h log(x/h) / log^2 x with h = L log x
check(all(0.65 * L < nv_prime[at(L)] < L and nv_prime[at(L)] > 2 * nv_zero[at(L)] for L in (1.0, 2.0, 3.0)),
      f"number variance of the primes at L = 1, 2: {nv_prime[at(1.0)]:.3f}, {nv_prime[at(2.0)]:.3f} "
      f"(Poisson 1, 2; Goldston-Montgomery leading term {gm[0]:.3f}, {gm[1]:.3f}; zeros "
      f"{nv_zero[at(1.0)]:.3f}, {nv_zero[at(2.0)]:.3f})")

# ---------------------------------------------------------------------------------------------------
print("(5) the trace formula")
NMAX = 3000
lam = np.zeros(NMAX + 1)
for q in small:
    if q > NMAX:
        break
    qk = q
    while qk <= NMAX:
        lam[qk] = math.log(q)
        qk *= q
nn = np.nonzero(lam)[0]
amp = lam[nn] / np.sqrt(nn) * np.exp(-np.log(nn) ** 2 / 2)
E = np.linspace(T1 + 2000, T1 + 2040, 400)
smooth = np.exp(-(gamma[None, :] - E[:, None]) ** 2 / 2).sum(axis=1) / math.sqrt(TWO_PI)
mean_density = np.array([float((mp.digamma(mp.mpc(0.25, e / 2)).real - mp.log(mp.pi)) / (2 * mp.pi)) for e in E])
orbit_sum = -(np.cos(np.outer(E, np.log(nn))) @ amp) / math.pi
gap = float(np.max(np.abs(smooth - mean_density - orbit_sum)))
check(gap < 3e-10, f"smoothed density of zeros = mean density + sum over prime powers at 400 energies: "
                  f"largest difference {gap:.1e}")
print(f"     mean density {mean_density.mean():.4f} per unit t; the oscillation ranges over "
      f"{orbit_sum.min():+.3f} .. {orbit_sum.max():+.3f}")
check(abs(mean_density.mean() - math.log(E.mean() / TWO_PI) / TWO_PI) < 1e-6, "mean density = log(E / 2 pi) / 2 pi")

# ---------------------------------------------------------------------------------------------------
print("(6) four packets, four spectra")
chi12 = [{1: 1, 5: -1, 7: -1, 11: 1}.get(n, 0) for n in range(12)]
FAMILIES = {"ζ": (1, [1], 0, BLUE), "χ₋₄": (4, [0, 1, 0, -1], 1, ORANGE), "χ₋₃": (3, [0, 1, -1], 1, AQUA),
            "χ₁₂": (12, chi12, 0, YELLOW)}
T6 = 600.0
probe = np.sort(rng.uniform(0.5, T6, 8))
worst = 0.0
for name, (q, chi, a, _) in FAMILIES.items():
    ref6 = np.array([complex(mp.zeta(mp.mpc(0.5, t))) if q == 1 else complex(mp.dirichlet(mp.mpc(0.5, t), chi))
                     for t in probe])
    worst = max(worst, float(np.max(np.abs(lf.dirichlet_L(probe, q, chi) - ref6))))
check(worst < 1e-10, f"the fast evaluator agrees with mpmath on the critical line: max error {worst:.1e}")
spectra, unfolded = {}, {}
for name, (q, chi, a, _) in FAMILIES.items():
    z = lf.find_zeros(lambda t: lf.hardy_Z(t, q, chi, a), 0.02, T6, 0.04)
    expect = float(lf.mean_count(np.array([T6]), q, a)[0])
    check(abs(len(z) - expect) < 1.5, f"{name}: {len(z)} zeros below {T6:.0f} (mean count {expect:.1f})")
    spectra[name] = z
    unfolded[name] = lf.mean_count(z, q, a)
own = {name: (ks(np.diff(xf), CDF["GUE"]), ks(np.diff(xf), CDF["Poisson"])) for name, xf in unfolded.items()}
print("     each spectrum alone, Kolmogorov-Smirnov distance to GUE / Poisson: "
      + ", ".join(f"{k} {a:.3f} / {b:.3f}" for k, (a, b) in own.items()))
small = {name: float(np.mean(np.diff(xf) < 0.5)) for name, xf in unfolded.items()}
check([round(a, 3) for a, _ in own.values()] == [0.058, 0.072, 0.079, 0.084] and all(b > 0.3 for _, b in own.values())
      and [round(v_, 3) for v_ in small.values()] == [0.062, 0.059, 0.050, 0.064],
      "each of the four spectra repels itself: share of spacings below half the mean "
      + ", ".join(f"{k} {v_:.3f}" for k, v_ in small.items()) + " (GUE 0.112, Poisson 0.393)")

labels = np.concatenate([np.full(len(z), i) for i, z in enumerate(spectra.values())])
heights = np.concatenate(list(spectra.values()))
order = np.argsort(heights)
heights, labels = heights[order], labels[order]
total_count = sum(lf.mean_count(heights, q, a) for (q, chi, a, _) in FAMILIES.values())
su = np.diff(total_count)
frac = np.array([len(z) for z in spectra.values()]) / len(heights)


def gap_gue(v_):
    """Probability that an interval of length v (mean spacings) holds no level: Wigner surmise, unitary."""
    return np.exp(-4 * v_ * v_ / math.pi) - v_ * (1 - erf(2 * v_ / math.sqrt(math.pi)))


def cdf_superposed(v_):
    """Spacing law of independent unitary spectra superposed with relative densities frac."""
    v_ = np.atleast_1d(np.asarray(v_, dtype=float))
    E_ = np.array([gap_gue(f * v_) for f in frac])
    tail = np.array([1 - CDF["GUE"](f * v_) for f in frac])
    out = np.ones_like(v_)
    for i in range(len(frac)):
        out -= frac[i] * tail[i] * np.prod(np.delete(E_, i, axis=0), axis=0)
    return out


ks_union = {"four independent spectra": ks(su, cdf_superposed), "one unitary spectrum": ks(su, CDF["GUE"]),
            "Poisson": ks(su, CDF["Poisson"])}
print("     the union of the four spectra, Kolmogorov-Smirnov distance to: "
      + ", ".join(f"{k} {v:.3f}" for k, v in ks_union.items()))
check(ks_union["four independent spectra"] < 0.04 and ks_union["one unitary spectrum"] > 0.15,
      f"the {len(heights)} zeros of the four L-functions together are spaced as four independent spectra")
close = su < 0.25
mixed = labels[:-1][close] != labels[1:][close]
check(mixed.mean() > 0.9, f"{int(close.sum())} spacings of the union are below a quarter of the mean; "
                          f"{int(mixed.sum())} of them join zeros of different L-functions")

# ---------------------------------------------------------------------------------------------------
print("figures")
v = np.linspace(0, 3.5, 400)

# (a) spacings
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
ax = axes[0]
ax.hist(s, bins=np.arange(0, 3.51, 0.1), density=True, color="#9ec5f4", edgecolor="white", linewidth=0.5)
ax.plot(v, PDF["GUE"](v), color=INK, lw=1.6)
ax.plot(v, PDF["GOE"](v), color=INK_2, lw=1.2, ls=(0, (4, 3)))
ax.plot(v, PDF["Poisson"](v), color=INK_2, lw=1.2, ls=(0, (1, 2)))
ax.annotate("unitary (GUE)", xy=(1.25, PDF["GUE"](1.25)), xytext=(1.8, 0.98), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.7))
ax.annotate("orthogonal (GOE)", xy=(1.62, PDF["GOE"](1.62)), xytext=(2.0, 0.7), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.7))
ax.annotate("Poisson", xy=(2.5, PDF["Poisson"](2.5)), xytext=(2.6, 0.4), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.7))
ax.set_xlabel("spacing / mean spacing")
ax.set_ylabel("density")
ax.set_ylim(0, 1.15)
ax.set_title(f"{len(gamma)} zeros of ζ near t = 10⁶: they repel", fontsize=10)
ax = axes[1]
ax.hist(gaps, bins=np.arange(0, 3.51, 0.25), density=True, color="#f6c3ad", edgecolor="white", linewidth=0.5)
ax.plot(v, PDF["Poisson"](v), color=INK_2, lw=1.2, ls=(0, (1, 2)))
ax.annotate("Poisson", xy=(1.6, PDF["Poisson"](1.6)), xytext=(2.0, 0.5), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.7))
ax.set_xlabel("gap / mean gap")
ax.set_title(f"{len(primes)} primes near 10⁹: they do not", fontsize=10)
save(fig, "zeros-spacings")

# (b) pair correlation and number variance
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
ax = axes[0]
uu = np.linspace(0.001, 3, 400)
ax.plot(u_mid, r2, "o", ms=3.5, color=BLUE)
ax.plot(uu, 1 - (np.sin(math.pi * uu) / (math.pi * uu)) ** 2, color=INK, lw=1.4)
ax.axhline(1, color=INK_2, lw=1, ls=(0, (1, 2)))
ax.text(1.5, 0.62, "1 − (sin πu / πu)²", fontsize=8.5, color=INK)
ax.text(0.08, 1.05, "Poisson", fontsize=8.5, color=INK)
ax.set_xlabel("distance u between two zeros (mean spacings)")
ax.set_ylabel("pair correlation")
ax.set_ylim(0, 1.3)
ax.set_title("Pair correlation of the zeros", fontsize=10)
ax = axes[1]
ax.plot(LS, nv_gue, color=INK_2, lw=1.2, ls=(0, (4, 3)))
ax.plot(LS, nv_berry, color=ORANGE, lw=1.6)
ax.plot(LS, nv_zero, "o", ms=3.2, color=BLUE)
ax.text(4.4, 0.535, "random matrices (GUE)", fontsize=8.5, color=INK, ha="center")
ax.annotate("sum over prime powers (Berry)", xy=(3.2, nv_berry[at(3.2)]), xytext=(1.5, 0.2), fontsize=8.5,
            color=INK, arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.7))
ax.annotate("zeros", xy=(1.0, nv_zero[at(1.0)]), xytext=(0.3, 0.47), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.7))
ax.set_xlabel("window length L (mean spacings)")
ax.set_ylabel("variance of the number of zeros")
ax.set_ylim(0, 0.6)
ax.set_xlim(0, 6.1)
ax.set_title("Beyond L ≈ 1 the primes set the rigidity", fontsize=10)
save(fig, "zeros-pair-variance")

# (c) the trace formula
fig, ax = plt.subplots(figsize=(7.4, 3.0))
ax.axhline(0, color=GRID, lw=0.8)
ax.plot(E - E[0], smooth - mean_density, color=BLUE, lw=3, alpha=0.45, label="zeros: smoothed density − mean density")
ax.plot(E - E[0], orbit_sum, color=ORANGE, lw=1.1, label="primes: −(1/π) Σ Λ(n) n^(−1/2) e^(−(log n)²/2) cos(E log n)")
ax.set_xlabel(f"E − {E[0]:.0f}")
ax.set_ylabel("fluctuation of the density")
ax.set_ylim(-0.75, 0.95)
ax.legend(loc="upper center", ncol=1, fontsize=8.5)
ax.set_title("The density of zeros oscillates with the periods of the primes", fontsize=10)
save(fig, "zeros-trace")

# (d) four packets, four spectra
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.28})
ax = axes[0]
lo_t, hi_t = 100.0, 124.0
for row, (name, (q, chi, a, col)) in enumerate(FAMILIES.items()):
    z = spectra[name]
    z = z[(z > lo_t) & (z < hi_t)]
    ax.vlines(z, 3.6 - row, 4.4 - row, color=col, lw=1.4)
    ax.text(lo_t - 0.5, 4 - row, name, ha="right", va="center", fontsize=9, color=INK)
zu = heights[(heights > lo_t) & (heights < hi_t)]
lu = labels[(heights > lo_t) & (heights < hi_t)]
cols = [c for (_, _, _, c) in FAMILIES.values()]
for zz, ll in zip(zu, lu):
    ax.vlines(zz, -0.9, -0.1, color=cols[ll], lw=1.4)
ax.text(lo_t - 0.5, -0.5, "all four", ha="right", va="center", fontsize=9, color=INK)
ax.set_xlim(lo_t - 4.2, hi_t)
ax.set_ylim(-1.2, 4.7)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xticks([100, 106, 112, 118, 124])
ax.set_xlabel("height t of the zero")
ax.set_title("Each spectrum keeps its distances;\ntogether they fall as they may", fontsize=10)
ax = axes[1]
ax.hist(su, bins=np.arange(0, 3.51, 0.125), density=True, color="#cde2fb", edgecolor="white", linewidth=0.5)
dv = v[1] - v[0]
ax.plot(v[:-1] + dv / 2, np.diff(cdf_superposed(v)) / dv, color=INK, lw=1.6)
ax.plot(v, PDF["GUE"](v), color=INK_2, lw=1.2, ls=(0, (4, 3)))
ax.plot(v, PDF["Poisson"](v), color=INK_2, lw=1.2, ls=(0, (1, 2)))
ax.annotate("four independent\nspectra", xy=(0.55, float(np.diff(cdf_superposed(np.array([0.5, 0.6])))[0] / 0.1)),
            xytext=(1.2, 0.82), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.7))
ax.annotate("one spectrum", xy=(1.45, PDF["GUE"](1.45)), xytext=(1.9, 0.58), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.7))
ax.annotate("Poisson", xy=(0.08, PDF["Poisson"](0.08)), xytext=(0.45, 1.02), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.7))
ax.set_xlabel("spacing / mean spacing")
ax.set_ylabel("density")
ax.set_ylim(0, 1.15)
ax.set_title(f"Spacings of all {len(heights)} zeros", fontsize=10)
save(fig, "zeros-four-spectra")
print("done")
