"""Note 04: what an L-function is, and what it looks like.

What it computes
  (1) The prime tones. For Re s = sigma > 1, log zeta(s) = sum_p sum_k p^(-ks)/k, so log|zeta(sigma + it)| is a
      superposition of waves cos(k t log p), one family for each prime, with amplitudes p^(-k sigma)/k.
      Summing the primes below 10^4 reproduces log|zeta(1.3 + it)| on 0 <= t <= 40 to better than 2%.
  (2) The critical-line curves. t -> L(1/2 + it, chi) for zeta and the three non-trivial primitive characters
      inside Q(zeta_12): chi_-4, chi_-3 (odd: a = 1) and chi_12 (even: a = 0).
  (3) Hardy's Z-functions. Z(t) = e^(i theta(t)) L(1/2 + it, chi), with theta the argument of the gamma factor
      (q/pi)^((s+a)/2) Gamma((s+a)/2). The root number is 1 for all four, so Z is real; its sign changes are
      the zeros on the critical line. The first zeros are located and checked, and their number below t = 30
      is compared with the Riemann-von Mangoldt main term (T/2 pi) log(qT/(2 pi e)): more zeros for larger q.

What it shows
  An L-function as a sum of waves whose frequencies are the logarithms of the primes, and the four L-functions
  of Q(zeta_12) as a bundle of curves winding as t increases, meeting the axis at their zeros.
"""
import math

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from common import AQUA, BLUE, BLUE_RAMP, GRID, INK, INK_2, ORANGE, check, save

mp.mp.dps = 20
YELLOW = "#eda100"


def primes(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.nonzero(sieve)[0]


# characters as lists of values mod q; (name, q, values, a = parity)
def chi_list(q, f):
    return [f(n) for n in range(q)]


CHARS = [
    ("ζ", 1, [1], 0, BLUE),
    ("χ₋₄", 4, chi_list(4, lambda n: [0, 1, 0, -1][n]), 1, ORANGE),
    ("χ₋₃", 3, chi_list(3, lambda n: [0, 1, -1][n]), 1, AQUA),
    ("χ₁₂", 12, chi_list(12, lambda n: {1: 1, 5: -1, 7: -1, 11: 1}.get(n, 0)), 0, YELLOW),
]


def L(s, q, vals):
    return mp.zeta(s) if q == 1 else mp.dirichlet(s, vals)


def hardy_Z(t, q, vals, a):
    s = mp.mpc(0.5, t)
    theta = mp.im((s + a) / 2 * mp.log(mp.mpf(q) / mp.pi) + mp.loggamma((s + a) / 2))
    return L(s, q, vals) * mp.exp(1j * theta)


# ---------------------------------------------------------------------------------------------------
print("(1) the prime tones")
sigma = 1.3
ts = np.linspace(0, 40, 801)
P = primes(10000)
tones = np.zeros_like(ts)
for p in P:
    for k in range(1, 12):
        amp = p ** (-k * sigma) / k
        if amp < 1e-12:
            break
        tones += amp * np.cos(k * ts * math.log(p))
exact = np.array([float(mp.log(abs(mp.zeta(mp.mpc(sigma, t))))) for t in ts])
err = np.max(np.abs(tones - exact)) / np.max(np.abs(exact))
check(err < 0.02, f"log|zeta(1.3 + it)| = sum over primes p < 10^4 of the tones, max relative error {err:.4f}")

# ---------------------------------------------------------------------------------------------------
print("(2) and (3) the critical line")
tgrid = np.linspace(0.05, 30, 1200)
curves, Zs, zeros = {}, {}, {}
for name, q, vals, a, col in CHARS:
    vals_c = [L(mp.mpc(0.5, t), q, vals) for t in tgrid]
    curves[name] = np.array([complex(v) for v in vals_c])
    zt = [hardy_Z(t, q, vals, a) for t in tgrid]
    check(max(abs(mp.im(z)) for z in zt) < 1e-10, f"{name}: Z(t) is real (root number 1, parity a = {a})")
    Zr = np.array([float(mp.re(z)) for z in zt])
    Zs[name] = Zr
    found = []
    for i in range(len(tgrid) - 1):
        if Zr[i] == 0 or Zr[i] * Zr[i + 1] < 0:
            r = mp.findroot(lambda t: mp.re(hardy_Z(t, q, vals, a)), (tgrid[i], tgrid[i + 1]), solver="anderson")
            found.append(float(r))
    zeros[name] = found
    print(f"     {name}: zeros in 0 < t < 30: {', '.join(f'{z:.6f}' for z in found)}")
    check(all(abs(L(mp.mpc(0.5, z), q, vals)) < 1e-12 for z in found), f"{name}: |L(1/2 + i t_n)| < 1e-12 at each")
check(abs(zeros["ζ"][0] - 14.134725) < 1e-6 and abs(zeros["ζ"][1] - 21.022040) < 1e-6
      and len(zeros["ζ"]) == 3, "zeta: 14.134725, 21.022040, 25.010858 (the classical first zeros)")
check(abs(zeros["χ₋₄"][0] - 6.020949) < 1e-5, "L(s, chi_-4): first zero 6.020949")
for name, q, vals, a, col in CHARS:
    T = 30
    main = T / (2 * math.pi) * math.log(q * T / (2 * math.pi * math.e)) + (7 / 8 if q == 1 else 0)
    check(abs(len(zeros[name]) - main) < 1.5,
          f"{name}: {len(zeros[name])} zeros below t = 30; Riemann-von Mangoldt main term {main:.2f}")

# ---------------------------------------------------------------------------------------------------
print("figures")
cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP[2:])

# (a) the prime tones
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]})
ax = axes[0]
for p, col in zip((2, 3, 5, 7), (BLUE, ORANGE, AQUA, YELLOW)):
    ax.plot(ts, p ** (-sigma) * np.cos(ts * math.log(p)), color=col, lw=1.5, label=f"p = {p}")
ax.set_ylabel("tone of p")
ax.legend(loc="upper right", ncol=4)
ax.set_ylim(-0.55, 0.75)
ax.set_title("Each prime is a pure tone in t, of frequency log p and amplitude $p^{-\\sigma}$", fontsize=10)
ax = axes[1]
ax.plot(ts, exact, color=INK_2, lw=3, alpha=0.35, label="log|ζ(1.3 + it)|")
ax.plot(ts, tones, color=BLUE, lw=1.2, label="sum of the tones of all primes below 10 000")
ax.set_xlabel("t")
ax.set_ylabel("log |ζ|")
ax.legend(loc="upper right", fontsize=8.5)
ax.set_title("Their sum is the zeta function", fontsize=10)
save(fig, "lfun-tones")

# (b) the curve of zeta on the critical line
fig, ax = plt.subplots(figsize=(6.0, 4.6))
z = curves["ζ"]
pts = np.column_stack([z.real, z.imag]).reshape(-1, 1, 2)
seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(seg, cmap=cmap, linewidths=1.6)
lc.set_array(tgrid[:-1])
ax.add_collection(lc)
ax.plot([0], [0], "o", ms=8, color=ORANGE, mec="white", mew=1.5, zorder=4)
ax.text(-1.5, 1.85, "passes through 0 at each zero:\nt = " + ", ".join(f"{tz:.3f}" for tz in zeros["ζ"]),
        fontsize=8.5, color=INK, va="top")
ax.axhline(0, color=GRID, lw=0.8, zorder=0)
ax.axvline(0, color=GRID, lw=0.8, zorder=0)
ax.set_xlim(-1.6, 3.2)
ax.set_ylim(-2.0, 2.0)
ax.set_aspect("equal")
ax.set_xlabel("Re ζ(½ + it)")
ax.set_ylabel("Im ζ(½ + it)")
cb = fig.colorbar(lc, ax=ax, shrink=0.75, pad=0.02)
cb.set_label("t")
cb.outline.set_visible(False)
ax.set_title("ζ(½ + it) for 0 < t < 30: a curve that winds through the origin", fontsize=10)
save(fig, "lfun-critical-curve")

# (c) the bundle
fig = plt.figure(figsize=(7.4, 4.8))
ax = fig.add_subplot(111, projection="3d")
for name, q, vals, a, col in CHARS:
    w = curves[name]
    ax.plot(tgrid, w.real, w.imag, color=col, lw=1.4, label=f"{name} ({'odd' if a else 'even'})")
ax.plot(tgrid, 0 * tgrid, 0 * tgrid, color=INK_2, lw=0.8, ls="--")
ax.set_xlabel("t")
ax.set_ylabel("Re L(½ + it)")
ax.set_zlabel("Im L(½ + it)")
ax.set_ylim(-1.5, 3)
ax.set_zlim(-2, 2)
ax.view_init(elev=18, azim=-62)
ax.xaxis.pane.set_visible(False)
ax.yaxis.pane.set_visible(False)
ax.zaxis.pane.set_visible(False)
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("The four L-functions of ℚ(ζ₁₂) on the critical line: a bundle winding in t\n"
             "(each curve meets the dashed axis at its zeros)", fontsize=10)
save(fig, "lfun-bundle")

# (d) Hardy's Z-functions
fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.8), sharex=True)
for ax, (name, q, vals, a, col) in zip(axes, CHARS):
    ax.axhline(0, color=GRID, lw=0.8)
    ax.plot(tgrid, Zs[name], color=col, lw=1.4)
    ax.plot(zeros[name], [0] * len(zeros[name]), "o", ms=5, color=INK, zorder=3)
    ax.set_title(f"{name},  q = {q},  {'odd' if a else 'even'}", loc="left", fontsize=9, pad=2)
    ax.set_ylabel("Z(t)")
axes[-1].set_xlabel("t")
fig.suptitle("Hardy's Z-functions: real on the critical line; their zeros are the zeros of L", fontsize=10)
save(fig, "lfun-hardy")
print("done")
