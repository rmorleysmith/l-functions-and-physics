"""Note 02, part 2: Gauss sums in diffraction (the Talbot effect) and in acoustics (Schroeder's diffuser).

What it computes
  (1) The quadratic Gauss sum G(q) = sum_{n mod q} exp(2 pi i n^2 / q) is (1 + i) sqrt q, sqrt q, 0, i sqrt q
      for q = 0, 1, 2, 3 mod 4 (Gauss), checked for q <= 400.
  (2) For an odd prime p, G(p) is the Gauss sum tau(chi_p) of the Legendre symbol, so |tau(chi_p)| = sqrt p.
  (3) The functional equations of L(s, chi) for chi_-4, chi_-3, chi_5, chi_12, chi_-7: with
      Lambda(s) = (q / pi)^((s + a)/2) Gamma((s + a)/2) L(s, chi), Lambda(1 - s) = Lambda(s), the root
      number tau(chi) / (i^a sqrt q) being 1.
  (4) Schroeder's quadratic-residue diffuser: wells of depth proportional to n^2 mod p reflect a plane wave
      with phases exp(2 pi i n^2 / p); the energy in every diffraction order is the same, 1/p, because every
      discrete Fourier coefficient has modulus sqrt p. Its entropy is log p, the maximum.
  (5) The Talbot effect: a grating of period 1 lit by a plane wave; in the paraxial approximation the field
      at distance u z_T (z_T = 2 d^2 / lambda) is psi(x, u) = sum c_n exp(2 pi i (n x - n^2 u)). At u = p/q the
      field is a sum of shifted copies of the grating, psi(x, p/q) = sum_m a_m g(x - m/q), with weights a_m
      given by Gauss sums: q copies of equal brightness 1/q for odd q, q/2 copies of brightness 2/q for even
      q. Checked for all p/q with q <= 12, and the decomposition itself on a grid.

What it shows
  The same identity |G(q)|^2 = q makes the Talbot images equally bright, makes the Schroeder diffuser scatter
  sound equally into every order, and normalises the functional equation of a quadratic Dirichlet L-function
  (its root number has modulus 1). Classical: Gauss; Schroeder (J. Acoust. Soc. Am. 65 (1979) 958-963);
  Berry and Klein (J. Mod. Opt. 43 (1996) 2139-2164).

What it does not claim
  Anything about the zeros of L-functions. The functional equation is a symmetry about Re s = 1/2; it does
  not locate the zeros on that line.
"""
import cmath
import math
from fractions import Fraction

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import BLUE, BLUE_RAMP, INK, INK_2, ORANGE, check, save

mp.mp.dps = 30


def G(q, a=1):
    return sum(cmath.exp(2j * math.pi * a * n * n / q) for n in range(q))


def primes(n):
    return [p for p in range(2, n + 1) if all(p % r for r in range(2, int(p ** 0.5) + 1))]


# ---------------------------------------------------------------------------------------------------
print("(1) quadratic Gauss sums")
closed = {0: 1 + 1j, 1: 1, 2: 0, 3: 1j}
check(all(abs(G(q) - closed[q % 4] * math.sqrt(q)) < 1e-8 for q in range(1, 401)),
      "G(q) = (1 + i) sqrt q, sqrt q, 0, i sqrt q for q = 0, 1, 2, 3 mod 4, all q <= 400")

# ---------------------------------------------------------------------------------------------------
print("(2) Gauss sums of the Legendre symbols")


def legendre(a, p):
    r = pow(a % p, (p - 1) // 2, p)
    return 0 if r == 0 else (1 if r == 1 else -1)


ok = True
for p in primes(500)[1:]:
    tau = sum(legendre(a, p) * cmath.exp(2j * math.pi * a / p) for a in range(1, p))
    ok &= abs(tau - G(p)) < 1e-8 and abs(abs(tau) - math.sqrt(p)) < 1e-8
check(ok, "tau(chi_p) = G(p) and |tau(chi_p)| = sqrt p for every odd prime p < 500")

# ---------------------------------------------------------------------------------------------------
print("(3) functional equations")


def kron(D, n):
    if math.gcd(D, n) != 1:
        return 0
    res, m, p = 1, n, 2
    while p * p <= m:
        while m % p == 0:
            res *= (1 if D % 8 in (1, 7) else -1) if p == 2 else legendre(D, p)
            m //= p
        p += 1
    if m > 1:
        res *= (1 if D % 8 in (1, 7) else -1) if m == 2 else legendre(D, m)
    return res


for D in (-4, -3, 5, 12, -7):
    q = abs(D)
    a = 0 if D > 0 else 1
    chi = [kron(D, n) for n in range(q)]
    tau = sum(chi[n] * mp.exp(2j * mp.pi * n / q) for n in range(q))
    eps = tau / (mp.mpc(0, 1) ** a * mp.sqrt(q))

    def Lam(s):
        return (mp.mpf(q) / mp.pi) ** ((s + a) / 2) * mp.gamma((s + a) / 2) * mp.dirichlet(s, chi)

    errs = [abs(Lam(1 - s) - Lam(s)) / abs(Lam(s)) for s in (mp.mpc(0.3, 4), mp.mpc(0.8, -2.5), mp.mpc(2, 7))]
    check(abs(eps - 1) < 1e-20 and max(errs) < 1e-15,
          f"chi_{D}: root number tau / (i^{a} sqrt {q}) = {mp.nstr(mp.re(eps), 6)}, Lambda(1 - s) = Lambda(s)")

# ---------------------------------------------------------------------------------------------------
print("(4) Schroeder's quadratic-residue diffuser")


def entropy(pw):
    pw = pw[pw > 1e-15]
    return float(-(pw * np.log(pw)).sum())


ok = True
for p in primes(200)[1:]:
    phases = np.exp(2j * np.pi * (np.arange(p) ** 2 % p) / p)
    power = np.abs(np.fft.fft(phases)) ** 2 / p ** 2
    ok &= np.allclose(power, 1 / p) and abs(entropy(power) - math.log(p)) < 1e-9
check(ok, "every odd prime p < 200: energy 1/p in each of the p orders, entropy log p (the maximum)")
rng = np.random.default_rng(1979)
p = 17
rand = np.abs(np.fft.fft(np.exp(2j * np.pi * rng.random(p)))) ** 2 / p ** 2
print(f"     p = 17: random phases give entropy {entropy(rand):.3f} < log 17 = {math.log(17):.3f}; "
      f"a flat panel gives 0")

# ---------------------------------------------------------------------------------------------------
print("(5) the Talbot effect")
width = 0.1
Nh = 300
n = np.arange(-Nh, Nh + 1)
c = width * np.sinc(n * width)          # slits of width 0.1 per period: g(x) = 1 for |x| < 0.05


def field(x, u):
    return np.exp(2j * np.pi * np.outer(x, n)) @ (c * np.exp(-2j * np.pi * n ** 2 * u))


def grating(x):
    return np.exp(2j * np.pi * np.outer(x, n)) @ c


def weights(p, q):
    nn = np.arange(q)
    return np.array([np.exp(2j * np.pi * (-p * nn ** 2 + nn * m) / q).sum() / q for m in range(q)])


ok_w, ok_d = True, True
xs = np.linspace(0, 1, 257)
for q in range(1, 13):
    for p in range(1, q):
        if math.gcd(p, q) != 1:
            continue
        a = weights(p, q)
        mods = np.abs(a)
        nz = mods[mods > 1e-12]
        if q % 2:
            ok_w &= len(nz) == q and np.allclose(nz, 1 / math.sqrt(q))
        else:
            ok_w &= len(nz) == q // 2 and np.allclose(nz, math.sqrt(2 / q))
        recon = sum(a[m] * grating(xs - m / q) for m in range(q))
        ok_d &= np.allclose(recon, field(xs, p / q), atol=1e-10)
check(ok_d, "psi(x, p/q) = sum_m a_m g(x - m/q) for every p/q with q <= 12 (Fourier series to |n| <= 300)")
check(ok_w, "odd q: q copies, each of intensity 1/q; even q: q/2 copies, each of intensity 2/q")

# ---------------------------------------------------------------------------------------------------
print("figures")
cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)

# (a) the Talbot carpet
xg = np.linspace(-0.5, 1.5, 601)
ug = np.linspace(0, 1, 1201)
I = np.abs(np.exp(2j * np.pi * np.outer(xg, n)) @ (c[:, None] * np.exp(-2j * np.pi * np.outer(n ** 2, ug)))) ** 2
I = I / I.max()
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.imshow(np.clip(I / np.quantile(I, 0.995), 0, 1) ** 0.5, origin="lower", aspect="auto", cmap=cmap,
          extent=(0, 1, -0.5, 1.5), interpolation="bilinear")
fracs = [0, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 1]
for f in fracs[1:-1]:
    ax.axvline(f, color=ORANGE, lw=1, ls=(0, (3, 3)))
ax.set_xticks(fracs)
ax.set_xticklabels(["0", "¼", "⅓", "½", "⅔", "¾", "1"])
ax.set_xlabel("distance behind the grating, in units of the Talbot length $z_T = 2d^2/\\lambda$")
ax.set_ylabel("position across\nthe grating ($x/d$)")
ax.set_yticks([-0.5, 0, 0.5, 1, 1.5])
ax.set_title("A Talbot carpet: at distance $(p/q)\\,z_T$ the weights of the images are Gauss sums")
save(fig, "waves-talbot")

# (b) Schroeder's diffuser
p = 17
s = np.arange(p) ** 2 % p
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
for k in range(2 * p):
    d = s[k % p]
    ax.add_patch(plt.Rectangle((k, -d), 0.86, d, color=BLUE, lw=0))
ax.axhline(0, color=INK_2, lw=1)
ax.set_xlim(-0.5, 2 * p)
ax.set_ylim(-p, 1)
ax.set_yticks([0, -8, -16])
ax.set_yticklabels(["0", "8", "16"])
ax.set_xticks([0, p, 2 * p])
ax.set_xlabel("well number (two periods)")
ax.set_ylabel("depth ∝ $n^2$ mod 17")
ax.set_title("Wells of depth $n^2$ mod 17", fontsize=10)
ax = axes[1]
orders = np.arange(-(p // 2), p // 2 + 1)
qrd = np.abs(np.fft.fft(np.exp(2j * np.pi * s / p))) ** 2 / p ** 2
flat = np.zeros(p)
flat[0] = 1
ax.bar(orders - 0.2, np.fft.fftshift(flat), width=0.4, color=ORANGE, label="flat panel")
ax.bar(orders + 0.2, np.fft.fftshift(qrd), width=0.4, color=BLUE, label="quadratic-residue wells")
ax.set_xlabel("diffraction order")
ax.set_ylabel("share of reflected energy")
ax.set_xticks([-8, -4, 0, 4, 8])
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right")
ax.set_title("Every order gets 1/17", fontsize=10)
save(fig, "waves-schroeder")
print("done")
