"""Note 03: light and number. Newton's rings, Balmer's formula and Planck's law.

What it computes
  (1) Newton and Young. Newton's rule (Opticks, Book II, Part I, Obs. 14): the thicknesses of air at which the
      seven confines of the colours are reflected are as the cube roots of the squares of the string lengths
      1, 8/9, 5/6, 3/4, 2/3, 3/5, 9/16, 1/2. Young's table of 1802 is recomputed from that rule and from his
      extreme red, .0000266 inch; Newton's interval of fits, 1/89000 inch, is converted to a wavelength, and
      the ratio 2^(2/3) is set against Newton's own bounds for it.
  (2) The octave. The frequency ratios of the visible band for three choices of limits; the orders of a grating
      (orders m and m + 1 of a band are separate exactly when the band is narrower than (m + 1)/m); the second
      harmonic of the ruby laser; and the identity 2 f_n - f_2n = f_0 for a comb f_n = f_0 + n f_r.
  (3) Balmer. His four coefficients 9/5, 4/3, 25/21, 9/8 times h = 3645.6 against the wavelengths of Angstrom
      that he used, his fifth line, his case n = 1, the consecutive ratios, and the span of each series.
  (4) The tones of a cubic cavity. The count 2 r_3(N) of tones of frequency (c/L) sqrt N, its sum against the
      volume term (8 pi/3) X^(3/2), the empty shells, and r_3(N) = (24/pi) sqrt N L(1, chi_D).
  (5) Planck's sum. The integral of x^3/(e^x - 1) is 6 zeta(4) = pi^4/15, term by term; the shares of Wien's
      term; the Stefan-Boltzmann constant from the defined values of h, k and c; the photons of the cosmic
      background; the surface of the Sun.
  (6) The cavity exactly. For the periodic box, with tau = k T L/(h c), the thermal energy in units of h c/L is
      (8 pi^5/15) tau^4 - 2 tau + Z(2)/(2 pi^3) - rho(tau), where Z(s) = sum r_3(N) N^(-s) is Epstein's zeta
      function of the cubic lattice, Z(-1/2) = -Z(2)/(2 pi^3), and rho is an explicit sum over the dual lattice
      of size 96 pi^3 tau^3 e^(-4 pi^2 tau).
  (7) Handedness. On each shell |k|^2 = N the curl has the eigenvalues 0, +2 pi sqrt N, -2 pi sqrt N on the
      three polarisations of each wave vector; a real field of one handedness is checked to satisfy
      curl u = 2 pi sqrt N u.

What it shows
  The numbers that Newton, Young and Balmer wrote down, the count of Rayleigh and Jeans and the constants of
  Stefan and Planck are reproduced from the sources and from Euler's and Gauss's arithmetic.
"""
import math
from fractions import Fraction

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

from common import BLUE, GRID, INK, INK_2, ORANGE, check, save

mp.mp.dps = 30
INCH = 25.4e6                      # nanometres
c = 299792458.0                    # m/s   (defined)
h = 6.62607015e-34                 # J s   (defined, SI 2019)
kB = 1.380649e-23                  # J/K   (defined, SI 2019)
R_H = 1.09677583e7                 # Rydberg constant for hydrogen, 1/m (vacuum wavelengths)

# ---------------------------------------------------------------------------------------------------
print("(1) Newton's rule and Young's table")
strings = [Fraction(1), Fraction(8, 9), Fraction(5, 6), Fraction(3, 4), Fraction(2, 3), Fraction(3, 5),
           Fraction(9, 16), Fraction(1, 2)]
young_confines = [266, 246, 235, 219, 203, 189, 181, 167]     # Young 1802, in units of 1e-7 inch
rule = [266 * float(s) ** (2 / 3) for s in strings]
check(all(abs(a - b) < 0.7 for a, b in zip(rule, young_confines)),
      "Young's eight confines, .0000266 to .0000167 inch, are Newton's rule s^(2/3) on the string "
      "1, 8/9, 5/6, 3/4, 2/3, 3/5, 9/16, 1/2 (to the last digit he printed)")
check(abs(59750 / 37640 - 2 ** (2 / 3)) < 1e-4 and abs(1200 * math.log2(2 ** (2 / 3)) - 800) < 1e-9
      and 3 / 2 < 2 ** (2 / 3) < 8 / 5 < 13 / 8 and 14.25 / 9 < 2 ** (2 / 3) < 14.34 / 9,
      f"Young's undulations per inch, 59750/37640 = {59750 / 37640:.4f} = 2^(2/3) = {2 ** (2 / 3):.4f}: eight "
      "equal semitones, under the minor sixth 8/5, inside Newton's own bounds 3/2 and 13/8 (Obs. 13)")
octaves = math.log2(561e12 / 256)
check(abs(octaves - 41) < 0.02, f"Young's yellow green, 561 millions of millions, is {octaves:.2f} octaves above 256")
red, violet = INCH / 37640, INCH / 59750
newton = 2 * INCH / 89000
check(abs(red - 674.8) < 0.1 and abs(violet - 425.1) < 0.1 and abs(newton - 570.8) < 0.1,
      f"in nanometres: Young's extremes {red:.1f} and {violet:.1f}; Newton's interval of fits 1/89000 inch is "
      f"half of {newton:.1f} nm")
young_yellow, young_confine = INCH / 44000, 235e-7 * INCH
check(abs(young_yellow - 577.3) < 0.1 and 1.04 < young_confine / newton < 1.05,
      f"Young's yellow, 44000 to the inch, is {young_yellow:.1f} nm; his confine of yellow and orange is "
      f"{young_confine:.0f} nm, {100 * (young_confine / newton - 1):.1f}% longer than Newton's length")

# ---------------------------------------------------------------------------------------------------
print("(2) the octave")
bands = {"Newton and Young": (violet, red), "380-750 nm": (380, 750), "360-830 nm (CIE)": (360, 830)}
ratios = {name: b / a for name, (a, b) in bands.items()}
check(abs(ratios["380-750 nm"] - 1.9737) < 1e-4 and abs(ratios["360-830 nm (CIE)"] - 2.3056) < 1e-4,
      "frequency ratio of the visible band: " + ", ".join(f"{n} {r:.3f}" for n, r in ratios.items()))


def orders_separate(lo, hi, m):
    """Orders m and m + 1 of a grating: sin(theta) = m lambda/d. Separate when m hi < (m + 1) lo."""
    return m * hi < (m + 1) * lo


check(orders_separate(380, 750, 1) and not orders_separate(380, 750, 2) and not orders_separate(360, 830, 1)
      and orders_separate(violet, red, 1),
      "grating, 380-750 nm: the first order is clear of the second (750 < 760) and the second overlaps the third; "
      "with the limits of the CIE tables the first two orders overlap")
check(abs(694.3 / 2 - 347.15) < 1e-9, "second harmonic of the ruby laser: 694.3 nm -> 347.15 nm (Franken et al.)")
f0, fr = Fraction(35_000_000), Fraction(1_000_000_000)        # an example comb, in Hz
tooth = lambda n: f0 + n * fr
check(all(2 * tooth(n) - tooth(2 * n) == f0 for n in (300_000, 400_000, 500_000)),
      "comb f_n = f_0 + n f_r: 2 f_n - f_2n = f_0, so a comb that spans an octave measures its own offset")

# ---------------------------------------------------------------------------------------------------
print("(3) Balmer's formula")
hB = 3645.6                                                    # Balmer's fundamental number, 1e-7 mm
coeff = [Fraction(m * m, m * m - 4) for m in range(3, 8)]
check(coeff[:4] == [Fraction(9, 5), Fraction(4, 3), Fraction(25, 21), Fraction(9, 8)],
      "Balmer's coefficients m^2/(m^2 - 4), m = 3..6: 9/5, 4/3, 25/21, 9/8")
angstrom = [6562.10, 4860.74, 4340.1, 4101.2]                  # Angstrom's wavelengths as Balmer quotes them
worst = max(abs(hB * float(q) - a) / a for q, a in zip(coeff, angstrom))
check(worst < 1 / 40000, f"h = 3645.6 times the coefficients against Angstrom: worst deviation 1/{1 / worst:.0f} "
      "(Balmer: not more than 1/40000)")
check(abs(hB * float(coeff[4]) - 3969.65) < 0.01, "Balmer's fifth line: 49/45 x 3645.6 = 3969.65")
check(abs(4e10 / R_H / 1.000283 - hB) < 1.0,
      f"the limit 4/R_H = {4e10 / R_H:.1f} (vacuum), {4e10 / R_H / 1.000283:.1f} in air: Balmer's 3645.6")
nu = lambda m, n: Fraction(1, m * m) - Fraction(1, n * n)
consecutive = [nu(2, n + 1) / nu(2, n) for n in range(3, 7)]
check(consecutive == [Fraction(27, 20), Fraction(28, 25), Fraction(200, 189), Fraction(405, 392)],
      f"consecutive Balmer frequency ratios {', '.join(map(str, consecutive))}")
spans = {m: Fraction(1, m * m) / nu(m, m + 1) for m in range(1, 5)}
check(spans == {m: Fraction((m + 1) ** 2, 2 * m + 1) for m in range(1, 5)}
      and spans[1] == Fraction(4, 3) and spans[2] == coeff[0],
      "series spans (m + 1)^2/(2m + 1): Lyman 4/3, Balmer 9/5 (his first coefficient), Paschen 16/7, Brackett 25/9")
case_one = [Fraction(m * m, m * m - 1) for m in range(2, 6)]
check(case_one == [Fraction(4, 3), Fraction(9, 8), Fraction(16, 15), Fraction(25, 24)]
      and case_one == [Fraction(m * m, m * m - 4) for m in (4, 6, 8, 10)],
      "Balmer's case n = 1, 4/3, 9/8, 16/15, 25/24, is every second coefficient of his own series (m = 4, 6, 8, 10)")
lyman = [1e9 / (R_H * float(nu(1, n))) for n in range(2, 6)]
check(all(abs(4e9 / R_H / 4 * float(q) - lam) < 1e-9 for q, lam in zip(case_one, lyman))
      and abs(lyman[0] - 121.57) < 0.01,
      "with a fundamental number four times smaller the same coefficients are Lyman's series: "
      + ", ".join(f"{lam:.2f}" for lam in lyman) + " nm")
balmer_nm = {n: 1e9 / (R_H * float(nu(2, n))) for n in range(3, 8)}

# ---------------------------------------------------------------------------------------------------
print("(4) the tones of a cubic cavity")
NMAX = 20000
r1 = np.zeros(NMAX + 1, dtype=np.int64)
for n in range(-int(math.isqrt(NMAX)), int(math.isqrt(NMAX)) + 1):
    r1[n * n] += 1
r2 = np.convolve(r1, r1)[: NMAX + 1]
r3 = np.convolve(r2, r1)[: NMAX + 1]
check(list(r3[1:9]) == [6, 12, 8, 6, 24, 24, 0, 12], "r_3(N) for N = 1..8: 6, 12, 8, 6, 24, 24, 0, 12")
empty = lambda N: any(N % 4 ** a == 0 and (N // 4 ** a) % 8 == 7 for a in range(8))
check(all((r3[N] == 0) == empty(N) for N in range(1, NMAX + 1)),
      f"no tone exactly at N = 4^a (8b + 7), N <= {NMAX} (Legendre, Gauss): 7, 15, 23, 28, 31, ...")
X = 10000
count, weyl = 2 * int(r3[1: X + 1].sum()), 8 * math.pi / 3 * X ** 1.5
check(abs(count / weyl - 1) < 2e-3,
      f"tones up to N = {X}: {count}; volume term (8 pi/3) X^(3/2) = {weyl:.0f}; ratio {count / weyl:.5f}")


def kron(D, n):
    """Kronecker symbol (D/n) for a fundamental discriminant D and n >= 1."""
    res = 1
    while n % 2 == 0:
        n //= 2
        res *= 0 if D % 2 == 0 else (1 if D % 8 in (1, 7) else -1)
    p = 3
    while n > 1:
        while n % p == 0:
            n //= p
            res *= 0 if D % p == 0 else (1 if pow(D % p, (p - 1) // 2, p) == 1 else -1)
        p += 2
    return res


def L1(D):
    """L(1, chi_D) for a fundamental discriminant D < 0, by the finite formula."""
    return -math.pi / abs(D) ** 1.5 * sum(a * kron(D, a) for a in range(1, abs(D)))


squarefree = lambda N: all(N % (p * p) for p in range(2, int(math.isqrt(N)) + 1))
ok = True
for N in range(1, 200):
    if squarefree(N) and N % 8 != 7:
        D = -N if N % 4 == 3 else -4 * N
        ok &= abs(r3[N] - 24 / math.pi * math.sqrt(N) * L1(D)) < 1e-8
check(ok, "r_3(N) = (24/pi) sqrt N L(1, chi_D) for squarefree N < 200, N not 7 mod 8 (Gauss, Dirichlet)")

# ---------------------------------------------------------------------------------------------------
print("(5) Planck's sum")
integral = mp.quad(lambda x: x ** 3 / mp.expm1(x), [0, 1, 10, 50, mp.inf])
check(abs(integral - mp.pi ** 4 / 15) < 1e-20 and abs(6 * mp.zeta(4) - mp.pi ** 4 / 15) < 1e-25,
      "integral of x^3/(e^x - 1) = sum of 6/m^4 = 6 zeta(4) = pi^4/15 (Euler: zeta(4) = pi^4/90)")
check(abs(mp.quad(lambda x: x ** 3 * mp.exp(-3 * x), [0, mp.inf]) - mp.mpf(6) / 81) < 1e-20,
      "term by term: the j-th quantum of every tone carries the integral of x^3 e^(-j x) = 6/j^4")
check(abs(mp.quad(lambda x: x ** 2 / mp.expm1(x), [0, 1, 10, 50, mp.inf]) - 2 * mp.zeta(3)) < 1e-20,
      "integral of x^2/(e^x - 1) = 2 zeta(3): the number of photons")
wien_energy, wien_number = 1 / mp.zeta(4), 1 / mp.zeta(3)
check(abs(wien_energy - 0.92394) < 1e-5 and abs(wien_number - 0.83191) < 1e-5,
      f"Wien's term j = 1 carries {float(wien_energy):.4f} of the energy and {float(wien_number):.4f} of the photons")
sigma = 2 * math.pi ** 5 * kB ** 4 / (15 * h ** 3 * c ** 2)
check(abs(sigma - 5.670374419e-8) < 1e-17, f"Stefan-Boltzmann constant 2 pi^5 k^4/(15 h^3 c^2) = {sigma:.9e} W/m^2/K^4")
photons = lambda T: 16 * math.pi * float(mp.zeta(3)) * (kB * T / (h * c)) ** 3
check(abs(photons(2.7255) / 1e6 - 410.7) < 0.1,
      f"cosmic background at 2.7255 K: 16 pi zeta(3) (kT/hc)^3 = {photons(2.7255) / 1e6:.1f} photons per cm^3")
sun_flux = sigma * 5772 ** 4
sun_lum = 4 * math.pi * 6.957e8 ** 2 * sun_flux
check(abs(sun_flux / 1e7 - 6.294) < 1e-3 and abs(sun_lum / 3.828e26 - 1) < 1e-3,
      f"the Sun at 5772 K: {sun_flux / 1e6:.1f} MW per m^2, and {sun_lum:.3e} W from its nominal radius "
      "(IAU nominal 3.828e26 W)")
check(abs(sigma * 300 ** 4 - 459.3) < 0.1, f"a black body at 300 K: {sigma * 300 ** 4:.1f} W per m^2")

# ---------------------------------------------------------------------------------------------------
print("(6) the cavity exactly")
theta = lambda t: 1 + 2 * mp.nsum(lambda n: mp.exp(-mp.pi * n * n * t), [1, mp.inf])


def epstein_completed(s):
    """pi^(-s) Gamma(s) Z(s) for the cubic lattice, by Riemann's method; symmetric under s -> 3/2 - s."""
    integrand = lambda t: (theta(t) ** 3 - 1) * (t ** (s - 1) + t ** (mp.mpf(1) / 2 - s))
    return mp.quad(integrand, [1, 2, 4, 8, 16, 40]) - 1 / s - 1 / (mp.mpf(3) / 2 - s)


Z2 = epstein_completed(mp.mpf(2)) * mp.pi ** 2
direct = float((r3[1:] / np.arange(1, NMAX + 1) ** 2.0).sum()) + 4 * math.pi / math.sqrt(NMAX + 0.5)
check(abs(Z2 - mp.mpf("16.5323159598")) < 1e-9 and abs(direct - float(Z2)) < 1e-3,
      f"Epstein zeta of the cubic lattice: Z(2) = sum r_3(N)/N^2 = {mp.nstr(Z2, 12)} (direct sum {direct:.4f})")
Zm = epstein_completed(mp.mpf(-1) / 2) / (mp.sqrt(mp.pi) * mp.gamma(mp.mpf(-1) / 2))
check(abs(Zm + Z2 / (2 * mp.pi ** 3)) < 1e-20 and abs(Zm + mp.mpf("0.2665961")) < 1e-6,
      f"functional equation: Z(-1/2) = -Z(2)/(2 pi^3) = {mp.nstr(Zm, 8)}: the zero-point energy in units of hc/L")


def thermal(tau):
    """Sum over n in Z^3, n != 0, of |n|/(e^(|n|/tau) - 1): half the thermal energy in units of hc/L."""
    top = int(min(NMAX, (60 * tau) ** 2))
    N = np.arange(1, top + 1)
    root = np.sqrt(N)
    return float((r3[1: top + 1] * root / np.expm1(root / tau)).sum())


def rho_half(tau):
    """Half of rho(tau): sum over xi != 0 of (2 pi^3 tau^3/|xi|) coth(a)/sinh(a)^2, a = 2 pi^2 tau |xi|."""
    total = 0.0
    for N in range(1, 400):
        a = 2 * math.pi ** 2 * tau * math.sqrt(N)
        if r3[N] and a < 300:
            total += r3[N] * 2 * math.pi ** 3 * tau ** 3 / math.sqrt(N) / math.tanh(a) / math.sinh(a) ** 2
    return total


closed = lambda tau: 4 * math.pi ** 5 / 15 * tau ** 4 - tau + float(Z2) / (4 * math.pi ** 3) - rho_half(tau)
worst = max(abs(thermal(tau) / closed(tau) - 1) for tau in (0.2, 0.3, 0.5, 1.0, 1.5, 2.0))
check(worst < 1e-12,
      "thermal energy of the periodic box = (hc/L) [(8 pi^5/15) tau^4 - 2 tau + Z(2)/(2 pi^3) - rho(tau)] for "
      f"tau = 0.2 to 2: worst relative difference {worst:.1e}")
check(abs(2 * rho_half(0.5) - 1.0e-6) < 1e-7 and 2 * rho_half(1.0) < 1e-12
      and abs(2 * rho_half(1.5) / (96 * math.pi ** 3 * 1.5 ** 3 * math.exp(-4 * math.pi ** 2 * 1.5)) - 1) < 1e-6,
      f"rho(1/2) = {2 * rho_half(0.5):.2e}, rho(1) = {2 * rho_half(1.0):.1e}, and rho(tau) -> "
      "96 pi^3 tau^3 e^(-4 pi^2 tau)")
check(abs(thermal(0.05) / (6 / math.expm1(20)) - 1) < 1e-2 and 2 * thermal(0.05) < 1e-7,
      "at tau = 0.05 only the six lowest tones are excited: the formula is one of high temperature")

# ---------------------------------------------------------------------------------------------------
print("(7) handedness")
rng = np.random.default_rng(3)
ok = True
for N in (1, 2, 3, 5, 6, 9, 14):
    m = int(math.isqrt(N))
    shell = [np.array(v) for v in np.ndindex(2 * m + 1, 2 * m + 1, 2 * m + 1)]
    shell = [v - m for v in shell if ((v - m) ** 2).sum() == N]
    ok &= len(shell) == r3[N]
    plus = 0
    amplitude = {}
    for kv in shell:
        cross = np.array([[0, -kv[2], kv[1]], [kv[2], 0, -kv[0]], [-kv[1], kv[0], 0]])
        w, V = np.linalg.eigh(1j * cross)                     # curl acts on e^(2 pi i k.x) as 2 pi (i k x)
        ok &= np.allclose(sorted(w), [-math.sqrt(N), 0, math.sqrt(N)])
        plus += int(np.sum(np.isclose(w, math.sqrt(N))))
        amplitude[tuple(kv)] = V[:, int(np.argmax(w))]
    ok &= plus == r3[N]
    coef = {kv: rng.normal() + 1j * rng.normal() for kv in amplitude}
    field = lambda x: sum((coef[kv] * amplitude[kv] * np.exp(2j * math.pi * np.dot(kv, x))).real for kv in coef)
    x0, d = rng.random(3), 1e-5
    J = np.array([(field(x0 + d * e) - field(x0 - d * e)) / (2 * d) for e in np.eye(3)])   # J[j, i] = d_j u_i
    curl = np.array([J[1, 2] - J[2, 1], J[2, 0] - J[0, 2], J[0, 1] - J[1, 0]])
    ok &= np.allclose(curl, 2 * math.pi * math.sqrt(N) * field(x0), atol=1e-4)
check(ok, "on each shell the curl has eigenvalues 0, +-2 pi sqrt N; one handedness has r_3(N) wave vectors, "
      "and a real field built on it satisfies curl u = 2 pi sqrt N u (N = 1, 2, 3, 5, 6, 9, 14)")

# ---------------------------------------------------------------------------------------------------
print("figures")


def rgb(lam):
    """Approximate sRGB colour of a wavelength in nm (Bruton's piecewise-linear approximation)."""
    if lam < 440:
        r, g, b = (440 - lam) / 60, 0, 1
    elif lam < 490:
        r, g, b = 0, (lam - 440) / 50, 1
    elif lam < 510:
        r, g, b = 0, 1, (510 - lam) / 20
    elif lam < 580:
        r, g, b = (lam - 510) / 70, 1, 0
    elif lam < 645:
        r, g, b = 1, (645 - lam) / 65, 0
    else:
        r, g, b = 1, 0, 0
    f = 0.3 + 0.7 * (lam - 380) / 40 if lam < 420 else (0.3 + 0.7 * (780 - lam) / 80 if lam > 700 else 1)
    f = min(max(f, 0.0), 1.0)
    return tuple(min(max(f * v, 0.0), 1.0) ** 0.8 for v in (r, g, b))


lams = np.linspace(350, 840, 900)
fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.imshow(np.array([[rgb(l) for l in lams]]), extent=(350, 840, 3.25, 3.85), aspect="auto")
rows = [("Newton 1704, Young 1802", violet, red, 2.75, r"$2^{2/3}=1.587$"),
        ("common limits", 380, 750, 2.2, "1.974"),
        ("CIE tables", 360, 830, 1.65, "2.306")]
for name, lo, hi, y, label in rows:
    ax.plot([lo, hi], [y, y], color=BLUE, lw=5, solid_capstyle="butt")
    ax.text(346, y, name, ha="right", va="center", fontsize=8.5, color=INK_2)
    ax.text(hi + 5, y, label, ha="left", va="center", fontsize=8.5, color=INK)
confines = [red * float(s) ** (2 / 3) for s in strings]
ax.plot(confines, [2.75] * 8, "|", color="white", ms=7, mew=1.3)
ax.plot([newton], [2.75], "v", color=ORANGE, ms=6)
ax.text(newton, 2.98, "Newton's fits, 571 nm", ha="center", va="bottom", fontsize=7.5, color=INK)
ax.text(346, 3.55, "spectrum\n(approximate)", ha="right", va="center", fontsize=8.5, color=INK_2)
names = [r"H$\alpha$", r"H$\beta$", r"H$\gamma$", r"H$\delta$", r"H$\epsilon$"]
labels = ["9/5", "4/3", "25/21", "9/8", "49/45"]
for i, (n, lam) in enumerate(balmer_nm.items()):
    ax.plot([lam, lam], [0.5, 1.1], color=INK, lw=1.5)
    lift = 0.22 if i == 3 else 0.0                              # H-delta sits between two close neighbours
    ax.text(lam, 1.14 + lift, names[i], fontsize=8.5, ha="center", va="bottom", color=INK)
    ax.text(lam, 0.46 - lift, labels[i], fontsize=7.5, ha="center", va="top", color=INK_2)
limit = 4e9 / R_H
ax.plot([limit, limit], [0.5, 1.1], color=INK, lw=1.5, ls=(0, (2, 2)))
ax.text(limit, 0.46, "limit", fontsize=7.5, ha="center", va="top", color=INK_2)
ax.text(346, 0.8, "hydrogen\n(Balmer)", ha="right", va="center", fontsize=8.5, color=INK_2)
ax.set_xlim(350, 880)
ax.set_ylim(0, 4)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("wavelength (nm)")
ax.set_title("The visible band, its frequency ratio, and the lines of hydrogen")
save(fig, "light-spectrum")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
XM = 64
Ns = np.arange(0, XM + 1)
stair = 2 * np.concatenate([[0], np.cumsum(r3[1: XM + 1])])
a1.step(Ns, stair, where="post", color=BLUE, lw=1.6, label=r"tones up to $N$: $2\sum r_3$")
xs = np.linspace(0, XM, 300)
a1.plot(xs, 8 * math.pi / 3 * xs ** 1.5, color=ORANGE, lw=1.4, label=r"volume term $\frac{8\pi}{3}N^{3/2}$")
gaps = [N for N in range(1, XM + 1) if r3[N] == 0]
a1.plot(gaps, [0] * len(gaps), "x", color=INK, ms=5, mew=1.2, label="empty shells", clip_on=False)
a1.set_xlabel(r"$N=(L\nu/c)^2$")
a1.set_ylabel("number of tones")
a1.set_title("Tones of a cubic cavity")
a1.legend(loc="upper left")
a1.grid(True, axis="y")
xx = np.linspace(1e-3, 12, 500)
a2.plot(xx, xx ** 3 / np.expm1(xx), color=INK, lw=1.8, label="Planck")
for j, col in ((1, BLUE), (2, ORANGE)):
    share = float(1 / (j ** 4 * mp.zeta(4)))
    a2.plot(xx, xx ** 3 * np.exp(-j * xx), color=col, lw=1.4,
            label=rf"$j={j}$: {100 * share:.1f}% of the energy")
a2.set_xlabel(r"$x=h\nu/kT$")
a2.set_ylabel(r"energy per unit $x$")
a2.set_title(r"Planck's law as a sum: $6\zeta(4)=\pi^4/15$")
a2.legend(loc="upper right")
a2.grid(True, axis="y")
fig.tight_layout()
save(fig, "light-cavity")
print("done")
