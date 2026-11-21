"""Note 13: speed, spread and swirl. Three equations of motion solved on a lattice.

What it computes
  (1) Speed. A positive definite symmetric 2 x 2 matrix A = [[a, b], [b, c]] of determinant 1 is the event
      (t, x, y) = ((a + c)/2, (a - c)/2, b) on the unit hyperboloid t^2 - x^2 - y^2 = 1 of 2+1 Minkowski space, the
      velocity of a body. For two such matrices the relative Lorentz factor is gamma = tr(A B^-1)/2 = cosh(delta),
      with delta the hyperbolic distance of the shapes; the functional S = tr(G - 1 - log G) of note 12 is 2(gamma - 1),
      twice the kinetic energy in units of the rest energy. SL(2, Z) acts on (a, b, c) by integer Lorentz
      transformations. For the square lattice, gamma(i, g i) = (a^2 + b^2 + c^2 + d^2)/2. The golden boost
      g = (2 1; 1 1) moves the frames on its axis at 3 sqrt5/7 (gamma = 7/2), with rapidity 4 log phi = 2 sqrt5 L(1, chi_5);
      n golden boosts give speed sqrt5 F_4n / L_4n, and Einstein's addition rule is the addition formula of the
      Fibonacci and Lucas numbers. Every hyperbolic g moves every frame at gamma >= 7/2.
  (2) Spread. The heat kernel of the circle is the theta function, as a sum of images and as a sum of modes
      (Poisson duality). Burgers' equation u_t + (u . grad) u = nu Lap u with a lattice of point explosions at t = 0:
      by the Cole-Hopf substitution u = -2 nu grad log(phi), phi solves the heat equation and is the theta function of
      the lattice; u = (x - <n>)/t with <n> a Gaussian average of lattice points. For small nu t, u = (x - n*)/t
      with n* the nearest lattice point (Voronoi cells); for large nu t, u decays as exp(-4 pi^2 nu N t) on the
      shortest dual vectors. Square and hexagonal lattices, in two dimensions.
  (3) Swirl. In the periodic box R^d / Z^d a velocity field built from the r_d(N) Fourier modes with |k|^2 = N of
      a stream function (d = 2) or of curl eigenfields (Beltrami fields, d = 3) has (u . grad) u a gradient, so it is
      a steady flow of the Euler equations; since Lap u = -4 pi^2 N u, it decays as exp(-4 pi^2 nu N t) under the
      Navier-Stokes equations. The Euler property is checked spectrally (FFT), with a mixed example that fails; the dimensions r_2(N) and r_3(N), with
      r_3(N) = (24/pi) sqrt N L(1, chi_D) for squarefree N not 7 (mod 8), and the ABC flow at N = 1.

What it shows
  Relativistic kinetic energy, viscous spreading and swirling flow are each computed by an object of the
  number-theoretic setting: the hyperbolic distance on the space of lattice shapes, the theta function of a lattice,
  and the number of lattice points on a sphere.
"""
import itertools
import math

import numpy as np

from common import AQUA, BLUE, GRID, INK, INK_2, ORANGE, check, plt, save

PHI = (1 + math.sqrt(5)) / 2
RNG = np.random.default_rng(13)


# ---------------------------------------------------------------- (1) speed
def M(z):
    x, y = z.real, z.imag
    return np.array([[(x * x + y * y) / y, x / y], [x / y, 1 / y]])


def event(A):
    a, b, c = A[0, 0], A[0, 1], A[1, 1]
    return np.array([(a + c) / 2, (a - c) / 2, b])


def minkowski(u, v):
    return u[0] * v[0] - u[1] * v[1] - u[2] * v[2]


def lorentz_factor(A, B):
    return 0.5 * np.trace(A @ np.linalg.inv(B))


def mobius(g, z):
    (a, b), (c, d) = g
    return (a * z + b) / (c * z + d)


def S(A, B):
    G = A @ np.linalg.inv(B)
    return float(np.trace(G) - 2 - math.log(np.linalg.det(G)))


zs = [complex(RNG.uniform(-2, 2), RNG.uniform(0.2, 3)) for _ in range(200)]
worst = 0.0
for z, w in zip(zs[::2], zs[1::2]):
    A, B = M(z), M(w)
    u, v = event(A), event(B)
    g = lorentz_factor(A, B)
    cosh_d = 1 + abs(z - w) ** 2 / (2 * z.imag * w.imag)
    worst = max(worst, abs(minkowski(u, u) - 1), abs(minkowski(u, v) - g), abs(g - cosh_d) / g,
                abs(S(A, B) - 2 * (g - 1)) / g)
check(worst < 1e-12, "M(z) is a unit timelike event; tr(M(z) M(w)^-1)/2 = <u, v> = cosh delta; "
      "S of note 12 = 2(gamma - 1) (100 random pairs)")
betas = np.array([1e-1, 1e-2, 1e-3])
from fractions import Fraction
coef = [2 * Fraction(math.comb(2 * k, k), 4 ** k) for k in range(1, 4)]   # S = 2((1 - v^2)^(-1/2) - 1)
vv = 1e-2
Sv = 2 * (1 / math.sqrt(1 - vv ** 2) - 1)
check(coef == [1, Fraction(3, 4), Fraction(5, 8)] and abs((Sv - vv ** 2 - 0.75 * vv ** 4) / vv ** 6 - 0.625) < 1e-3,
      "slow bodies: S = v^2 + (3/4) v^4 + (5/8) v^6 + ..., so (S/2) m c^2 = m v^2 / 2 to leading order")
RHO = complex(-0.5, math.sqrt(3) / 2)
g_ir = lorentz_factor(M(1j), M(RHO))
check(abs(g_ir - 2 / math.sqrt(3)) < 1e-14 and abs(math.sqrt(1 - 1 / g_ir ** 2) - 0.5) < 1e-14,
      "square and hexagonal frames: gamma = 2/sqrt 3, relative speed c/2 (note 08)")

# integer Lorentz transformations: (a, b, c) -> entries of g A g^T
def lorentz_matrix(g):
    cols = []
    for A in (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [1, 0]]), np.array([[0, 0], [0, 1]])):
        B = g @ A @ g.T
        cols.append([B[0, 0], B[0, 1], B[1, 1]])
    return np.array(cols).T


q = np.array([[0, 0, 0.5], [0, -1, 0], [0.5, 0, 0]])   # ac - b^2 in the coordinates (a, b, c)
ok = True
for a, b, c, d in itertools.product(range(-5, 6), repeat=4):
    if a * d - b * c != 1:
        continue
    g = np.array([[a, b], [c, d]])
    L = lorentz_matrix(g)
    ok &= np.allclose(L.T @ q @ L, q) and np.all(L == np.round(L))
    gam = lorentz_factor(M(1j), M(mobius(((a, b), (c, d)), 1j)))
    ok &= abs(gam - (a * a + b * b + c * c + d * d) / 2) < 1e-9
    ok &= abs(2 * (gam - 1) - ((a - d) ** 2 + (b + c) ** 2)) < 1e-9
check(ok, "SL(2, Z) acts on forms by integer matrices preserving ac - b^2; the square frame and its image "
      "under (a b; c d) have gamma = (a^2 + b^2 + c^2 + d^2)/2 and S = 2(gamma - 1) = (a - d)^2 + (b + c)^2 "
      "(all matrices with entries |.| <= 5)")
r2c = lambda n: sum(1 for x in range(-9, 10) for y in range(-9, 10) if x * x + y * y == n)
NMAX_E = 60
count = {}
for a, b, c, d in itertools.product(range(-8, 9), repeat=4):    # S <= 60 forces |a|, |b|, |c|, |d| <= 8
    if a * d - b * c == 1 and (a - d) ** 2 + (b + c) ** 2 <= NMAX_E:
        n = (a - d) ** 2 + (b + c) ** 2
        count[n] = count.get(n, 0) + 1
energies = sorted(count)
check(energies == [n for n in range(NMAX_E + 1) if r2c(n) and r2c(n + 4)],
      "the kinetic energies of the square frame, in units of m c^2 / 2, are the n with n and n + 4 both sums of "
      "two squares: " + ", ".join(map(str, energies[:12])) + ", ... (n <= 60)")
check(all(count[n] == r2c(n) * r2c(n + 4) // (1 if n % 4 == 0 else 2) for n in energies) and count[0] == 4,
      "the number of matrices in SL(2, Z) giving energy n is r_2(n) r_2(n + 4), halved for n odd; "
      "n = 0: the 4 matrices +-I, +-(0 -1; 1 0)")
# the hyperbolic lattice-point count: #{g : S <= X} ~ 6X
XC = 200000
RC = math.isqrt(XC + 4) + 1
c_all = np.zeros(XC + 5, dtype=np.int64)
c_odd = np.zeros(XC + 5, dtype=np.int64)
for pp in range(-RC, RC + 1):
    if pp * pp > XC + 4:
        continue
    qm = math.isqrt(XC + 4 - pp * pp)
    vals = pp * pp + np.arange(-qm, qm + 1) ** 2
    np.add.at(c_all, vals, 1)
    if pp % 2:
        np.add.at(c_odd, vals, 1)
c_even = c_all - c_odd
total = int(np.sum(c_odd[:XC + 1] * c_odd[4:XC + 5] + c_even[:XC + 1] * c_even[4:XC + 5]))
check(all(total_n == count[n] for n, total_n in ((n, int(c_odd[n] * c_odd[n + 4] + c_even[n] * c_even[n + 4]))
                                                 for n in energies))
      and abs(total / XC - 6) < 1e-3,
      f"#{{g in SL(2, Z) : S <= X}} = {total} at X = 2e5, that is {total / XC:.4f} X (hyperbolic lattice-point "
      "problem at i: 6X)")

# the golden boost
G = np.array([[2, 1], [1, 1]])
z_axis = complex(0.5, math.sqrt(5) / 2)                    # on the axis: the semicircle over the fixed points
F = [0, 1]
while len(F) < 60:
    F.append(F[-1] + F[-2])
Lu = [2, 1]
while len(Lu) < 60:
    Lu.append(Lu[-1] + Lu[-2])
ok = True
Gn = np.eye(2, dtype=np.int64)
Az = M(z_axis)
for n in range(1, 6):
    Gn = Gn @ G
    Gi = np.array([[Gn[1, 1], -Gn[0, 1]], [-Gn[1, 0], Gn[0, 0]]])   # exact inverse
    gam = 0.5 * np.trace(Az @ Gi.T @ np.linalg.inv(Az) @ Gi)       # tr(M(z) M(g z)^-1)/2, M(g z) = g M(z) g^T
    beta = math.sqrt(1 - 1 / gam ** 2)
    ok &= abs(gam - Lu[4 * n] / 2) / gam < 1e-12 and abs(beta - math.sqrt(5) * F[4 * n] / Lu[4 * n]) < 1e-12
    ok &= abs(math.acosh(gam) - n * 4 * math.log(PHI)) < 1e-11
axis = [1j] + [0.5 + math.sqrt(5) / 2 * complex(math.cos(th), math.sin(th)) for th in RNG.uniform(0.1, 3.0, 20)]
for z in axis:
    Az = M(z)
    Gn = np.eye(2, dtype=np.int64)
    for n in range(1, 4):
        Gn = Gn @ G
        Gi = np.array([[Gn[1, 1], -Gn[0, 1]], [-Gn[1, 0], Gn[0, 0]]])
        ok &= abs(0.5 * np.trace(Az @ Gi.T @ np.linalg.inv(Az) @ Gi) - Lu[4 * n] / 2) < 1e-9 * Lu[4 * n]
check(ok, "on the axis of (2 1; 1 1) (i and 20 random points; n = 1..5 at the top): n golden boosts give "
      "gamma = L_4n / 2, speed sqrt5 F_4n / L_4n, rapidity 4n log phi; "
      "n = 1: gamma = 7/2, speed 3 sqrt5/7 = 0.9583")
check(abs(math.sqrt(5) * 21 / 47 - 0.99909) < 5e-6 and (F[8], Lu[8]) == (21, 47),
      f"two golden boosts: speed 21 sqrt5 / 47 = {math.sqrt(5) * 21 / 47:.5f}")
check(all(2 * F[m + n] == F[m] * Lu[n] + F[n] * Lu[m] and 2 * Lu[m + n] == Lu[m] * Lu[n] + 5 * F[m] * F[n]
          for m in range(25) for n in range(25)),
      "Einstein's addition rule for golden boosts follows from 2F_(m+n) = F_m L_n + F_n L_m, 2L_(m+n) = L_m L_n + 5 F_m F_n")
chi5 = {1: 1, 2: -1, 3: -1, 4: 1}
L1chi5 = -sum(chi5[a] * math.log(math.sin(math.pi * a / 5)) for a in range(1, 5)) / math.sqrt(5)
check(abs(4 * math.log(PHI) - 2 * math.sqrt(5) * L1chi5) < 1e-14,
      f"the golden rapidity 4 log phi = 2 sqrt5 L(1, chi_5), L(1, chi_5) = {L1chi5:.12f}")
low = math.inf
for a, b, c, d in itertools.product(range(-6, 7), repeat=4):
    if a * d - b * c != 1 or abs(a + d) <= 2:
        continue
    for z in zs[:40]:
        low = min(low, lorentz_factor(M(z), M(mobius(((a, b), (c, d)), z))))
eq = []
for a, b, c, d in itertools.product(range(-6, 7), repeat=4):
    if a * d - b * c != 1 or abs(a + d) != 3 or c == 0:
        continue
    disc = math.sqrt((a + d) ** 2 - 4)
    x1, x2 = ((a - d) + disc) / (2 * c), ((a - d) - disc) / (2 * c)       # fixed points on the real line
    top = complex((x1 + x2) / 2, abs(x1 - x2) / 2)
    eq.append(lorentz_factor(M(top), M(mobius(((a, b), (c, d)), top))))
check(low >= 3.5 - 1e-9 and len(eq) > 20 and max(abs(e - 3.5) for e in eq) < 1e-9,
      f"every hyperbolic g in SL(2, Z) (entries |.| <= 6) gives every sampled frame gamma >= 7/2 (least {low:.4f}); "
      f"equality on the axis of each of the {len(eq)} elements of trace +-3 with c != 0")

# figure 1: the velocity disc and the energy spectrum of the square frame
fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.3})
ax = axes[0]
th = np.linspace(0, 2 * math.pi, 400)
ax.plot(np.cos(th), np.sin(th), color=INK_2, lw=1)
seen = {}
for a, b, c, d in itertools.product(range(-4, 5), repeat=4):
    n = (a - d) ** 2 + (b + c) ** 2
    if a * d - b * c == 1 and n <= 13:
        e = event(M(mobius(((a, b), (c, d)), 1j)))
        seen[(round(e[1] / e[0], 9), round(e[2] / e[0], 9))] = n
shade = {0: INK, 1: BLUE, 4: AQUA, 5: ORANGE, 9: INK_2, 13: INK_2}
label = {0: "n = 0: at rest", 1: "n = 1: v = (√5/3) c", 4: "n = 4: v = (2√2/3) c", 5: "n = 5: v = (3√5/7) c",
         9: "n = 9, 13: faster", 13: None}
for n in (0, 1, 4, 5, 9, 13):
    v = np.array([k for k, m in seen.items() if m == n])
    ax.plot(v[:, 0], v[:, 1], "o", ms=6 if n <= 5 else 3.5, color=shade[n], mec="white", mew=0.8,
            label=label[n], ls="none")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.5, handletextpad=0.2,
          columnspacing=0.8)
ax.set_aspect("equal")
ax.set_xlim(-1.08, 1.08)
ax.set_ylim(-1.08, 1.08)
ax.set_xlabel("v_x / c")
ax.set_ylabel("v_y / c")
ax.set_title("Velocities of the moved square frame", fontsize=9.5)
ax = axes[1]
ns = np.arange(0, NMAX_E + 1)
cnt = np.array([count.get(n, 0) for n in ns])
ax.vlines(ns[cnt > 0], 0, cnt[cnt > 0], color=BLUE, lw=2)
ax.plot(ns[cnt > 0], cnt[cnt > 0], "o", color=BLUE, ms=4)
ax.plot(ns[cnt == 0], np.zeros((cnt == 0).sum()), "x", color=ORANGE, ms=4, mew=1.2)
ax.set_xlabel("kinetic energy n, in units of m c² / 2")
ax.set_ylabel("number of matrices")
ax.set_title("Energies reached: n and n + 4 sums of two squares", fontsize=9.5)
ax.grid(axis="y", color=GRID, lw=0.6)
save(fig, "motion-velocities")


# ---------------------------------------------------------------- (2) spread
def heat2_images(x, s, basis, m=12):
    pts = np.array([i * basis[0] + j * basis[1] for i in range(-m, m + 1) for j in range(-m, m + 1)])
    return np.sum(np.exp(-((x - pts) ** 2).sum(1) / (4 * s))) / (4 * math.pi * s)


def heat2_modes(x, s, basis, m=12):
    dual = np.linalg.inv(basis).T
    ks = np.array([i * dual[0] + j * dual[1] for i in range(-m, m + 1) for j in range(-m, m + 1)])
    V = abs(np.linalg.det(basis))
    return float(np.sum(np.exp(-4 * math.pi ** 2 * s * (ks ** 2).sum(1)) * np.cos(2 * math.pi * ks @ x))) / V



def heat_images(x, s, m=30):
    n = np.arange(-m, m + 1)
    return np.sum(np.exp(-(x - n) ** 2 / (4 * s))) / math.sqrt(4 * math.pi * s)


def heat_modes(x, s, m=30):
    n = np.arange(-m, m + 1)
    return float(np.sum(np.exp(-4 * math.pi ** 2 * n ** 2 * s) * np.cos(2 * math.pi * n * x)))


check(max(abs(heat_images(x, s) - heat_modes(x, s)) / heat_modes(0.0, s)
          for x in (0.0, 0.13, 0.5) for s in (0.003, 0.05, 1.0)) < 1e-12,
      "heat on a ring: sum of images = sum of modes (Poisson duality), short and long times")
HEXB = np.array([[1.0, 0.0], [0.5, math.sqrt(3) / 2]])
check(max(abs(heat2_images(np.array(x), s, HEXB) - heat2_modes(np.array(x), s, HEXB)) / heat2_modes(np.zeros(2), s, HEXB)
          for x in ((0.0, 0.0), (0.31, 0.2), (0.5, 0.29)) for s in (0.01, 0.05, 0.3)) < 1e-12,
      "heat on the hexagonal torus (V = sqrt3/2): images = (1/V) sum over the dual lattice")

SQUARE = np.array([[1.0, 0.0], [0.0, 1.0]])
HEX = np.array([[1.0, 0.0], [0.5, math.sqrt(3) / 2]])


def lattice_points(basis, m=7):
    return np.array([i * basis[0] + j * basis[1] for i in range(-m, m + 1) for j in range(-m, m + 1)])


def burgers(X, s, t, pts):
    """u = -2 nu grad log phi, phi = sum_n exp(-|x - n|^2 / 4s), s = nu t: u = (x - <n>)/t."""
    d2 = ((X[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    w = np.exp(-(d2 - d2.min(1, keepdims=True)) / (4 * s))
    w /= w.sum(1, keepdims=True)
    return (X - w @ pts) / t


# residual of Burgers' equation by finite differences, nu = 0.05, at t = 0.4 (s = 0.02)
nu, t0, h = 0.05, 0.4, 1e-4
pts_sq = lattice_points(SQUARE)
X0 = RNG.uniform(0, 1, (40, 2))
def U(X, t): return burgers(X, nu * t, t, pts_sq)
ut = (U(X0, t0 + h) - U(X0, t0 - h)) / (2 * h)
ux = (U(X0 + [h, 0], t0) - U(X0 - [h, 0], t0)) / (2 * h)
uy = (U(X0 + [0, h], t0) - U(X0 - [0, h], t0)) / (2 * h)
lap = (U(X0 + [h, 0], t0) + U(X0 - [h, 0], t0) + U(X0 + [0, h], t0) + U(X0 - [0, h], t0) - 4 * U(X0, t0)) / h ** 2
u0 = U(X0, t0)
res = ut + u0[:, :1] * ux + u0[:, 1:] * uy - nu * lap
check(np.abs(res).max() / np.abs(ut).max() < 1e-4,
      f"Cole-Hopf: u = (x - <n>)/t solves Burgers' equation (relative residual {np.abs(res).max() / np.abs(ut).max():.0e})")


def nearest(X, pts):
    d2 = ((X[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    srt = np.sort(d2, 1)
    return pts[d2.argmin(1)], srt[:, 1] - srt[:, 0]


for name, basis in (("square", SQUARE), ("hexagonal", HEX)):
    pts = lattice_points(basis)
    X = RNG.uniform(-0.5, 0.5, (4000, 2)) @ basis
    n_star, gap = nearest(X, pts)
    away = gap > 0.05
    err = np.abs(burgers(X, 5e-4, 1.0, pts) - (X - n_star))[away].max()
    check(err < 1e-9, f"{name}: for nu t = 5e-4, u t = x - (nearest lattice point) off the cell walls "
          f"(error {err:.0e}): each cell is a Hubble flow u = r/t from its own explosion")
# long times: the slowest tone. With nu = 1, compare t = 0.3 and t = 0.4 (12 rings of explosions around the cell).
for name, basis, kmin2, nk in (("square", SQUARE, 1.0, 4), ("hexagonal", HEX, 4 / 3, 6)):
    pts = lattice_points(basis, 12)
    X = RNG.uniform(0, 1, (3000, 2)) @ basis
    amp = [np.abs(burgers(X, t, t, pts)).max() for t in (0.3, 0.4)]
    rate = math.log(amp[0] / amp[1]) / 0.1
    dual = np.linalg.inv(basis).T
    ks = np.array([i * dual[0] + j * dual[1] for i in range(-3, 4) for j in range(-3, 4)])
    k2 = np.sort(np.round((ks ** 2).sum(1), 9))
    check(abs(k2[1] - kmin2) < 1e-9 and np.sum(np.abs(k2 - kmin2) < 1e-9) == nk
          and abs(rate / (4 * math.pi ** 2 * kmin2) - 1) < 1e-4,
          f"{name}: at long times u decays at the rate 4 pi^2 nu |k|^2 = {rate:.4f} nu of the {nk} shortest dual "
          f"vectors, |k|^2 = {kmin2:.4f}")

# figure 2: Burgers flow from a lattice of explosions
fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.0))
for row, (name, basis) in enumerate((("square", SQUARE), ("hexagonal", HEX))):
    pts = lattice_points(basis, 5)
    gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 241), np.linspace(-1.2, 1.2, 241))
    X = np.c_[gx.ravel(), gy.ravel()]
    for col, s in enumerate((2e-3, 0.06)):
        u = burgers(X, s, 1.0, pts)
        sp = np.hypot(u[:, 0], u[:, 1]).reshape(gx.shape)
        ax = axes[row, col]
        ax.imshow(sp, extent=(-1.2, 1.2, -1.2, 1.2), origin="lower", cmap="Blues")
        sl = (slice(None, None, 20), slice(None, None, 20))
        ax.quiver(gx[sl], gy[sl], u[:, 0].reshape(gx.shape)[sl], u[:, 1].reshape(gx.shape)[sl], color=INK,
                  width=0.004, scale=None if col == 0 else None)
        ax.plot(pts[:, 0], pts[:, 1], "o", color=ORANGE, ms=3.5)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}, ν t = {s:g}" + ("  (Voronoi cells)" if col == 0 else "  (slowest tones)"),
                     fontsize=8.5)
fig.suptitle("Burgers flow from a lattice of explosions (orange): colour is the speed, darker is faster", fontsize=9, y=0.99)
save(fig, "motion-burgers")


# ---------------------------------------------------------------- (3) swirl
def shell(N, d):
    r = int(math.isqrt(N)) + 1
    return [k for k in itertools.product(range(-r, r + 1), repeat=d) if sum(x * x for x in k) == N]


def half(ks):
    out = []
    for k in ks:
        if tuple(-x for x in k) not in out:
            out.append(k)
    return out


n2 = 64
k1 = np.fft.fftfreq(n2, 1 / n2)
KX, KY = np.meshgrid(k1, k1, indexing="ij")
xs = np.arange(n2) / n2
XX, YY = np.meshgrid(xs, xs, indexing="ij")


def d2(f, K):
    return np.real(np.fft.ifft2(2j * math.pi * K * np.fft.fft2(f)))


def stream(Ns):
    psi = np.zeros((n2, n2))
    for N in Ns:
        for k in half(shell(N, 2)):
            c = RNG.normal() + 1j * RNG.normal()
            psi += 2 * np.real(c * np.exp(2j * math.pi * (k[0] * XX + k[1] * YY)))
    return psi


def jacobian_of_vorticity(psi):
    om = np.real(np.fft.ifft2(-(2 * math.pi) ** 2 * (KX ** 2 + KY ** 2) * np.fft.fft2(psi)))
    return d2(psi, KX) * d2(om, KY) - d2(psi, KY) * d2(om, KX), om


ok = True
for N in (1, 2, 5, 25, 65):
    J, om = jacobian_of_vorticity(stream([N]))
    ok &= np.abs(J).max() < 1e-9 * np.abs(om).max() ** 2
J, om = jacobian_of_vorticity(stream([1, 2]))
check(ok and np.abs(J).max() > 1e-2 * np.abs(om).max() ** 2,
      "2D: a stream function on one circle |k|^2 = N (N = 1, 2, 5, 25, 65) gives u . grad omega = 0, a steady Euler "
      "flow; mixing N = 1 and 2 does not")
chi4 = lambda m: 0 if m % 2 == 0 else (1 if m % 4 == 1 else -1)
r2 = [len(shell(N, 2)) for N in range(1, 51)]
def odd_power_3mod4(N):
    for p in range(3, N + 1, 4):
        if all(p % q for q in range(2, int(p ** 0.5) + 1)):
            e = 0
            while N % p == 0:
                N //= p
                e += 1
            if e % 2:
                return True
    return False


check(all(r2[N - 1] == 4 * sum(chi4(dd) for dd in range(1, N + 1) if N % dd == 0) for N in range(1, 51))
      and all((r2[N - 1] == 0) == odd_power_3mod4(N) for N in range(1, 51)),
      "the cellular flows at |k|^2 = N form a space of dimension r_2(N) = 4 sum_(d | N) chi_-4(d), zero exactly when a "
      "prime 3 mod 4 divides N to an odd power (N <= 50): "
      + ", ".join(str(N) for N in range(1, 31) if r2[N - 1] == 0) + ", ...")

n3 = 16
k3 = np.fft.fftfreq(n3, 1 / n3)
K3 = np.array(np.meshgrid(k3, k3, k3, indexing="ij"))
x3 = np.array(np.meshgrid(*(np.arange(n3) / n3,) * 3, indexing="ij"))


def grad3(f):
    F = np.fft.fftn(f)
    return np.array([np.real(np.fft.ifftn(2j * math.pi * K3[i] * F)) for i in range(3)])


def curl3(u):
    g = [grad3(u[i]) for i in range(3)]
    return np.array([g[2][1] - g[1][2], g[0][2] - g[2][0], g[1][0] - g[0][1]])


def beltrami(Ns):
    u = np.zeros((3,) + x3.shape[1:])
    for N in Ns:
        for k in half(shell(N, 3)):
            kv = np.array(k, float)
            kh = kv / np.linalg.norm(kv)
            e1 = np.cross(kh, [1.0, 0, 0] if abs(kh[0]) < 0.9 else [0, 1.0, 0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(kh, e1)
            a = e1 + 1j * e2                                     # i k^ x a = a: curl eigenvalue +2 pi |k|
            c = RNG.normal() + 1j * RNG.normal()
            phase = np.exp(2j * math.pi * np.tensordot(kv, x3, axes=1))
            u += 2 * np.real(c * a[:, None, None, None] * phase)
    return u


def nonlinear(u):
    return np.array([sum(u[j] * grad3(u[i])[j] for j in range(3)) for i in range(3)])


ok = True
for N in (1, 2, 3, 5, 6):
    u = beltrami([N])
    div = sum(grad3(u[i])[i] for i in range(3))
    scale = np.abs(u).max()
    ok &= np.abs(div).max() < 1e-10 * scale and np.abs(curl3(u) - 2 * math.pi * math.sqrt(N) * u).max() < 1e-9 * scale
    ok &= np.abs(curl3(nonlinear(u))).max() < 1e-8 * scale ** 2
u = beltrami([1, 2])
check(ok and np.abs(curl3(nonlinear(u))).max() > 1e-2 * np.abs(u).max() ** 2,
      "3D Beltrami fields on |k|^2 = N (N = 1, 2, 3, 5, 6): div u = 0, curl u = 2 pi sqrt N u, and (u . grad) u is "
      "a gradient, so u is a steady Euler flow; mixing N = 1 and 2 fails")
A_, B_, C_ = 1.0, 0.7, 0.4
X_, Y_, Z_ = x3 * 2 * math.pi
abc = np.array([A_ * np.sin(Z_) + C_ * np.cos(Y_), B_ * np.sin(X_) + A_ * np.cos(Z_), C_ * np.sin(Y_) + B_ * np.cos(X_)])
check(np.abs(curl3(abc) - 2 * math.pi * abc).max() < 1e-10 and len(shell(1, 3)) == 6,
      "the ABC flow is the case N = 1: curl u = 2 pi u, with r_3(1) = 6 real parameters (three amplitudes, and three phases that a translation removes)")


def kron(D, n):
    """Kronecker symbol (D / n) for n >= 1."""
    r = 1
    while n % 2 == 0:
        n //= 2
        if D % 2 == 0:
            return 0
        r *= 1 if D % 8 in (1, 7) else -1
    a, m = D % n, n
    while a:
        while a % 2 == 0:
            a //= 2
            if m % 8 in (3, 5):
                r = -r
        a, m = m, a
        if a % 4 == 3 and m % 4 == 3:
            r = -r
        a %= m
    return r if m == 1 else 0


def L1(D):
    """L(1, chi_D) for a fundamental discriminant D < 0: -(pi / |D|^(3/2)) sum_a a chi_D(a) (also D = -3, -4)."""
    return -math.pi / abs(D) ** 1.5 * sum(a * kron(D, a) for a in range(1, abs(D)))


r3 = {N: len(shell(N, 3)) for N in range(1, 51)}
ok, tested = True, 0
for N in range(1, 51):
    if any(N % (p * p) == 0 for p in range(2, 8)) or N % 8 == 7:
        continue
    D = -4 * N if N % 4 in (1, 2) else -N
    ok &= abs(r3[N] - 24 / math.pi * math.sqrt(N) * L1(D)) < 1e-9
    tested += 1
legendre = lambda N: any(N % (4 ** a) == 0 and (N // 4 ** a) % 8 == 7 for a in range(4))
check(ok and all((r3[N] == 0) == legendre(N) for N in range(1, 51)),
      f"the Beltrami flows at |k|^2 = N form a space of dimension r_3(N) = (24/pi) sqrt N L(1, chi_D) "
      f"({tested} squarefree N <= 50, N not 7 mod 8, from N = 1); zero exactly at N = 4^a(8b + 7) (N <= 50)")

# figure 3: two cellular flows
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"wspace": 0.3})
for ax, N in zip(axes, (2, 5)):
    psi = stream([N])
    ax.contour(xs, xs, psi.T, levels=14, colors=BLUE, linewidths=0.9)
    ax.contour(xs, xs, psi.T, levels=[0], colors=ORANGE, linewidths=1.4)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title(f"|k|² = {N}, a family of dimension r₂({N}) = {len(shell(N, 2))}", fontsize=8.5)
for ax in axes:
    for sp in ax.spines.values():
        sp.set_color(GRID)
save(fig, "motion-swirl")
