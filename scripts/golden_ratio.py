"""Note 05: the golden ratio in the modular world and in physics.

What it computes
  (1) Continued fractions. phi = [1; 1, 1, 1, ...]: q |q phi - p| -> 1/sqrt 5 at the convergents, the worst possible
      (Hurwitz). log2(3/2) = [0; 1, 1, 2, 2, 3, 1, 5, 2, 23, ...] is much better approximated: 0.235 at q = 12.
  (2) Phyllotaxis. Points at angle n * alpha and radius sqrt n (Vogel's model) for alpha = 7/12, log2(3/2) and
      1/phi^2 of a turn. 7/12 gives exactly 12 rays; the golden angle 137.508 degrees spreads the points most
      evenly (largest-to-smallest gap ratio, by the three-gap theorem, stays the smallest of the three).
  (3) The shortest closed geodesic on the modular surface. The smallest |trace| of a hyperbolic element of
      SL(2, Z) is 3, attained by [[2, 1], [1, 1]], whose axis joins -1/phi to phi; its length is
      2 arccosh(3/2) = 4 log phi. The geodesic is reciprocal: its image is a loop through i of length 2 log phi,
      traversed in both directions. The next lengths, 2 log of the units (t + sqrt(t^2 - 4))/2, are listed.
  (4) Level 5 and the icosahedron. PSL(2, Z/5) has order 60 and the element orders of A5 (1, 15, 20, 24 of
      orders 1, 2, 3, 5). The modular curve X(5) has 12 cusps, 30 images of i and 20 of rho: 12 - 30 + 20 = 2,
      a sphere tiled as the icosahedron, whose 12 vertices are the cyclic permutations of (0, +-1, +-phi).
  (5) The E8 masses. The Perron-Frobenius eigenvector of the E8 Dynkin diagram (eigenvalue 2 cos(pi/30))
      gives Zamolodchikov's eight masses of the Ising field theory in a magnetic field; m2/m1 = phi.

What it shows
  The golden ratio is the number least well approximated by fractions, the opposite extreme from
  log2(3/2) ~ 7/12. It appears in the modular world (the shortest closed geodesic, level 5) and in physics where
  five-fold or E8 structure is present (phyllotaxis; quasicrystals; the Ising chain in a field, Coldea et al.,
  Science 327 (2010) 177).
"""
import itertools
import math

import matplotlib.pyplot as plt
import numpy as np

from common import BLUE, GRID, INK, INK_2, ORANGE, check, save

PHI = (1 + math.sqrt(5)) / 2
L32 = math.log2(1.5)


def cf(x, n):
    out = []
    for _ in range(n):
        a = math.floor(x)
        out.append(a)
        x = 1 / (x - a)
    return out


def convergents(a):
    p0, q0, p1, q1 = 1, 0, a[0], 1
    yield p1, q1
    for ai in a[1:]:
        p0, q0, p1, q1 = p1, q1, ai * p1 + p0, ai * q1 + q0
        yield p1, q1


# ---------------------------------------------------------------------------------------------------
print("(1) the most irrational number, and the fifth")
check(cf(PHI, 15) == [1] * 15, "phi = [1; 1, 1, 1, ...]")
hp = [q * abs(q * PHI - p) for p, q in convergents([1] * 25)][3:]
check(all(v > 0.38 for v in hp) and abs(hp[-1] - 1 / math.sqrt(5)) < 1e-6,
      f"phi: q |q phi - p| at the convergents tends to 1/sqrt 5 = {1 / math.sqrt(5):.4f} (Hurwitz's worst case)")
import mpmath as mp
mp.mp.dps = 40
xq, cf_fifth = mp.log(1.5, 2), []
for _ in range(10):
    aq = int(mp.floor(xq))
    cf_fifth.append(aq)
    xq = 1 / (xq - aq)
check(cf_fifth == [0, 1, 1, 2, 2, 3, 1, 5, 2, 23], "log2(3/2) = [0; 1, 1, 2, 2, 3, 1, 5, 2, 23, ...]")
hl = {q: q * abs(q * L32 - p) for p, q in convergents([0, 1, 1, 2, 2, 3, 1, 5, 2])}
print(f"     log2(3/2): q |q x - p| = {', '.join(f'{v:.3f} (q = {q})' for q, v in hl.items() if q > 1)}")
check(abs(hl[12] - 0.2346) < 1e-3 and hl[12] < 1 / math.sqrt(5), "the fifth: 0.235 at q = 12, far better than phi")

# ---------------------------------------------------------------------------------------------------
print("(2) phyllotaxis")
golden_angle = 360 / PHI ** 2
check(abs(golden_angle - 137.5078) < 1e-4, f"golden angle 360/phi^2 = {golden_angle:.4f} degrees")
angles = {"7/12 of a turn (the tempered fifth)": 7 / 12, "log₂(3/2) of a turn (the pure fifth)": L32,
          "1/φ² of a turn (the golden angle)": 1 / PHI ** 2}


def gap_ratio(alpha, n):
    pts = np.sort((np.arange(n) * alpha) % 1.0)
    gaps = np.diff(np.concatenate([pts, [pts[0] + 1]]))
    return gaps.max() / gaps.min()


worst = {k: max(gap_ratio(a, n) for n in range(20, 801)) for k, a in angles.items() if k[0] != "7"}
for k, v in worst.items():
    print(f"     {k}: largest/smallest gap among the first n points, worst over 20 <= n <= 800: {v:.3f}")
check(worst["1/φ² of a turn (the golden angle)"] < worst["log₂(3/2) of a turn (the pure fifth)"],
      "the golden angle spreads points more evenly than the fifth")
check(len({round((n * 7) % 12) for n in range(1, 801)}) == 12, "7/12 of a turn: exactly 12 rays")

# ---------------------------------------------------------------------------------------------------
print("(3) the shortest closed geodesic on the modular surface")
traces = set()
R = range(-12, 13)
for a, b, c in itertools.product(R, R, R):
    if a != 0 and (1 + b * c) % a == 0:
        d = (1 + b * c) // a
        if abs(a + d) > 2:
            traces.add(abs(a + d))
check(min(traces) == 3, "the smallest |trace| of a hyperbolic element is 3 (entries up to 12 searched)")
check(abs(2 * math.acosh(1.5) - 4 * math.log(PHI)) < 1e-12, f"its length 2 arccosh(3/2) = 4 log phi = {4 * math.log(PHI):.6f}")
check(abs(PHI ** 2 - PHI - 1) < 1e-12 and abs((-1 / PHI) ** 2 + 1 / PHI - 1) < 1e-12,
      "the axis of [[2, 1], [1, 1]] joins its fixed points -1/phi and phi (roots of z^2 - z - 1)")
Sm, gm = np.array([[0, -1], [1, 0]]), np.array([[2, 1], [1, 1]])
check((Sm @ gm @ np.linalg.inv(Sm)).round().astype(int).tolist() == [[1, -1], [-1, 2]],
      "S gamma S^-1 = gamma^-1: the geodesic is reciprocal and passes through the elliptic point i")
wall = complex(0.5, math.sqrt(5) / 2)
arc = math.acosh(1 + abs(1j - wall) ** 2 / (2 * 1 * wall.imag))
check(abs(arc - math.log(PHI)) < 1e-12 and abs(abs(1j - 0.5) - math.sqrt(5) / 2) < 1e-12,
      "the axis passes through i; from i to the wall point 1/2 + i sqrt(5)/2 is log phi, so the image is a loop "
      "through i of length 2 log phi, traversed in both directions")
for t in range(3, 9):
    eps = (t + math.sqrt(t * t - 4)) / 2
    print(f"     trace {t}: length {2 * math.acosh(t / 2):.4f} = 2 log {eps:.4f}, unit of Q(sqrt {t * t - 4})")

# ---------------------------------------------------------------------------------------------------
print("(4) level 5 and the icosahedron")
mats = [m for m in itertools.product(range(5), repeat=4) if (m[0] * m[3] - m[1] * m[2]) % 5 == 1]
check(len(mats) == 120, "SL(2, Z/5) has 120 elements, so PSL(2, Z/5) has 60")


def mul(x, y):
    return ((x[0] * y[0] + x[1] * y[2]) % 5, (x[0] * y[1] + x[1] * y[3]) % 5,
            (x[2] * y[0] + x[3] * y[2]) % 5, (x[2] * y[1] + x[3] * y[3]) % 5)


def proj_order(m):
    k, x = 1, m
    while x not in ((1, 0, 0, 1), (4, 0, 0, 4)):
        x = mul(x, m)
        k += 1
    return k


orders = {}
for m in mats:
    if m <= tuple((-v) % 5 for v in m):
        o = proj_order(m)
        orders[o] = orders.get(o, 0) + 1
check(orders == {1: 1, 2: 15, 3: 20, 5: 24}, f"element orders in PSL(2, Z/5): {dict(sorted(orders.items()))}, those of A5")
cusps = {min((a, c), ((-a) % 5, (-c) % 5)) for a in range(5) for c in range(5) if (a, c) != (0, 0)}
check(len(cusps) == 12 and 12 - 60 // 2 + 60 // 3 == 2,
      "X(5): 12 cusps, 30 images of i, 20 of rho; 12 - 30 + 20 = 2 (a sphere)")
V = []
for s1, s2 in itertools.product((1, -1), repeat=2):
    for perm in ((0, s1, s2 * PHI), (s1, s2 * PHI, 0), (s2 * PHI, 0, s1)):
        V.append(np.array(perm))
D = np.array([[np.linalg.norm(u - v) for v in V] for u in V])
edges = [(i, j) for i in range(12) for j in range(i + 1, 12) if abs(D[i, j] - 2) < 1e-9]
check(len(V) == 12 and len(edges) == 30 and all(sum(abs(D[i] - 2) < 1e-9) == 5 for i in range(12)),
      "(0, +-1, +-phi): 12 vertices, 30 edges of length 2, five at each vertex: the icosahedron")

# ---------------------------------------------------------------------------------------------------
print("(5) the E8 masses")
A = np.zeros((8, 8))
for i, j in [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 3)]:   # Bourbaki: 1-3-4-5-6-7-8, 2-4
    A[i, j] = A[j, i] = 1
w, v = np.linalg.eigh(A)
pf = np.abs(v[:, -1])
m = np.sort(pf / pf.min())
check(abs(w[-1] - 2 * math.cos(math.pi / 30)) < 1e-12, "Perron-Frobenius eigenvalue 2 cos(pi/30) (Coxeter number 30)")
c = lambda x: 2 * math.cos(x)
zam = [1, c(math.pi / 5), c(math.pi / 30), c(math.pi / 5) * c(7 * math.pi / 30), c(math.pi / 5) * c(2 * math.pi / 15),
       c(math.pi / 5) * c(math.pi / 30), c(math.pi / 5) ** 2 * c(7 * math.pi / 30),
       c(math.pi / 5) ** 2 * c(2 * math.pi / 15)]
check(np.allclose(m, sorted(zam), atol=1e-12), "the eight masses agree with Zamolodchikov's closed forms")
check(abs(m[1] - PHI) < 1e-12, f"m2/m1 = 2 cos(pi/5) = phi; masses {', '.join(f'{x:.3f}' for x in m)}")

# ---------------------------------------------------------------------------------------------------
print("figures")

# (a) phyllotaxis
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
n = np.arange(1, 701)
for ax, (lab, a) in zip(axes, angles.items()):
    th = 2 * np.pi * a * n
    ax.scatter(np.sqrt(n) * np.cos(th), np.sqrt(n) * np.sin(th), s=5, color=BLUE if "φ" not in lab else ORANGE,
               lw=0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(lab, fontsize=8.5)
fig.suptitle("Seeds placed at a fixed angle per step: the fifth makes twelve rays, the golden angle fills the disc",
             fontsize=10)
save(fig, "golden-phyllotaxis")


# (b) the shortest closed geodesic on the modular surface
def reduce(z):
    for _ in range(200):
        z = complex(z.real - math.floor(z.real + 0.5), z.imag)
        if abs(z) < 1 - 1e-12:
            z = -1 / z
        else:
            return z
    return z


fig, ax = plt.subplots(figsize=(6.6, 3.4))
top = 2.2
edge = ([complex(-0.5, y) for y in np.linspace(top, math.sqrt(3) / 2, 60)]
        + [np.exp(1j * th) for th in np.linspace(2 * math.pi / 3, math.pi / 3, 100)]
        + [complex(0.5, y) for y in np.linspace(math.sqrt(3) / 2, top, 60)])
fx, fy = [z.real for z in edge], [z.imag for z in edge]
ax.fill(fx + [0.5, -0.5], fy + [top, top], color="#cde2fb", lw=0, zorder=1)
ax.plot(fx, fy, color=BLUE, lw=1.5, zorder=2)
for k in range(-2, 3):
    th = np.linspace(0, np.pi, 200)
    ax.plot(k + np.cos(th), np.sin(th), color=GRID, lw=0.8, zorder=0)
    ax.axvline(k + 0.5, color=GRID, lw=0.8, zorder=0)
centre, rad = 0.5, math.sqrt(5) / 2
th = np.linspace(0, np.pi, 400)
ax.plot(centre + rad * np.cos(th), rad * np.sin(th), color=ORANGE, lw=1.2, ls=(0, (4, 3)), zorder=2)
# fold one period of the axis into the fundamental domain
z0 = complex(centre, rad)
g = np.array([[2, 1], [1, 1]])
pts = []
for s_ in np.linspace(0, 4 * math.log(PHI), 3000):
    # point at hyperbolic distance s_ from z0 along the axis, via the conjugation to the imaginary axis
    w = 1j * math.exp(s_)
    lo, hi = -1 / PHI, PHI
    z = (hi * w + lo) / (w + 1)
    pts.append(reduce(z))
pts = np.array(pts)
jumps = np.where(np.abs(np.diff(pts)) > 0.2)[0]
for seg in np.split(np.arange(len(pts)), jumps + 1):
    ax.plot(pts[seg].real, pts[seg].imag, color=ORANGE, lw=2.2, zorder=3)
ax.plot([-1 / PHI, PHI], [0, 0], "o", ms=7, color=ORANGE, mec="white", mew=1.5, zorder=4, clip_on=False)
ax.text(PHI + 0.08, 0.07, "φ", ha="left", fontsize=10, color=INK)
ax.text(-1 / PHI - 0.08, 0.07, "−1/φ", ha="right", fontsize=10, color=INK)
ax.text(1.75, 1.45, "the axis of [[2, 1], [1, 1]]\n(dashed), folded into the\nfundamental domain (solid):\n"
        "a closed geodesic of\nlength 4 log φ = 1.925", fontsize=8.5, color=INK, va="top")
ax.set_xlim(-1.2, 2.9)
ax.set_ylim(0, top)
ax.set_aspect("equal")
ax.set_yticks([0, 1, 2])
ax.axhline(0, color=INK_2, lw=0.8)
ax.set_title("The shortest closed geodesic on the modular surface belongs to the golden ratio", fontsize=10)
save(fig, "golden-geodesic")

# (c) the icosahedron and the E8 masses
fig = plt.figure(figsize=(7.4, 3.3))
ax = fig.add_subplot(1, 2, 1, projection="3d")
for i, j in edges:
    ax.plot(*zip(V[i], V[j]), color=INK_2, lw=0.9)
for rect in range(3):
    corners = [p for p in V if abs(p[rect]) < 1e-9]
    corners = sorted(corners, key=lambda p: math.atan2(p[(rect + 2) % 3], p[(rect + 1) % 3]))
    xs, ys, zs = zip(*(corners + corners[:1]))
    ax.plot(xs, ys, zs, color=BLUE, lw=1.4)
Vx = np.array(V)
ax.scatter(Vx[:, 0], Vx[:, 1], Vx[:, 2], color=ORANGE, s=30, depthshade=False)
ax.set_box_aspect((1, 1, 1))
ax.axis("off")
ax.set_title("X(5): an icosahedron, its 12 vertices\nthe cusps; golden rectangles 1 × φ", fontsize=9)
ax = fig.add_subplot(1, 2, 2)
for k, x in enumerate(m):
    ax.plot([0, 1], [x, x], color=BLUE if k > 1 else ORANGE, lw=2)
    ax.text(1.06, x, f"m{k + 1} = {x:.3f}", va="center", fontsize=8.5, color=INK)
ax.annotate("", xy=(0.5, m[1]), xytext=(0.5, m[0]), arrowprops=dict(arrowstyle="<->", color=INK_2, lw=1))
ax.text(0.15, (m[0] + m[1]) / 2, "×φ", fontsize=10, color=INK, va="center")
ax.set_xlim(0, 1.8)
ax.set_ylim(0.5, 5.2)
ax.axis("off")
ax.set_title("E8 masses of the Ising chain in a field\n(Zamolodchikov; seen by Coldea et al. 2010)", fontsize=9)
save(fig, "golden-icosahedron-e8")
print("done")
