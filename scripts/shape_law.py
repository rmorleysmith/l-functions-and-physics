"""Note 11: the law of shape on 201 shells.

What it computes
  For every squarefree N <= 400 with N not 7 (mod 8): the hexadecapole a(N) of the N-th shell of Z^3 (the
  coefficient of theta^3 psi, normalised by a(1) = 1) and the central value L(5, F x chi_D), with F the newform
  of weight 10 and level 2 and D the discriminant of Q(sqrt(-N)). The L-values come from the functional equation
  (root number +1, conductor 2D^2 for odd D and D^2 for even D) with 26 N_max terms.

What it shows
  a(N)^2 / N^(9/2) = L(5, F x chi_D) / L(5, F x chi_-4) with one constant for all N, in each of the seven classes
  of N in Q_2^x modulo squares that occur (N = 1, 3, 5 mod 8 and N = 2, 6, 10, 14 mod 16), to 1e-10.

Status
  Waldspurger's theorem gives the proportionality within each class of N; the equality of the constant across
  the seven classes is the computation made here.
"""
import math

import numpy as np

from common import check

NMAX = 400
MF = 26 * NMAX + 50
def sparse_mul(dense, sparse, m):
    out = [0] * (m + 1)
    for e, c in sparse:
        for i in range(m + 1 - e):
            if dense[i]: out[i + e] += c * dense[i]
    return out
pent = []
k = 0
while k * (3 * k - 1) // 2 <= MF:
    pent.append((k * (3 * k - 1) // 2, (-1) ** k))
    if k and k * (3 * k + 1) // 2 <= MF: pent.append((k * (3 * k + 1) // 2, (-1) ** k))
    k += 1
jac = [(k * (k + 1) // 2, (-1) ** k * (2 * k + 1)) for k in range(200) if k * (k + 1) // 2 <= MF]
def scale(sp, s): return [(e * s, c) for e, c in sp if e * s <= MF]
f = [1] + [0] * MF
for sp in (jac, jac, pent, pent, scale(jac, 2), scale(jac, 2), scale(pent, 2), scale(pent, 2)):
    f = sparse_mul(f, sp, MF)                      # prod (1-q^n)^8 (1-q^2n)^8 ; F = q * f * E
sig = [0] * (MF + 1)
for d in range(1, MF + 1):
    for n in range(d, MF + 1, d): sig[n] += d
E2 = [1] + [-24 * sig[n] for n in range(1, MF + 1)]
E = [2 * (E2[n // 2] if n % 2 == 0 else 0) - E2[n] for n in range(MF + 1)]
F = np.convolve(np.array(f, dtype=object), np.array(E, dtype=object))[:MF]
assert list(F[:7]) == [1, 16, -156, 256, 870, -2496, -952], F[:7]
Ff = np.array([float(v) for v in F]); nF = np.arange(1, MF + 1, dtype=float)
def kron(D, n):
    r = 1
    if n % 2 == 0:
        if D % 2 == 0: return 0
        s = 1 if D % 8 in (1, 7) else -1
        while n % 2 == 0: n //= 2; r *= s
    # Jacobi (D/n), n odd positive
    a = D % n; m = n
    while a:
        while a % 2 == 0:
            a //= 2
            if m % 8 in (3, 5): r = -r
        a, m = m, a
        if a % 4 == 3 and m % 4 == 3: r = -r
        a %= m
    return r if m == 1 else 0
def Lc(D):
    cond = 2 * D * D if D % 2 else D * D
    y = 2 * math.pi * nF / math.sqrt(cond)
    chi = np.array([kron(D, n) for n in range(1, MF + 1)], dtype=float)
    tail = np.exp(-y) * (1 + y + y**2 / 2 + y**3 / 6 + y**4 / 24)
    return 2 * float((Ff * chi * tail / nF**5).sum())
# a(N): theta^3 * psi
r1 = [0] * (NMAX + 1)
for x in range(-20, 21):
    if x * x <= NMAX: r1[x * x] += 1
def mul(a, b): 
    return [sum(a[i] * b[n - i] for i in range(n + 1)) for n in range(NMAX + 1)]
r3 = mul(mul(r1, r1), r1)
psi = [0] + [(-1) ** (n - 1) * sum(d**3 for d in range(1, n + 1) if n % d == 0 and (n // d) % 2 == 1) for n in range(1, NMAX + 1)]
a = mul(r3, psi)
def sqfree(n): return all(n % (p * p) for p in range(2, int(n**.5) + 1))
L4 = Lc(-4)
worst, count = {}, {}
for N in range(1, NMAX + 1):
    if not sqfree(N) or N % 8 == 7:
        continue
    D = -4 * N if N % 4 in (1, 2) else -N
    rhs = Lc(D) / L4
    cls = N % 16 if N % 2 == 0 else N % 8
    worst[cls] = max(worst.get(cls, 0.0), abs(a[N] ** 2 / N ** 4.5 - rhs) / rhs)
    count[cls] = count.get(cls, 0) + 1
check(abs(1 / L4 - 0.377316968934) < 1e-11, f"1/L(5, F x chi_-4) = {1 / L4:.12f}")
check(sorted(count) == [1, 2, 3, 5, 6, 10, 14] and sum(count.values()) == 201,
      "201 squarefree N <= 400 not 7 mod 8, in seven classes: "
      + ", ".join(f"{c} mod {16 if c % 2 == 0 else 8}: {count[c]}" for c in sorted(count)))
check(max(worst.values()) < 1e-10,
      f"law of shape: a(N)^2 / N^(9/2) = L(5, F x chi_D)/L(5, F x chi_-4) on all 201 shells "
      f"(largest relative error {max(worst.values()):.0e})")
