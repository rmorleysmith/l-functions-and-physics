# Reusable derivations

The derivations the notes rely on, each with its statement, its reading in physical terms, and where it is
proved or checked. Numbers `n.k` match the **Derivation n.k** boxes in note `n`; the others point at a theorem or
proposition in the note. Every script asserts the numbers it prints (`make check`).

## Impulses, packets and spectra (note 07, [`scripts/impulses_packets.py`](scripts/impulses_packets.py))

| | Statement | In physical terms |
|---|---|---|
| 7.1 | Poisson summation: `Σ f(x + n) = Σ f̂(k) e^{2πikx}`; for the heat kernel, `Σ (4πt)^{−1/2} e^{−(x−n)²/4t} = Σ e^{−4π²k²t} e^{2πikx}`; for the Gaussian at `x = 0`, `θ(1/t) = √t θ(t)`. | A comb of impulses that diffuses for a time `t` is a sum of waves, the `k`-th damped by `e^{−4π²k²t}`. |
| 7.2 | The character table mod 12 is a Hadamard matrix (`HHᵀ = 4I`), so `ψ(x;12,a) = ¼ Σ_χ χ(a) ψ(x,χ)` and `Σ_a ψ(x;12,a)² = ¼ Σ_χ ψ(x,χ)²`. | A prism: four orthogonal channels, energy conserved. |
| 7.3 | `−L′/L(s,χ) = Σ χ(n)Λ(n)n^{−s} = ∫ e^{−su} dψ_χ(e^u)`. | The L-function is the transfer function (Laplace transform) of the impulse train. |
| 7.4 | Explicit formula: `ψ(x) = x − Σ_ρ x^ρ/ρ − log 2π − ½ log(1 − x^{−2})`, and its form for `ψ(x,χ)`. | Each zero is a normal mode; the impulse train is the sum of its modes. |
| 7.5 | Landau: `Σ_{0<γ≤T} x^ρ = −(T/2π) Λ(x) + O(log T)`; for zeros on the critical line, `Σ x^{iγ} = −(T/2π) Λ(x)/√x + O(log T)`. | The zeros, played together, ring only at the logarithms of the prime powers: primes and zeros are a Fourier pair. |

## Light, cones and spheres (note 08, [`scripts/light_cone.py`](scripts/light_cone.py))

| | Statement | In physical terms |
|---|---|---|
| 8.1 | Modes of light in a periodic box: `2#{n ∈ ℤ³ : \|n\|² ≤ N} ~ Vω³/(3π²c³)`; multiplicity `2r₃(N)` at `ω = (2πc/L)√N`. | `c` converts counting lattice points into counting frequencies (Planck's `c³`). |
| 8.2 | Gauss: `r₃(n) = 12h(−4n)`, `24h(−n)` or `0` (squarefree `n > 3`); so `r₃(n) = (24√n/π) L(1, χ₋₄ₙ)` for `n ≡ 1, 2 mod 4`. | The number of ways light fits a cubic box at pitch `√n` is an L-value. |
| 8.3 | `(T, X, Y) = (a + c, b, a − c)` makes `b² − 4ac = X² + Y² − T²`; `SL(2, ℤ)` acts by integral Lorentz transformations. | Gauss's forms are integer vectors in 2+1 spacetime: definite = particle, degenerate = light, indefinite = geodesic. |
| 8.4 | Relative rapidity `cosh η = −B(f, g)/√(D_f D_g)`; for `x² + y²` and `x² + xy + y²`, `tanh η = ½`. In the rest frame of `x² + y²`, the form `x² + bxy + cy²` has Lorentz factor `(1 + c)/√\|D\|`. | The Gaussian and Eisenstein forms move relative to each other at `c/2`; the nine class-number-one forms have speeds `0, ½, ⅓, √2/3, …, √1601/42`. |
| 8.5 | `h(D) = (w√\|D\|/2π) L(1, χ_D)` counts the `SL(2, ℤ)`-orbits of primitive integer forms on the future mass shell of discriminant `D`. | The number of essentially different integer particles of a given mass is an L-value. |
| 8.6 | `X ↦ AXA*` on Hermitian matrices preserves `t² − x² − y² − z²`; on light rays `vv*` it is a Möbius map of the sky. | Lorentz boosts move the stars by Möbius maps; `SL(2, ℤ[i])` is a lattice of boosts. |

## The zeros as energy levels (note 09, [`scripts/zeros_statistics.py`](scripts/zeros_statistics.py))

| | Statement | In physical terms |
|---|---|---|
| 9.1 | Riemann–Siegel: `Z(t) = 2 Σ_{n≤N} cos(θ(t) − t log n)/√n + (−1)^{N−1}(t/2π)^{−1/4}[C₀(p) + C₁(p)(t/2π)^{−1/2}] + O(t^{−5/4})`, `N = ⌊√(t/2π)⌋`. | At height `t`, `Z` is a chord of `√(t/2π)` pure tones. |
| 9.2 | Riemann–von Mangoldt: `N(T) = θ(T)/π + 1 + S(T)`; mean density `log(t/2π)/2π`. | A Weyl term: states counted by phase-space volume (Berry–Keating's `H = xp`). |
| 9.3 | Explicit formula with a Gaussian: `Σ_γ g_σ(γ − E) = d̄(E) − (1/π) Σ_n Λ(n) n^{−1/2} e^{−σ²(log n)²/2} cos(E log n) + ε`. | A trace formula: the level density is a mean term plus one oscillation per periodic orbit; the orbits are the prime powers. |
| 9.4 | Montgomery: pair correlation `1 − (sin πu/πu)²`, that of the Gaussian unitary ensemble. | Levels repel; the unitary class is that of systems without time-reversal symmetry. |
| 9.5 | Berry: `Σ²(L) = (2/π²)[Σ_{p^k<E/2π} sin²(πLk log p/log(E/2π))/(k²p^k) + ∫₁^∞ sin²(πLτ)/τ² dτ]`. | Universal statistics at short range; at long range the shortest orbits, `log 2`, `log 3`, …, set the rigidity. |
| 9.6 | Independent spectra with relative densities `f_i` superpose with gap probability `E(s) = Π E_i(f_i s)`; spacing distribution `1 + E′(s)`. | Levels with different quantum numbers do not repel; the four characters mod 12 label four sectors. |

## Earlier notes

| Where | Statement | In physical terms | Checked by |
|---|---|---|---|
| Note 01, Theorems 1–2 | Every character of `SL(2, ℤ)` is a power of the `η²` multiplier; `T ↦ e^{2πi/12}`, `S ↦ −i`. | The modular group has twelve "pitches". | Lean ([morley-smith-formal](https://github.com/rmorleysmith/morley-smith-formal), `tau-track/TauTrack/Tones`); [`twelve_tones.py`](scripts/twelve_tones.py) |
| Note 02, Theorem 1 | For class number one, `#{k ∈ O_D : N(k) = n} = w_D Σ_{d∣n} χ_D(d)`. | Plane waves of equal frequency on a periodic lattice are counted by a character. | [`lattice_waves.py`](scripts/lattice_waves.py) |
| Note 02, Proposition 2 | `Σ′ (−1)^{x+y}(x² + y²)^{−s} = −4(1 − 2^{1−s}) ζ(s) L(s, χ₋₄)`. | The energy of a square ionic crystal is a product of L-values at `s = ½`. | [`lattice_waves.py`](scripts/lattice_waves.py) |
| Note 02, §4 | `ψ(x, p/q) = Σ_m a_m g(x − md/q)` with Gauss-sum weights; `\|a_m\| = 1/√q` for odd `q`. | Talbot images and Schroeder diffusers are equally bright because `\|G(q)\|² = q`. | [`gauss_sums_waves.py`](scripts/gauss_sums_waves.py) |
| Note 04, §2 | `log L(σ + it, χ) = Σ_p Σ_k χ(p)^k p^{−kσ} e^{−ikt log p}/k`. | Each prime is a tone of frequency `log p`. | [`lfunction_picture.py`](scripts/lfunction_picture.py) |
| Note 05, §3 | A hyperbolic `γ ∈ SL(2, ℤ)` gives a closed geodesic of length `2 arccosh(\|tr γ\|/2)`; the shortest is `4 log φ`, a loop of length `2 log φ` through `i` traversed in both directions. | The lowest note of the modular surface belongs to the golden ratio. | [`golden_ratio.py`](scripts/golden_ratio.py) |
| Note 06, §3 | The anomaly equations force `{y_u, y_d} = {2y_Q, −4y_Q}`, `y_L = −3y_Q`, `y_e = 6y_Q`. | The Standard Model's consistency fixes its charges. | [`standard_model.py`](scripts/standard_model.py) |
