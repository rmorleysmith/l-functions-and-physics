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
| 7.5 | Landau: `Σ_{0<γ≤T} x^ρ = −(T/2π) Λ(x) + O(log T)`; under the Riemann hypothesis, `Σ x^{iγ} = −(T/2π) Λ(x)/√x + O(log T)`. | The zeros, played together, ring only at the logarithms of the prime powers: primes and zeros are a Fourier pair. |

## Light, cones and spheres (note 08, [`scripts/light_cone.py`](scripts/light_cone.py))

| | Statement | In physical terms |
|---|---|---|
| 8.1 | Modes of light in a periodic box: `2#{n ∈ ℤ³ : \|n\|² ≤ N} ~ Vω³/(3π²c³)`; multiplicity `2r₃(N)` at `ω = (2πc/L)√N`. | `c` converts counting lattice points into counting frequencies (Planck's `c³`). |
| 8.2 | Gauss: `r₃(n) = 12h(−4n)`, `24h(−n)` or `0` (squarefree `n > 3`); so `r₃(n) = (24√n/π) L(1, χ₋₄ₙ)` for `n ≡ 1, 2 mod 4`. | For squarefree `n`, the number of ways light fits a cubic box at pitch `√n` is a multiple of an L-value. |
| 8.3 | `(T, X, Y) = (a + c, b, a − c)` makes `b² − 4ac = X² + Y² − T²`; `SL(2, ℤ)` acts by integral Lorentz transformations. | Gauss's forms are integer vectors in 2+1 spacetime: definite = particle, degenerate = light, indefinite = geodesic. |
| 8.4 | Relative rapidity `cosh η = −B(f, g)/√(D_f D_g)`; for `x² + y²` and `x² + xy + y²`, `tanh η = ½`. In the rest frame of `x² + y²`, the form `x² + bxy + cy²` has Lorentz factor `(1 + c)/√\|D\|`. | The Gaussian and Eisenstein forms move relative to each other at `c/2`; the nine class-number-one forms have speeds `0, ½, ⅓, √2/3, …, √1601/42`. `tanh d(i, ρ) = ½` is proved in Lean (`MorleySmithFormal/ModularSystole`). |
| 8.5 | `h(D) = (w√\|D\|/2π) L(1, χ_D)` counts the `SL(2, ℤ)`-orbits of primitive integer forms on the future mass shell of discriminant `D`. | The number of essentially different integer particles of a given mass is an L-value. |
| 8.6 | `X ↦ AXA*` on Hermitian matrices preserves `t² − x² − y² − z²`; on light rays `vv*` it is a Möbius map of the sky. | Lorentz boosts move the stars by Möbius maps; `SL(2, ℤ[i])` is a lattice of boosts. |

## The zeros as energy levels (note 09, [`scripts/zeros_statistics.py`](scripts/zeros_statistics.py))

| | Statement | In physical terms |
|---|---|---|
| 9.1 | Riemann–Siegel: `Z(t) = 2 Σ_{n≤N} cos(θ(t) − t log n)/√n + (−1)^{N−1}(t/2π)^{−1/4}[C₀(p) + C₁(p)(t/2π)^{−1/2}] + O(t^{−5/4})`, `N = ⌊√(t/2π)⌋`. | At height `t`, `Z` is a chord of `√(t/2π)` pure tones. |
| 9.2 | Riemann–von Mangoldt: `N(T) = θ(T)/π + 1 + S(T)`; mean density `log(t/2π)/2π`. | A Weyl term: states counted by phase-space volume (Berry–Keating's `H = xp`). |
| 9.3 | Explicit formula with a Gaussian: `Σ_γ g_σ(γ − E) = d̄(E) − (1/π) Σ_n Λ(n) n^{−1/2} e^{−σ²(log n)²/2} cos(E log n) + ε`. | A trace formula: the level density is a mean term plus one oscillation per periodic orbit; the orbits are the prime powers. |
| 9.4 | Montgomery (a conjecture): pair correlation `1 − (sin πu/πu)²`, that of the Gaussian unitary ensemble. | Levels repel; the unitary class is that of systems without time-reversal symmetry. |
| 9.5 | Berry (a semiclassical formula): `Σ²(L) = (2/π²)[Σ_{p^k<E/2π} sin²(πLk log p/log(E/2π))/(k²p^k) + ∫₁^∞ sin²(πLτ)/τ² dτ]`. | Universal statistics at short range; at long range the shortest orbits, `log 2`, `log 3`, …, set the rigidity. |
| 9.6 | Independent spectra with relative densities `f_i` superpose with gap probability `E(s) = Π E_i(f_i s)`; spacing distribution `1 + E′(s)`. | Levels with different quantum numbers do not repel; the four characters mod 12 label four sectors. |

## Spheres (note 10, [`scripts/spheres.py`](scripts/spheres.py))

| | Statement | In physical terms |
|---|---|---|
| 10.1 | Harmonic polynomials of degree `l` in `d` variables have dimension `2`, `2l + 1`, `(l + 1)²` for `d = 2, 3, 4`, and are orthogonal on the sphere for different `l`. | Packets of angular momentum; `(l + 1)²` is the degeneracy of hydrogen's level `n = l + 1` (Fock). |
| 10.2 | A flash at every point of `ℤ³`: `u(0, t) = Σ r₃(N) δ(t − √N)/(4π√N) = t + Σ r₃(N) sin(2π√N t)/(2π√N)`; `r₃(N)/(4π√N) = L(1, χ_D)/ζ(2)` for squarefree `N ≢ 7 mod 8`. | Echoes at the times `√N`, each as loud as an L-value, none at `N = 4^a(8b + 7)`; the echoes and the tones carry the same integers. |
| 10.3 | `Σ_{x∈ℤ^d} K₄(x) q^{\|x\|²} = (2d(d − 1)/(d + 2)) θ^d ψ`, `K₄ = Σ xᵢ⁴ − (3/(d + 2))\|x\|⁴`, `ψ = η(z)⁸η(4z)⁸/η(2z)⁸`; for `d = 2, 3, 4`: `η(z)⁴η(2z)²η(4z)⁴`, `(12/5) η(z)²η(2z)⁷η(4z)²`, `4 η(2z)¹²`. | The hexadecapole of a shell is a coefficient of a modular form of weight `4 + d/2`. The crystal field of a simple cubic lattice is the nearest-neighbour field times `L(g, 9/2) = 0.888064766…` (rock salt: `1.0224521…`). |
| 10.4 | For squarefree `N ≢ 7 mod 8`: `a(N)²/N^{9/2} = L(5, F ⊗ χ_D)/L(5, F ⊗ χ₋₄)`, with `g = Σ a(N) q^N = η(z)²η(2z)⁷η(4z)²` and `F` the newform of weight 10 and level 2, the Shimura lift of `g`. Waldspurger's theorem gives the proportionality within each class of `N` in `ℚ₂ˣ` modulo squares; one constant for the seven classes is computed (`N ≤ 400`). | The size of a shell is an L-value at the edge of the critical strip; the square of its shape is an L-value at the centre. |
| 10.5 | Anisotropy `μ_d(N)` of a shell: `¼ cos 4θ_p` for `d = 2`; `μ₃² = (π²/100) L(5, F ⊗ χ_D)/(L(5, F ⊗ χ₋₄) L(1, χ_D)² √N)`; `\|μ₄(p)\| ≤ √p/(p + 1)` (Deligne). | In four dimensions shells are as round as random points, anisotropy about `1/√(number of points)`: this is the Ramanujan bound (Deligne). In space the same rate is conjectural, and the formula for `μ₃²` is computed. |
| 10.6 | Forms fixed by the 24 Hurwitz units have generating function `(1 + t¹²)/((1 − t⁶)(1 − t⁸))`, the count of newforms of weight `l + 2` and level 2; `Σ_x ⟨v, x·v⟩ q^{N(x)}` is the newform (`Φ ↦ η(z)⁸η(2z)⁸`, `Ψ ↦ F`). | State vectors fixed by the 24 Hurwitz units (octahedron, cube) are newforms; in the four cases computed the root number is a parity of the state. |
| 10.7 | The six rotations from `1 ± 2i`, `1 ± 2j`, `1 ± 2k` sum to `2(cos ϑJ_x + cos ϑJ_y + cos ϑJ_z)`, `cos ϑ = −3/5`, with eigenvalues `\|λ\| ≤ 2√5` on every degree `l ≥ 1` (Lubotzky–Phillips–Sarnak). | A walk by these rotations forgets its start as fast as a walk on a tree; they are the V gates of a quantum bit. |

## Four laws for a lattice of impulses (note 11)

Note 11 restates derivations 10.2 and 10.4, with Gauss's count from note 08, as four lines for one system, a cubic lattice of impulses heard from one of its sites.

| Law | Statement | Grade |
|---|---|---|
| Duality | Σ E_N δ(t − √N) = t + Σ r₃(N) sin(2π√N t)/(2π√N), with E_N = r₃(N)/(4π√N) | classical, computed |
| Size | E_N ζ(2) = L(1, χ_D) for squarefree N ≢ 7 mod 8 | classical, computed |
| Shape | S_N L(5, F⊗χ₋₄) = L(5, F⊗χ_D), with S_N = a(N)²/N^{9/2} | classical within a class of N in ℚ₂ˣ modulo squares; one constant for the seven classes computed |
| Silence | E_N = 0 exactly when N = 4^a(8b+7) | "if": Lean ([`SilentShells`](https://github.com/rmorleysmith/morley-smith-formal/tree/main/MorleySmithFormal/SilentShells)); "only if": classical |

## Lattice metrics and the cosmological-term functional (note 12, [`scripts/lattice_divergence.py`](scripts/lattice_divergence.py))

`S(A, B) = tr(G − 1 − log G)`, `G = AB⁻¹`: the form of the emergent cosmological term of Bianconi's gravity from entropy, and twice the Kullback–Leibler divergence of `N(0, A)` from `N(0, B)`.

| | Statement | In physical terms |
|---|---|---|
| 12.1 | For `M(z) = (1/y)[[x² + y², x], [x, 1]]` (the Gram matrix of the basis `(z, 1)` divided by the covolume) and `G = M(z)M(w)⁻¹`: `det G = 1`, `tr G = 2 cosh δ(z, w)`, eigenvalues `e^{±δ}`, and `S = 2 cosh δ − 2 = \|z − w\|²/(Im z Im w)` (the point-pair invariant); `S = δ² + O(δ⁴)`, and the second-order term is the Fisher metric of `N(0, M(z))`, the hyperbolic metric. | A harmonic trap switched suddenly between thermal covariances `σ²M(z)` and `σ²M(w)` dissipates on average `k_BT (cosh δ − 1)`. |
| 12.2 | For `γ = (a b; c d) ∈ SL(2, ℤ)` with trace `t`: `S(M(z), M(γz)) = t² − 4 + (c(x² + y²) + (d − a)x − b)²/y² ≥ t² − 4`, with equality exactly when `c(x² + y²) + (d − a)x − b = 0`; for `\|t\| > 2` the least value is `5`, attained by `(2 1; 1 1)` at `z = i`. | A sudden switch between two bases of one lattice related by a hyperbolic `γ` dissipates on average at least `k_BT (t² − 4)/2`. |
| 12.3 | For hyperbolic `γ`, `t > 2`: `S_γ = t² − 4 = 4 sinh²(ℓ/2)` and `ℓ/√S_γ = (k/(h(D) f)) L(1, χ_D)` (Dirichlet), with `t² − 4 = Df²` and `ε_t = ε_D^k`. | `ℓ` and `S_γ` determine each other, and `ℓ/√S_γ` is `k/(h(D)f)` times `L(1, χ_D)`. |

Lean: 12.1 and 12.2 in `MorleySmithFormal/LatticeMetricDivergence`.

## The hypercharge curve (note 06, [`scripts/standard_model.py`](scripts/standard_model.py))

| | Statement | In physical terms |
|---|---|---|
| 6.1 | For one generation with `N` colours and rational hypercharges, the non-zero solutions of the gauge anomaly conditions, up to scale, are the rational points of `X³ + Y³ = Z³`, `(X, Y, Z) = (y_e, y_u + (1 − N)y_Q, y_u + (N + 1)y_Q)`. There are three, collinear, a group of order 3 in which `u ↔ d` is negation. The curve is `y² + y = x³ − 7` (conductor 27, complex multiplication by `ℤ[ω]`); it has `p + 1 − a(p)` points modulo `p`, with `Σ a(n)qⁿ = η(3z)²η(9z)²`; `Λ(2 − s) = Λ(s)` and `L(1) = Ω/3 = 0.588879583…`, not zero, which by Coates–Wiles gives finitely many rational points (and, with the counts 6 and 9 modulo 5 and 7, exactly three). All classical. | The hypercharges of a generation are the rational points of one elliptic curve, the same for every number of colours; that they are fixed is visible in the central value of its L-function. Lean: the list of solutions (`MorleySmithFormal/Hypercharge`), and the count for `p ≡ 2 mod 3` (`MorleySmithFormal/FermatCubicModP`). |

## Earlier notes

| Where | Statement | In physical terms | Checked by |
|---|---|---|---|
| Note 01, Theorems 1–2 | Every character of `SL(2, ℤ)` is a power of the `η²` multiplier; `T ↦ e^{2πi/12}`, `S ↦ −i`. The pitch `p : SL(2, ℤ) → ℤ/12` is the abelianisation, with an explicit formula in the entries modulo 12 (Proposition 3). | The modular group has twelve "pitches". | Lean ([morley-smith-formal](https://github.com/rmorleysmith/morley-smith-formal), `tau-track/TauTrack/Tones`); [`twelve_tones.py`](scripts/twelve_tones.py) |
| Note 02, Theorem 1 | For class number one, `#{k ∈ O_D : N(k) = n} = w_D Σ_{d∣n} χ_D(d)`. | Plane waves of equal frequency on a periodic lattice are counted by a character. | [`lattice_waves.py`](scripts/lattice_waves.py) |
| Note 02, Proposition 2 | `Σ′ (−1)^{x+y}(x² + y²)^{−s} = −4(1 − 2^{1−s}) ζ(s) L(s, χ₋₄)`. | The energy of a square ionic crystal is a product of L-values at `s = ½`. | [`lattice_waves.py`](scripts/lattice_waves.py) |
| Note 02, §4 | `ψ(x, p/q) = Σ_m a_m g(x − md/q)` with Gauss-sum weights; `\|a_m\| = 1/√q` for odd `q`. | Talbot images and Schroeder diffusers are equally bright because `\|G(q)\|² = q`. | [`gauss_sums_waves.py`](scripts/gauss_sums_waves.py) |
| Note 04, §2 | `log L(σ + it, χ) = Σ_p Σ_k χ(p)^k p^{−kσ} e^{−ikt log p}/k`. | Each prime is a tone of frequency `log p`. | [`lfunction_picture.py`](scripts/lfunction_picture.py) |
| Note 04, §4 | For `f = η(z)²η(11z)²` and `Λ(s) = (2π)^{−s}Γ(s)L(f, s)`: `Λ(2 − s) = 11^{s−1}Λ(s)`, `L(f, 1) > 0`, and `11^{s/2}Λ(s)` is real on `Re s = 1`. | Hardy's function of a modular form: its zeros on the line are the sign changes of a real signal. | Lean ([morley-smith-formal](https://github.com/rmorleysmith/morley-smith-formal), `tau-track/TauTrack/ElevenFunctionalEquation`); [`lfunction_picture.py`](scripts/lfunction_picture.py) |
| Note 05, §3 | A hyperbolic `γ ∈ SL(2, ℤ)` gives a closed geodesic of length `2 arccosh(\|tr γ\|/2)`; the shortest is `4 log φ`, a loop of length `2 log φ` through `i` traversed in both directions. | The shortest periodic orbit on the modular surface belongs to the golden ratio. | Lean ([morley-smith-formal](https://github.com/rmorleysmith/morley-smith-formal), `MorleySmithFormal/ModularSystole`: the displacement formula and the bound `4 log φ`); [`golden_ratio.py`](scripts/golden_ratio.py) |
| Note 06, §3 | For one generation of fifteen fermions with `y_Q ≠ 0`, the four anomaly equations force `{y_u, y_d} = {2y_Q, −4y_Q}`, `y_L = −3y_Q`, `y_e = 6y_Q`; for `y_Q = 0` they give `y_L = y_e = 0`, `y_d = −y_u`. For rational hypercharges, without the gravitational condition, the cubic one is Fermat's `y_e³ + (y_u − 2y_Q)³ = (y_u + 4y_Q)³` (Lohitsiri–Tong), with the same solutions; for `N` colours it is `y_e³ + (y_u + (1 − N)y_Q)³ = (y_u + (N + 1)y_Q)³`. | For quantised hypercharge, consistency fixes the hypercharges of a generation up to one scale; the electric charges follow once the Higgs doublet is added. | [`standard_model.py`](scripts/standard_model.py); Lean ([morley-smith-formal](https://github.com/rmorleysmith/morley-smith-formal), `MorleySmithFormal/Hypercharge`); also Lean in [Physlib](https://github.com/leanprover-community/physlib) (`SM.SMNoGrav.One.accGravSatisfied`) |
