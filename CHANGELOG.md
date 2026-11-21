# Changelog

## Unreleased: corrections from an adversarial review (2026-11-21)

Three independent cold reads by AI agents were made of the notes and of the Lean statements, as a sceptical expert would
read them. The numerics were confirmed throughout; the corrections are to physics statements, framing, grading
and attribution.

- **Note 06 (rewritten).** The checked facts now come first and the programme last, as a working hypothesis with
  five questions (formerly five conjectures). Corrected physics: the hypercharges are fixed for one generation of
  fifteen with `y_Q ≠ 0`; with a right-handed neutrino the family `Y + ε(B − L)` solves the conditions; the
  gauge-only theorem of Lohitsiri and Tong needs quantised (rational) hypercharge, and a real counterexample is
  given; the electric charges need the Higgs doublet; the Fermat cubic appears for every number `N` of colours, so
  the exponent is that of the triangle diagram and not of `SU(3)`. The parity of a character and the Hodge types
  of a modular form are no longer called chirality. The citations of Costa–Dobrescu–Fox and
  Allanach–Gripaios–Tooby-Smith now say what those papers solve. The Lohitsiri–Tong paper was read.
- **Lean.** The hypercharge theorem is now also proved for `N` colours
  (`Hypercharge.gauge_anomaly_iff_colours`, `gravitational_anomaly_colours`).
- **Note 11 (rewritten).** The comparison with Maxwell's equations is removed; the powder-diffraction
  reading is stated as an identity for a monatomic primitive cubic crystal. Attributions corrected: the
  spherical-means solution of the wave equation is Poisson's (1819), Kirchhoff's 1883 paper is the integral
  formula; the list is in order of time; Dyson's remark is cited from Montgomery's paper.
- **Note 10.** Waldspurger's theorem gives the shell anisotropy relation within each class of `N` in `ℚ₂ˣ` modulo squares; the
  single constant across the seven classes is graded as computed. Roundness "as chance allows" is a theorem in
  four dimensions and a conjecture in space. The spin-2 tetrahedral state is fixed as a ray (up to a cube root of
  unity); spin 3 is the first strictly invariant vector.
- **Note 09.** The independence of the zeros of different `L`-functions is attributed to Bogomolny and Leboeuf
  (1994). Unitary statistics are described as the absence of time-reversal symmetry, not as handedness; the
  universality of level statistics is stated as a conjecture, with the modular surface as an exception.
- **Note 05.** The conjecture that the golden ratio "governs propagation" is replaced by two classical links
  (Hurwitz's constant as twice the height of the shortest geodesic; the icosahedron and `E8` through McKay) and
  two questions. Corrected: `m₂/m₁` is the ratio of the two lightest masses, not the lowest ratio; the cusps of
  `X(5)` in Klein's coordinate; the comparison with the musical fifth.
- **Notes 01–04, 07, 08.** `L(1, χ)` values are those of the primitive characters; the pitch reading is a
  convention up to the automorphisms of `ℤ/12`; hydrogen's rational ratios are those of the Rydberg formula; the
  logarithm of an `L`-function is the sum of prime tones; Landau's formula on the critical line assumes RH;
  CODATA 2022; the squarefree condition is stated wherever the `L`-value form is quoted.
- **New in note 06: Derivation 6.1, the hypercharge curve and its L-function.** The solutions of the gauge
  conditions are the three rational points of the elliptic curve `X³ + Y³ = Z³` (conductor 27, complex
  multiplication by `ℤ[ω]`), a group of order 3 in which `u ↔ d` is negation; its L-function is that of
  `η(3z)²η(9z)²`, with `L(1) = Ω/3 ≠ 0` (Coates–Wiles then gives finitely many points). All classical;
  reviewed by an independent AI agent before publication. Lean: `FermatCubicModP` (the count modulo primes `p ≡ 2 mod 3`).
- **New: note 12, lattice metrics and the cosmological-term functional of gravity from entropy.** The functional
  `Tr(𝒢 − I − ln 𝒢)` of the emergent cosmological term of Bianconi's theory, evaluated on normalised Gram matrices
  of plane lattices, equals `2 cosh d − 2`; a change of basis costs at least the discriminant `t² − 4`, least value
  5; Dirichlet's formula relates the minimum and the geodesic length to `L(1, χ_D)`. Lean:
  `LatticeMetricDivergence`. Reviewed by an independent AI agent before publication.
- **Note 11: hearing the lattice.** Beside Poisson duality, Kac's question for the torus `R³/Z³`: its tones give
  its echoes, and by Schiemann's theorem (1997, real ternary forms) they determine the cubic lattice up to rotation
  among lattices in `R³`; in four dimensions they do not (Schiemann, 1990), nor for plane membranes
  (Gordon–Webb–Wolpert, 1992). Milnor's tori on `E8 ⊕ E8` and `D16⁺` (1964, after Witt) are checked in the new
  `scripts/hearing_lattices.py`: equal theta series to squared length 60, the identity that gives equality at every
  length, and root graphs with two components and one. Suggested by the author; reviewed by an independent AI agent
  before publication.
- **Note 11: standard names.** "Four laws for a lattice of impulses" is now "Four relations for a cubic lattice
  of impulses", and the laws of duality, size, shape and silence are Poisson duality, echo strength, shell anisotropy
  and missing shells. Files: `notes/11-four-relations/`, `pdf/four-relations.pdf`, `scripts/shell_anisotropy.py`
  (formerly `notes/11-laws/`, `pdf/laws.pdf`, `scripts/shape_law.py`).
- **Grades and checks.** A new grade, *physics*, for published derivations and measurements. Every number quoted
  in a note is now asserted by its script to the precision printed.

## 0.6.4 (2026-11-21)

- Note 11, new section "Where the law is measured": in powder diffraction by a primitive cubic crystal the
  multiplicity of the ring of index N is `m_N = r₃(N) = (24/π) √N L(1, χ_D)`, and the rings
  `N = 4^a(8b+7)` are absent (the gap after the sixth line of a primitive cubic pattern).

## 0.6.3 (2026-11-21)

- The law of shape is now checked on all 201 squarefree `N ≤ 400` with `N ≢ 7 (mod 8)`, in each of the seven
  classes of `N` in `ℚ₂ˣ` modulo squares, to `10⁻¹¹` (`scripts/shape_law.py`; previously 16 shells).

## 0.6.2 (2026-11-21)

- Note 11: the laws written as one framed set of four lines (duality, size, shape, silence); the
  historical section is now a dated record from Huygens (1690) to Berry and Keating (1999), with a
  bibliography checked against Crossref.
- The "if" direction of the law of silence (no sum of three squares equals `4^a(8b+7)`) is proved in
  Lean in morley-smith-formal (`MorleySmithFormal/SilentShells`).
- Note 11 is indexed in `DERIVATIONS.md`.

## 0.6.1 (2026-11-21)

- Note 11 opens with the headline law `E_N = L(1, χ_D)/ζ(2)`: the strength of the N-th echo of a cubic
  lattice of flashes is a Dirichlet L-value (Gauss, Dirichlet, Kirchhoff).

## 0.6.0 (2026-11-21)

- Three statements are now proved in Lean in morley-smith-formal and graded so: the pitch map as the
  abelianisation of `SL(2, ℤ)`, with its explicit formula (note 01, new Proposition 3); the displacement
  formula in the modular group and the bound `4 log φ` (note 05), with `tanh d(i, ρ) = ½` (note 08); the
  solutions of the gauge anomaly conditions of one generation (note 06).
- Note 11, three laws for a lattice of impulses: the laws of duality, size and shape stated as the model's
  central equations, with their physical readings, grades and attributions.
- Lean paths break at slashes, and every line of the built notes stays inside the margin.

## 0.5.0 (2026-11-21)

- Note 10, spheres: echoes, multipoles and rotations (Derivations 10.1–10.7). The echoes of a cubic lattice of
  flashes, with strengths `L(1, χ_D)/ζ(2)`; the hexadecapole of a lattice shell as a coefficient of a modular
  form (`η(z)⁴η(2z)²η(4z)⁴`, `η(z)²η(2z)⁷η(4z)²`, `η(2z)¹²` in dimensions 2, 3, 4), and its square as a central
  L-value; the crystal field of a cubic lattice as the nearest-neighbour field times `0.888064766…`; roundness
  of shells and the Ramanujan bound; the spin states fixed by the Hurwitz units and the newforms of level 2;
  six rotations at the bound `2√5`.
- Notes 02 and 04 cite the Lean proof of the functional equation of the newform of level 11
  (`ElevenFunctionalEquation` in morley-smith-formal: Fricke eigenvalue −1, `Λ(2 − s) = 11^{s−1}Λ(s)`,
  `L(f, 1) > 0`, reality on the critical line). Note 04 evaluates the central value and the five zeros below
  `t = 15`.
- Note 06: without the gravitational condition, anomaly cancellation for one generation is Fermat's cubic
  `y_e³ + (y_u − 2y_Q)³ = (y_u + 4y_Q)³` (Lohitsiri and Tong), whose trivial solutions are the observed
  hypercharges; the statement is formalised in Lean in Physlib.

## 0.4.0 (2026-11-20)

- Note 09, the zeros as energy levels (Derivations 9.1–9.6): all 9511 zeros of ζ in a window near t = 10⁶; the
  explicit formula as a trace formula, verified to 2 × 10⁻¹⁰; random-matrix statistics at short range and
  Berry's sum over prime powers at long range; the primes as a Poisson process; the four L-functions of
  ℚ(ζ₁₂) have independent spectra.
- scripts/lfunctions.py: a vectorised evaluator for zeta and Dirichlet L-functions on the critical line
  (Hurwitz zeta by Euler–Maclaurin), checked against mpmath.

## 0.3.0 (2026-11-20)

- Note 07, impulses, packets and spectra (Derivations 7.1–7.5).
- Note 08, the speed of light, the light cone and spheres (Derivations 8.1–8.6).
- DERIVATIONS.md: an index of the reusable derivations, each with its physical reading and its check.
- Shared preamble: numbered Derivation boxes, Conjecture environment, "In physical terms" paragraphs.
- Editorial pass over all notes, the README and the script headers: a formal register throughout, abstracts in
  prose, open points stated as conjectures and hypotheses, a grade legend under every status table. Note 05
  retitled.
- Note 01: the two twelves on the modular surface (continued fractions as geodesics). Note 03: Newton's 1672
  colour list, quoted. Note 05: the shortest geodesic is reciprocal, a loop of length 2 log φ through i
  traversed in both directions; the conjecture stated. Note 06: the programme stated as four propositions and
  five conjectures; the six roots of SU(3) as one orbit of the stabiliser of ρ. Note 07: Poisson summation in
  heat-kernel form; Landau's formula stated unconditionally. Note 08: the speeds of the nine class-number-one
  forms.

## 0.2.0 (2026-11-20)

- Note 04, what an L-function is, and what it looks like.
- Note 05, the golden ratio in the modular world and in physics.
- Note 06, towards a number-theoretic standard model (conjectures with exact evidence).
- Note 01: the weight of Δ attributed to the valence formula only. Note 02: the level-11 root number's Lean
  proof is in the author's working repository, not yet public.
- DOI badge and Zenodo record.

## 0.1.0 (2026-11-20)

- Note 01, the twelve characters of the modular group (Lean proofs in morley-smith-formal).
- Note 02, quadratic characters, lattices and waves.
- Note 03, how many colours in the rainbow.
- Scripts that assert every number, CI for the scripts and the LaTeX build.
