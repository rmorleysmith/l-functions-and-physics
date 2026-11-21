# Changelog

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
