"""Note 03: how many colours in the rainbow? Light, the octave and the hydrogen series.

What it computes
  (1) The visible band. With the common limits 380-750 nm it spans a frequency ratio of 1.974, just under an
      octave; with the CIE colorimetric limits 360-830 nm it spans 2.31. The seven diatonic steps of the octave
      750 -> 375 nm are set beside the conventional colour bands.
  (2) Hydrogen. The Rydberg formula 1/lambda = R (1/m^2 - 1/n^2) makes every frequency ratio rational. The
      visible Balmer lines have consecutive ratios 27/20, 28/25, 200/189, 405/392; each series m spans the
      exact ratio (m + 1)^2 / (2m + 1) from its first line to its limit: 4/3 (Lyman), 9/5 (Balmer),
      16/7 (Paschen), 25/9 (Brackett).

What it shows
  Newton chose seven colours by analogy with the seven notes of the scale; the spectrum is a continuum and colour
  vision is three-dimensional. The rational frequency ratios of hydrogen are exact consequences of the Rydberg
  formula, and several lie close to musical intervals.
"""
import math
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np

from common import GRID, INK, INK_2, check, save

c = 299792458.0
R_H = 1.09677583e7          # Rydberg constant for hydrogen, 1/m (vacuum wavelengths)

# ---------------------------------------------------------------------------------------------------
print("(1) the visible band")
check(abs(750 / 380 - 1.9737) < 1e-4, "380-750 nm: frequency ratio 1.9737, just under an octave (2)")
check(abs(830 / 360 - 2.3056) < 1e-4, "360-830 nm (CIE tables): frequency ratio 2.3056, just over an octave")
bands = [("violet", 380, 450), ("blue", 450, 495), ("green", 495, 570), ("yellow", 570, 590),
         ("orange", 590, 620), ("red", 620, 750)]
diat = [0, 2, 4, 5, 7, 9, 11, 12]
notes = ["C", "D", "E", "F", "G", "A", "B", "C′"]
edges = [750 * 2 ** (-k / 12) for k in diat]
widths = [round(edges[i] - edges[i + 1]) for i in range(7)]
print(f"     diatonic steps of 750 -> 375 nm: {', '.join(f'{n} {e:.0f}' for n, e in zip(notes, edges))}")
print(f"     widths of the 7 diatonic bands (nm): {widths}; of the 6 colour bands: {[b - a for _, a, b in bands]}")
check(widths == [82, 73, 33, 61, 55, 49, 22], "the diatonic widths differ from the colour bands")

# ---------------------------------------------------------------------------------------------------
print("(2) the hydrogen series")
nu = lambda m, n: Fraction(1, m * m) - Fraction(1, n * n)
balmer = {n: 1e9 / (R_H * float(nu(2, n))) for n in range(3, 8)}
for n, lam in balmer.items():
    print(f"     H{n} -> 2: 1/4 - 1/{n * n} = {nu(2, n)}, lambda = {lam:.1f} nm (vacuum)")
check(abs(balmer[3] - 656.47) < 0.05 and abs(balmer[4] - 486.27) < 0.05, "H-alpha 656.5 nm, H-beta 486.3 nm")
ratios = [nu(2, n + 1) / nu(2, n) for n in range(3, 7)]
check(ratios == [Fraction(27, 20), Fraction(28, 25), Fraction(200, 189), Fraction(405, 392)],
      f"consecutive Balmer frequency ratios {', '.join(map(str, ratios))}")
cents = lambda r: 1200 * math.log2(float(r))
print(f"     27/20 is {cents(Fraction(27, 20)) - cents(Fraction(4, 3)):+.1f} cents from the fourth 4/3; "
      f"28/25 is {cents(Fraction(28, 25)) - cents(Fraction(9, 8)):+.1f} cents from the tone 9/8")
spans = {m: Fraction(1, m * m) / nu(m, m + 1) for m in range(1, 5)}
check(spans == {1: Fraction(4, 3), 2: Fraction(9, 5), 3: Fraction(16, 7), 4: Fraction(25, 9)},
      "series spans (m+1)^2/(2m+1): Lyman 4/3 (a fourth), Balmer 9/5 (a just minor seventh), Paschen 16/7, "
      "Brackett 25/9")

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
    return tuple((f * v) ** 0.8 for v in (r, g, b))


lams = np.linspace(370, 760, 800)
fig, ax = plt.subplots(figsize=(7.2, 3.3))
strip = np.array([[rgb(l) for l in lams]])
ax.imshow(strip, extent=(370, 760, 3.15, 3.85), aspect="auto")
ax.text(366, 3.5, "spectrum\n(approximate)", ha="right", va="center", fontsize=8.5, color=INK_2)
for i, (name, a, b) in enumerate(bands):
    ax.add_patch(plt.Rectangle((a, 2.15), b - a, 0.7, color="#f0efec" if i % 2 else "#e4e3df", lw=0))
    ax.text((a + b) / 2, 2.5, name, ha="center", va="center", fontsize=8.5, color=INK, rotation=0 if b - a > 25 else 90)
ax.text(366, 2.5, "colour names\n(6 bands)", ha="right", va="center", fontsize=8.5, color=INK_2)
for i in range(7):
    a, b = edges[i + 1], edges[i]
    ax.add_patch(plt.Rectangle((a, 1.15), b - a, 0.7, color="#cde2fb" if i % 2 else "#9ec5f4", lw=0))
    ax.text((a + b) / 2, 1.5, notes[i], ha="center", va="center", fontsize=9, color=INK)
ax.text(366, 1.5, "diatonic octave\n750 → 375 nm", ha="right", va="center", fontsize=8.5, color=INK_2)
for n, lam in list(balmer.items())[:4]:
    ax.plot([lam, lam], [0.15, 0.85], color=INK, lw=1.6)
    ax.text(lam + 3, 0.5, ["Hα", "Hβ", "Hγ", "Hδ"][n - 3], fontsize=8.5, va="center", color=INK)
ax.text(366, 0.5, "hydrogen\n(Balmer lines)", ha="right", va="center", fontsize=8.5, color=INK_2)
ax.set_xlim(370, 760)
ax.set_ylim(0, 4)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("wavelength (nm)")
ax.set_title("Two divisions of the visible octave: seven colours and seven diatonic steps")
save(fig, "light-spectrum")
print("done")
