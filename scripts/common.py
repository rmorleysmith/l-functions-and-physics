"""Shared plotting style and helpers for the scripts in this repository.

Colours are the light-mode slots of a colour-blind-checked categorical palette, with near-black and grey
ink on a white surface, so the figures read the same on screen, on GitHub (light or dark theme) and in
print.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"     # slot 1
ORANGE = "#eb6834"   # slot 2
AQUA = "#1baf7a"     # slot 3
BLUE_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.size": 10,
    "lines.linewidth": 2,
    "mathtext.fontset": "cm",
    "svg.hashsalt": "l-functions-and-physics",
    "pdf.fonttype": 42,
})


def save(fig, name):
    """Write figures/<name>.pdf (for the LaTeX notes) and figures/<name>.svg (for the web)."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf, figures/{name}.svg")


def check(cond, what):
    """Assert a reported fact; the scripts fail loudly if a number stops holding."""
    if not cond:
        raise AssertionError(what)
    print(f"  ok  {what}")
