"""Shared matplotlib style and colour palette for the manuscript figures."""

from __future__ import annotations

import matplotlib as mpl

# Palette
TEAL_LHD = "#2E8B8B"
TEAL_LHD_DARK = "#1F6363"
OCHRE_P = "#C4892A"
OCHRE_P_DARK = "#8C601E"
RED_CENTRAL = "#B73E3E"
BLUE_PORTAL = "#3E68B7"
CHARCOAL = "#222222"
LIGHT_GRAY = "#888888"
GRID_GRAY = "#DDDDDD"

# Tier saturations: lightest gray for Tier 0, deeper for Tier 3
TIER_COLORS = ["#BBBBBB", "#888888", "#555555", "#2E8B8B"]

# Cohort assignments
LHD_SAMPLES = ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")
P_SAMPLES_ALL = ("P2", "P3", "P6", "P7", "P14", "P17", "P18", "P21")
STEATOTIC = ("M1", "M2", "M3", "P6")

# Per-donor categorical palette for LHD plots: warm hues for the three
# steatotic donors (M1-M3), cool hues for the five non-steatotic donors
# (M4-M8). Hues drawn from Okabe-Ito + Tol-muted; all colorblind-safe and
# legible on white. The warm/cool grouping carries the steatosis story.
DONOR_PALETTE = {
    "M1": "#D55E00",  # vermillion (steatotic, strongest)
    "M2": "#E69F00",  # orange (steatotic)
    "M3": "#CC79A7",  # reddish purple (steatotic)
    "M4": "#0072B2",  # deep blue (non-steatotic)
    "M5": "#44AA99",  # teal (non-steatotic)
    "M6": "#009E73",  # bluish green (non-steatotic)
    "M7": "#56B4E9",  # sky blue (non-steatotic)
    "M8": "#332288",  # indigo (non-steatotic)
}


def donor_color(sample: str) -> str:
    return DONOR_PALETTE.get(sample, LIGHT_GRAY)

# Marker classification (from utils)
PORTAL_MARKERS = ("SDS", "ASS1", "CYP2A6", "HAL", "CPS1")
CENTRAL_MARKERS = ("CYP2E1", "CYP1A2", "GLUL", "CYP3A4")
HEADLINE_MARKERS = ("SDS", "ASS1", "CYP2A6", "HAL", "CYP2E1", "CYP1A2", "GLUL", "CYP3A4")


def marker_color(gene: str) -> str:
    if gene in CENTRAL_MARKERS:
        return RED_CENTRAL
    if gene in PORTAL_MARKERS:
        return BLUE_PORTAL
    return LIGHT_GRAY


def cohort_color(sample: str, dark: bool = False) -> str:
    if sample in LHD_SAMPLES:
        return TEAL_LHD_DARK if dark else TEAL_LHD
    if sample in P_SAMPLES_ALL:
        return OCHRE_P_DARK if dark else OCHRE_P
    return LIGHT_GRAY


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.grid": True,
        "grid.color": GRID_GRAY,
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "mathtext.default": "regular",
    })


# mm to inches
def mm(v: float) -> float:
    return v / 25.4


SINGLE_COL_W = mm(89)
ONE_HALF_COL_W = mm(140)
DOUBLE_COL_W = mm(180)


def panel_label(ax, label: str, x: float = -0.15, y: float = 1.05,
                ha: str = "left"):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha=ha)


def axis_orientation_strip(ax, y_offset: float = -0.18):
    """Add 'portal | central' annotation strip below an s-axis plot."""
    ax.annotate("portal", xy=(0.0, y_offset), xycoords="axes fraction",
                ha="left", va="top", fontsize=6.5, color=BLUE_PORTAL,
                style="italic")
    ax.annotate("central", xy=(1.0, y_offset), xycoords="axes fraction",
                ha="right", va="top", fontsize=6.5, color=RED_CENTRAL,
                style="italic")
