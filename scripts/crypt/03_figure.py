"""
Moor 2018 intestinal crypt-villus — supplementary validation figure (Fig S10).

A 2-panel figure:

  Panel A — bar chart of the OLS slope for the 15 canonical crypt-villus
            markers, coloured by literature-expected direction (blue =
            should decrease crypt->tip; red = should increase), each bar
            marked correct/incorrect, annotated "X/15 correct".
  Panel B — W stability: empirical block-bootstrap stable-entry fraction
            vs the gene-permutation null (mean and p95), at N=7 zones.

Reads moor_crypt_results.json (produced by 01_marker_recovery.py and
02_wmatrix_stability.py). Uses sgd.plotstyle for visual consistency with
the manuscript figures.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Make the src/sgd package importable without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sgd import plotstyle  # noqa: E402
from sgd.config import RESULTS, FIGURES  # noqa: E402

warnings.filterwarnings("ignore")
np.random.seed(0)

RESULTS_JSON = RESULTS / "moor_crypt_results.json"
FIG_PDF = FIGURES / "figS10_crypt_validation.pdf"

# plotstyle palette: blue = "should decrease", red = "should increase".
COL_DECREASE = plotstyle.BLUE_PORTAL
COL_INCREASE = plotstyle.RED_CENTRAL


def panel_a(ax, a1: dict):
    """Marker-recovery bar chart."""
    markers = a1["markers"]
    genes = [m["gene"] for m in markers]
    slopes = [m["slope"] if m["slope"] is not None else 0.0 for m in markers]
    expected = [m["expected_direction"] for m in markers]
    matches = [m["sign_match"] for m in markers]

    x = np.arange(len(genes))
    colors = [COL_DECREASE if e == "negative" else COL_INCREASE
              for e in expected]

    # Correct bars filled solid; incorrect bars hatched.
    for xi, sl, c, mt in zip(x, slopes, colors, matches):
        if mt == "correct":
            ax.bar(xi, sl, color=c, edgecolor=plotstyle.CHARCOAL,
                   linewidth=0.6, width=0.72)
        else:
            ax.bar(xi, sl, color="white", edgecolor=c, linewidth=1.0,
                   hatch="////", width=0.72)

    ax.axhline(0.0, color=plotstyle.CHARCOAL, linewidth=0.8)

    # Correct / incorrect glyphs above/below each bar.
    ymax = max(slopes) if max(slopes) > 0 else 1.0
    ymin = min(slopes) if min(slopes) < 0 else 0.0
    span = ymax - ymin
    for xi, sl, mt in zip(x, slopes, matches):
        glyph = "✓" if mt == "correct" else "✗"
        # Place the glyph just outside the bar tip.
        if sl >= 0:
            yy = sl + 0.03 * span
            va = "bottom"
        else:
            yy = sl - 0.03 * span
            va = "top"
        ax.text(xi, yy, glyph, ha="center", va=va, fontsize=7,
                color=plotstyle.CHARCOAL)

    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("OLS slope on s (Crypt→0, V6 tip→1)")
    ax.set_title("Crypt-villus marker recovery")
    ax.margins(y=0.18)

    n_correct = a1["n_markers_correct"]
    n_total = a1["n_markers_total"]
    ax.text(0.03, 0.95, f"{n_correct}/{n_total} correct",
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")

    # Direction legend.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COL_DECREASE),
        plt.Rectangle((0, 0), 1, 1, color=COL_INCREASE),
        plt.Rectangle((0, 0), 1, 1, facecolor="white",
                      edgecolor=plotstyle.CHARCOAL, hatch="////"),
    ]
    labels = ["expected ↓ (crypt-high)", "expected ↑ (tip-high)",
              "sign mismatch"]
    ax.legend(handles, labels, loc="lower right", frameon=False,
              fontsize=6, handlelength=1.4)
    plotstyle.panel_label(ax, "A")


def panel_b(ax, a2: dict):
    """W stability: empirical block-bootstrap vs gene-permutation null."""
    emp = a2["block_bootstrap"]["empirical_stable_fraction"]
    perm = a2["permutation_null"]
    perm_vals = np.asarray(perm["stable_fraction_per_permutation"], dtype=float)
    perm_mean = perm["mean_stable_fraction"]
    perm_p95 = perm["p95_stable_fraction"]
    n_perm = perm["n_permutations"]
    n_zones = a2["n_zones"]

    # Permutation-null distribution as a strip of jittered points.
    rng = np.random.RandomState(0)
    jitter = rng.uniform(-0.16, 0.16, size=perm_vals.size)
    ax.scatter(np.full(perm_vals.size, 0.0) + jitter, perm_vals,
               s=10, color=plotstyle.LIGHT_GRAY, alpha=0.55,
               edgecolor="none", zorder=2,
               label=f"gene-permutation null ({n_perm} perms)")

    # Null mean and p95 lines.
    ax.hlines(perm_mean, -0.32, 0.32, color=plotstyle.CHARCOAL,
              linewidth=1.4, zorder=3,
              label=f"null mean = {perm_mean:.3f}")
    ax.hlines(perm_p95, -0.32, 0.32, color=plotstyle.CHARCOAL,
              linewidth=1.2, linestyle="--", zorder=3,
              label=f"null p95 = {perm_p95:.3f}")

    # Empirical stable fraction.
    ax.scatter([1.0], [emp], s=90, color=plotstyle.TEAL_LHD,
               edgecolor=plotstyle.TEAL_LHD_DARK, linewidth=1.0, zorder=4,
               label=f"empirical = {emp:.3f}")
    ax.annotate(f"{emp:.3f}", xy=(1.0, emp), xytext=(1.16, emp),
                fontsize=7, va="center", ha="left",
                color=plotstyle.TEAL_LHD_DARK)

    ax.set_xlim(-0.6, 1.8)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["permutation\nnull", "empirical"])
    ax.set_ylabel(f"W stable-entry fraction (|z| > 1.96)")
    ax.set_title(f"W bootstrap stability (N = {n_zones} zones)")
    ax.margins(y=0.20)
    ax.legend(loc="upper right", frameon=False, fontsize=5.8,
              handlelength=1.4)
    plotstyle.panel_label(ax, "B")


def main():
    results = json.loads(RESULTS_JSON.read_text())
    a1 = results["analysis1_marker_recovery"]
    a2 = results["analysis2_W_nonidentifiability"]

    plotstyle.apply_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(plotstyle.DOUBLE_COL_W, plotstyle.mm(78)),
    )
    panel_a(axes[0], a1)
    panel_b(axes[1], a2)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(FIG_PDF)
    plt.close(fig)
    print(f"[FIG] Wrote {FIG_PDF}")


if __name__ == "__main__":
    main()
