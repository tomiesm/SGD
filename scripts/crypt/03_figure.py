"""
Moor 2018 intestinal crypt-villus — supplementary validation figure (Fig S10).

A 2-panel figure:

  Panel A — bar chart of the OLS slope for the 15 canonical crypt-villus
            markers, coloured by literature-expected direction (blue =
            should decrease crypt->tip; red = should increase), each bar
            marked correct/incorrect, annotated "X/15 correct".
  Panel B — cumulative variance of the seven-zone expression trajectory,
            showing its low effective dimension in the 20-gene state space.

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
    ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.05, 1.0),
              frameon=False, fontsize=6, handlelength=1.4)
    plotstyle.panel_label(ax, "A")


def panel_b(ax, a2: dict):
    """Cumulative PCA variance of the observed seven-zone trajectory."""
    pca = a2["pca"]
    cumulative = np.asarray(pca["cumulative_explained_variance"], dtype=float)
    components = np.arange(1, len(cumulative) + 1)
    ax.plot(components, cumulative, "o-", color=plotstyle.TEAL_LHD_DARK,
            linewidth=1.5, markersize=4)
    ax.axhline(0.95, color=plotstyle.CHARCOAL, linestyle="--", linewidth=0.8)
    n95 = pca["N_eff_95pct"]
    ax.axvline(n95, color=plotstyle.LIGHT_GRAY, linestyle=":", linewidth=0.8)
    ax.scatter([n95], [cumulative[n95 - 1]], s=55, color=plotstyle.TEAL_LHD,
               edgecolor=plotstyle.TEAL_LHD_DARK, zorder=3)
    ax.text(0.96, 0.12,
            f"{n95} components capture\n{100*cumulative[n95-1]:.1f}% of variance",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="#CCCCCC",
                      linewidth=0.5, boxstyle="round,pad=0.25"))
    ax.set_xlim(0.7, len(cumulative) + 0.3)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("principal components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("Low-dimensional crypt-villus trajectory")
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
