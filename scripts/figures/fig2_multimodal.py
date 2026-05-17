"""Figure 2 — Multi-modal reproducibility (revised v2)."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

# Make the src/sgd package importable without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

from sgd.config import FIGURES, RESULTS, SUPP_T2
from sgd.plotstyle import (BLUE_PORTAL, CHARCOAL, GRID_GRAY, LIGHT_GRAY,
                           RED_CENTRAL, TEAL_LHD, TEAL_LHD_DARK, apply_style,
                           marker_color, mm, panel_label)

warnings.filterwarnings("ignore")
apply_style()

CROSS_PLATFORM = RESULTS / "cross_platform_M2_M6.json"
CROSS_PIPE = RESULTS / "cross_pipeline_reproducibility.json"
SIGN_BASE = RESULTS / "sign_correlation_baseline.json"
B8 = RESULTS / "b8_sensitivity.json"
PER_GRAD = RESULTS / "per_gene_gradient.parquet"

MARKERS_CENTRAL = ("CYP2E1", "CYP1A2", "GLUL", "CYP3A4")
MARKERS_PORTAL = ("SDS", "ASS1", "CYP2A6", "HAL", "CPS1")
MARKERS = MARKERS_PORTAL + MARKERS_CENTRAL


def annot_box(ax, text: str, x: float = 0.04, y: float = 0.96,
                fontsize: float = 6.8) -> None:
    """Subtle annotation box anchored top-left of axes."""
    ax.text(x, y, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=fontsize, color=CHARCOAL,
            bbox=dict(facecolor="white", edgecolor="#CCCCCC", linewidth=0.5,
                       boxstyle="round,pad=0.30", alpha=0.92))


def shade_agreement_quadrants(ax, xlim, ylim, color=TEAL_LHD, alpha=0.12):
    """Shade upper-right and lower-left quadrants for "agreement" emphasis."""
    ax.axhspan(0, ylim[1], xmin=0.5 + (0 - xlim[0]) / (xlim[1] - xlim[0]) * 0,
                facecolor=color, alpha=alpha, zorder=0)
    ax.add_patch(mpatches.Rectangle((0, 0), xlim[1], ylim[1],
                                      facecolor=color, alpha=alpha, zorder=0))
    ax.add_patch(mpatches.Rectangle((xlim[0], ylim[0]), -xlim[0], -ylim[0],
                                      facecolor=color, alpha=alpha, zorder=0))


def panel_A(ax_left, ax_right) -> None:
    """Cross-platform Visium-vs-HD on M2 + M6."""
    d = json.loads(CROSS_PLATFORM.read_text())
    for ax, sample in [(ax_left, "M2"), (ax_right, "M6")]:
        info = d["per_donor"][sample]
        slopes = info.get("panel_slopes")
        if slopes is None:
            ax.text(0.5, 0.5, f"{sample}\nno data", ha="center", va="center")
            continue
        genes = slopes["genes"]
        v = np.asarray(slopes["visium_slope"])
        h = np.asarray(slopes["hd_slope"])

        is_marker = np.array([g in MARKERS for g in genes])
        ax.axhline(0, color=GRID_GRAY, lw=0.6, zorder=0)
        ax.axvline(0, color=GRID_GRAY, lw=0.6, zorder=0)
        all_vals = np.concatenate([v, h])
        lim_lo, lim_hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        pad = 0.15 * (lim_hi - lim_lo)
        lim = (lim_lo - pad, lim_hi + pad)
        ax.plot([lim[0], lim[1]], [lim[0], lim[1]], color=LIGHT_GRAY,
                 lw=0.7, linestyle="--", zorder=1)
        # Non-markers first (small gray)
        ax.scatter(v[~is_marker], h[~is_marker], color=LIGHT_GRAY, s=10,
                    alpha=0.55, zorder=2, edgecolor="white", linewidths=0.3)
        # Markers — large, bold
        for i in np.where(is_marker)[0]:
            c = (RED_CENTRAL if genes[i] in MARKERS_CENTRAL
                 else BLUE_PORTAL)
            ax.scatter(v[i], h[i], color=c, s=44, zorder=4,
                        edgecolor=CHARCOAL, linewidths=0.6)

        hc = info["high_confidence_subset"]
        sa = hc["sign_agreement_pct"]
        rho = info["rank_correlation_all_panel"]
        annot_box(ax, f"{sample}\nsign {sa:.0f}% high-conf\n$\\rho$ = {rho:.2f}",
                   x=0.04, y=0.96, fontsize=7)
        ax.set_xlabel("Visium slope", fontsize=7)
        if ax is ax_left:
            ax.set_ylabel("Visium HD slope", fontsize=7)
        ax.tick_params(axis="both", labelsize=6.5)
        ax.set_xlim(*lim); ax.set_ylim(*lim)


def _load_t2_q005():
    """Return DataFrame indexed by gene with column 'layer_diff' for the
    q<0.05 subset of Supp T2."""
    t2 = pd.read_excel(SUPP_T2, sheet_name="mean zonation of non steatotic",
                        header=1)
    t2.columns = [str(c).strip() for c in t2.columns]
    if "gene_name" in t2.columns:
        t2 = t2.set_index("gene_name")
    elif "Gene_Name" in t2.columns:
        t2 = t2.set_index("Gene_Name")
    for c in ("mean_zone_1", "mean_zone_2", "mean_zone_7", "mean_zone_8"):
        t2[c] = pd.to_numeric(t2[c], errors="coerce")
    if "qValue" in t2.columns:
        q = pd.to_numeric(t2["qValue"], errors="coerce")
        t2 = t2[q < 0.05]
    diff = (t2[["mean_zone_1", "mean_zone_2"]].mean(axis=1)
             - t2[["mean_zone_7", "mean_zone_8"]].mean(axis=1))
    diff = diff.rename("layer_diff")
    diff = diff[~diff.index.duplicated()].dropna()
    return diff


def panel_B(ax) -> None:
    """Cross-pipeline scatter on Supp T2 q<0.05 subset (~3,200 genes)."""
    diff = _load_t2_q005()
    df = pd.read_parquet(PER_GRAD)
    cohort = df[(df["scope"] == "cohort") & (df["donor"].isna())
                 & (df["panel"] == "relaxed")
                 & (df["axis"] == "s")][["gene", "slope"]].drop_duplicates("gene")
    common = sorted(set(cohort["gene"]) & set(diff.index))
    slope_map = cohort.set_index("gene")["slope"]
    x = slope_map.reindex(common).to_numpy()
    y = diff.reindex(common).to_numpy()
    valid = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    x, y = x[valid], y[valid]
    common = [g for g, v in zip(common, valid) if v]
    is_marker = np.array([g in MARKERS for g in common])

    ax.axhline(0, color=GRID_GRAY, lw=0.6, zorder=0)
    ax.axvline(0, color=GRID_GRAY, lw=0.6, zorder=0)

    # X-axis: include canonical markers' x range with extra label padding
    marker_xs = x[is_marker]
    if len(marker_xs):
        x_min = min(np.percentile(x, 2), float(marker_xs.min()))
        x_max = max(np.percentile(x, 98), float(marker_xs.max()))
    else:
        x_min, x_max = np.percentile(x, [2, 98])
    pad_x = 0.20 * (x_max - x_min)
    x_lim = (x_min - pad_x, x_max + pad_x)
    # Y-axis: covers the full data range (including the central-marker
    # outliers up to ~8e-3). Symlog handles the dynamic range gracefully
    # below.
    y_data_min = float(np.nanmin(y))
    y_data_max = float(np.nanmax(y))
    y_lim = (y_data_min * 1.4, y_data_max * 1.4)

    # Brighter agreement quadrants
    ax.add_patch(mpatches.Rectangle((0, 0), x_lim[1], y_lim[1],
                                      facecolor=TEAL_LHD, alpha=0.12, zorder=0))
    ax.add_patch(mpatches.Rectangle((x_lim[0], y_lim[0]), -x_lim[0], -y_lim[0],
                                      facecolor=TEAL_LHD, alpha=0.12, zorder=0))

    # Non-markers first
    ax.scatter(x[~is_marker], y[~is_marker], color=LIGHT_GRAY, s=4,
                alpha=0.30, zorder=1, edgecolor="none")
    # Markers — large, with leader-line labels
    for i in np.where(is_marker)[0]:
        c = (RED_CENTRAL if common[i] in MARKERS_CENTRAL
             else BLUE_PORTAL)
        ax.scatter(x[i], y[i], color=c, s=48, zorder=4,
                    edgecolor=CHARCOAL, linewidths=0.6)

    # Stagger marker labels with fixed pixel offsets (clip-safe).
    # Stagger central markers vertically so labels don't pile up.
    central_in_panel = [i for i in np.where(is_marker)[0]
                        if common[i] in MARKERS_CENTRAL]
    portal_in_panel = [i for i in np.where(is_marker)[0]
                        if common[i] in MARKERS_PORTAL]
    # Central markers: CYP2E1 sits at top-right corner; CYP3A4 mid-right;
    # GLUL+CYP1A2 collide near the bottom-right, so split them vertically
    # (one labeled above the point, one below).
    central_label_map = {
        "CYP2E1": (16, 4),    # top-right cluster: label to the right
        "CYP3A4": (16, 4),    # mid-right
        "CYP1A2": (12, 14),   # near GLUL — push UP and right
        "GLUL":   (-26, -16), # near CYP1A2 — push DOWN and left
    }
    for i in central_in_panel:
        gene = common[i]
        c = RED_CENTRAL
        ox, oy = central_label_map.get(gene, (16, 4))
        ax.annotate(gene, xy=(x[i], y[i]), xytext=(ox, oy),
                     textcoords="offset points",
                     fontsize=6.5, color=c, fontstyle="italic",
                     ha="left" if ox > 0 else "right", va="center",
                     arrowprops=dict(arrowstyle="-", color=c, lw=0.4,
                                       shrinkA=2, shrinkB=2))
    # Portal markers — explicit per-gene offsets. ASS1 and CYP2A6 share
    # exactly the same y=-4.6e-4 with x within 0.07 of each other; they
    # have to split vertically (one label above, one below). HAL and SDS
    # sit close to zero (linear region) and need similar treatment.
    portal_label_map = {
        "SDS":    (-22, 14),   # at slope -1.36, y -1e-4 — upper-left
        "HAL":    (-22, -14),  # at slope -1.16, y -8e-5 — lower-left
        "ASS1":   (-30, 18),   # at slope -1.20, y -4.6e-4 — upper-left
        "CYP2A6": (-30, -18),  # at slope -1.13, y -4.6e-4 — lower-left
        "CPS1":   (-32, 4),    # at slope -0.56, y +1.3e-4 — left (only rightmost portal)
    }
    for i in portal_in_panel:
        gene = common[i]
        c = BLUE_PORTAL
        ox, oy = portal_label_map.get(gene, (-30, 0))
        ax.annotate(gene, xy=(x[i], y[i]), xytext=(ox, oy),
                     textcoords="offset points",
                     fontsize=6.5, color=c, fontstyle="italic",
                     ha="right", va="center",
                     arrowprops=dict(arrowstyle="-", color=c, lw=0.4,
                                       shrinkA=2, shrinkB=2))

    cp = json.loads(CROSS_PIPE.read_text())
    sa_b_hc = cp["approach_b_from_B11"]["high_confidence_q_lt_005"]["sign_agreement_pct"]
    rho_a = cp["approach_a_from_B12"]["signed_magnitude_spearman"]
    n_compared = len(x)
    n_agree = int(((np.sign(x) == np.sign(y)) & (x != 0) & (y != 0)).sum())
    pct = 100.0 * n_agree / max(1, n_compared)
    annot_box(ax,
                f"q<0.05 Supp T2 ∩ relaxed panel · n = {n_compared:,}\n"
                f"sign agreement {pct:.1f}%  "
                f"(B11 high-conf {sa_b_hc:.1f}%)\n"
                f"signed-mag $\\rho$ = {rho_a:.2f}",
                x=0.04, y=0.96, fontsize=6.5)

    ax.set_xlabel("framework slope (this study)", fontsize=7)
    ax.set_ylabel("Yakubovsky layer-ratio\n(Supp T2: Z1-2 − Z7-8)", fontsize=7)
    # Symlog Y: bulk of genes have |layer_ratio| < 5e-4, but four central
    # markers run up to ~8e-3 (CYP2E1). On a linear scale those four
    # outliers dominate the panel and compress everything else into a
    # thin band at y=0. Linear region for |y| < 5e-4 keeps portal markers
    # readable; log region beyond compresses the central-marker tail so
    # the bulk gets visible vertical spread.
    linthresh = 5e-4
    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.6)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_yticks([-1e-3, -1e-4, 0, 1e-4, 1e-3, 5e-3])
    ax.set_yticklabels(["−10⁻³", "−10⁻⁴", "0", "10⁻⁴", "10⁻³", "5×10⁻³"])
    ax.tick_params(axis="both", labelsize=6.5)


def panel_baseline(ax) -> None:
    """Sign-of-correlation baseline scatter."""
    sb = json.loads(SIGN_BASE.read_text())
    df = pd.read_parquet(PER_GRAD)
    cohort = df[(df["scope"] == "cohort") & (df["donor"].isna())
                 & (df["n_bins"] == 30) & (df["panel"] == "strict")
                 & (df["axis"] == "s")][["gene", "slope"]].drop_duplicates("gene")
    import scanpy as sc
    print("[fig2] Loading visium for panel E...")
    adata = sc.read_h5ad(RESULTS / "visium_human.h5ad")
    s = adata.obs["s"].to_numpy()
    samples = adata.obs["sample_id"].astype(str).to_numpy()
    keep = np.isfinite(s) & np.isin(samples, [f"M{i}" for i in range(1, 9)])
    if "fibrotic_spot" in adata.obs.columns:
        keep &= ~adata.obs["fibrotic_spot"].astype(bool).to_numpy()
    sub = adata[keep]
    s_sub = sub.obs["s"].to_numpy()
    var_names = list(sub.var_names)
    panel = list(cohort["gene"])
    common = [g for g in panel if g in var_names]
    from scipy.sparse import issparse
    X = sub[:, common].X
    if issparse(X):
        X = X.toarray()
    rs = np.array([pearsonr(s_sub, X[:, i])[0] for i in range(X.shape[1])])
    slope_map = cohort.set_index("gene")["slope"]
    slopes = slope_map.reindex(common).to_numpy()

    is_marker = np.array([g in MARKERS for g in common])
    sign_disagree = ((np.sign(slopes) != np.sign(rs))
                      & (slopes != 0) & (rs != 0))

    ax.axhline(0, color=GRID_GRAY, lw=0.6, zorder=0)
    ax.axvline(0, color=GRID_GRAY, lw=0.6, zorder=0)
    xlim = (np.nanmin(slopes) - 0.5, np.nanmax(slopes) + 0.5)
    ylim = (np.nanmin(rs) - 0.05, np.nanmax(rs) + 0.05)
    ax.add_patch(mpatches.Rectangle((0, 0), xlim[1], ylim[1],
                                      facecolor=TEAL_LHD, alpha=0.12, zorder=0))
    ax.add_patch(mpatches.Rectangle((xlim[0], ylim[0]), -xlim[0], -ylim[0],
                                      facecolor=TEAL_LHD, alpha=0.12, zorder=0))

    # Non-markers gray
    not_marker_mask = ~is_marker & ~sign_disagree
    ax.scatter(slopes[not_marker_mask], rs[not_marker_mask],
                color=LIGHT_GRAY, s=14, alpha=0.6, zorder=2,
                edgecolor="white", linewidths=0.3)
    # Sign-disagreers: orange ring (explained in caption)
    sign_only = sign_disagree & ~is_marker
    ax.scatter(slopes[sign_only], rs[sign_only], facecolor="#EEEEEE",
                edgecolor=CHARCOAL, s=30, linewidths=0.8, zorder=3)
    for i in np.where(sign_only)[0]:
        ax.annotate(common[i], xy=(slopes[i], rs[i]), xytext=(5, 5),
                     textcoords="offset points", fontsize=5.5,
                     fontstyle="italic", color=CHARCOAL)
    # Markers — staggered fixed-pixel offsets to prevent collisions
    central_offsets = [(20, 22), (28, 8), (22, -8), (28, -22)]
    portal_offsets = [(-22, -18), (-30, -2), (-22, 14), (-30, 0), (-30, 14)]
    central_in = [i for i in np.where(is_marker)[0]
                   if common[i] in MARKERS_CENTRAL]
    portal_in = [i for i in np.where(is_marker)[0]
                  if common[i] in MARKERS_PORTAL]
    for k, i in enumerate(central_in):
        gene = common[i]
        c = RED_CENTRAL
        ax.scatter(slopes[i], rs[i], color=c, s=48, zorder=4,
                    edgecolor=CHARCOAL, linewidths=0.6)
        ox, oy = central_offsets[k % len(central_offsets)]
        ax.annotate(gene, xy=(slopes[i], rs[i]), xytext=(ox, oy),
                     textcoords="offset points",
                     fontsize=6.5, color=c, fontstyle="italic",
                     ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", color=c, lw=0.4,
                                       shrinkA=2, shrinkB=2))
    for k, i in enumerate(portal_in):
        gene = common[i]
        c = BLUE_PORTAL
        ax.scatter(slopes[i], rs[i], color=c, s=48, zorder=4,
                    edgecolor=CHARCOAL, linewidths=0.6)
        ox, oy = portal_offsets[k % len(portal_offsets)]
        ax.annotate(gene, xy=(slopes[i], rs[i]), xytext=(ox, oy),
                     textcoords="offset points",
                     fontsize=6.5, color=c, fontstyle="italic",
                     ha="right", va="center",
                     arrowprops=dict(arrowstyle="-", color=c, lw=0.4,
                                       shrinkA=2, shrinkB=2))

    pct = sb["strict_panel_sign_agreement_pct"]
    rho_mag = sb["magnitude_spearman_abs_r_vs_abs_slope"]
    annot_box(ax,
                f"strict-66 panel\n63/66 ({pct:.1f}%) sign agree\n"
                f"signed-mag $\\rho$ = {rho_mag:.2f}\n"
                f"○ = sign-disagreer",
                x=0.04, y=0.96, fontsize=6.5)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("framework slope", fontsize=7)
    ax.set_ylabel("Pearson $r(g, s)$", fontsize=7)
    ax.tick_params(axis="both", labelsize=6.5)


def main() -> None:
    # 3-row layout. Row 0: cross-platform M2 + M6 (Panel A, two sub-axes
    # at half-width each). Row 1: cross-pipeline scatter (Panel B), full
    # width — the dense corner clusters of portal and central markers
    # need horizontal room for label placement. Row 2: sign-of-correlation
    # baseline (Panel C), full width to keep the marker scatter readable.
    # Old C (pooled bar restating headline numbers) and old D (sensitivity
    # sweep, flat 100% lines) dropped — both belong in the caption.
    fig = plt.figure(figsize=(mm(190), mm(200)))
    gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.30,
                   height_ratios=[1.0, 1.15, 1.0],
                   left=0.08, right=0.97, bottom=0.07, top=0.95)
    ax_A1 = fig.add_subplot(gs[0, 0])
    ax_A2 = fig.add_subplot(gs[0, 1])
    ax_B = fig.add_subplot(gs[1, 0:2])
    ax_C = fig.add_subplot(gs[2, 0:2])

    panel_A(ax_A1, ax_A2)
    panel_B(ax_B)
    panel_baseline(ax_C)

    panel_label(ax_A1, "A", x=-0.14, y=1.04)
    panel_label(ax_B, "B", x=-0.06, y=1.04)
    panel_label(ax_C, "C", x=-0.06, y=1.04)

    out = FIGURES / "fig2_multimodal.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] Wrote {out}")


if __name__ == "__main__":
    main()
