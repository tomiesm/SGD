"""Figure 1 — Theoretical framework + axis validation (revised v2)."""

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
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.sparse import issparse

from sgd.config import DATA, FIGURES, RESULTS
from sgd.plotstyle import (BLUE_PORTAL, CHARCOAL, GRID_GRAY, LIGHT_GRAY,
                           RED_CENTRAL, TEAL_LHD, TEAL_LHD_DARK, TIER_COLORS,
                           apply_style, marker_color, mm, panel_label)

warnings.filterwarnings("ignore")
apply_style()

VISIUM = RESULTS / "visium_human.h5ad"
TIER0 = RESULTS / "tier0_scoreboard.json"
TIER1 = RESULTS / "tier1_leave_family.json"
CROSS = RESULTS / "cross_donor_correlation.json"
VARIANT_B = RESULTS / "stage_B_variant_B_addendum.json"

MARKERS = ("SDS", "ASS1", "CYP2A6", "HAL", "CPS1",
            "CYP2E1", "CYP1A2", "GLUL", "CYP3A4")
N_BINS = 30


# ------- shared data loaders -------

def load_marker_bin_means() -> dict[str, dict]:
    print("[fig1] Loading visium for marker binning...")
    adata = sc.read_h5ad(VISIUM)
    s = adata.obs["s"].to_numpy()
    samples = adata.obs["sample_id"].astype(str).to_numpy()
    keep = np.isfinite(s) & np.isin(samples, [f"M{i}" for i in range(1, 9)])
    if "fibrotic_spot" in adata.obs.columns:
        keep &= ~adata.obs["fibrotic_spot"].astype(bool).to_numpy()
    sub = adata[keep].copy()
    s_sub = sub.obs["s"].to_numpy()
    edges = np.linspace(0, 1, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    var = list(sub.var_names)
    X = sub.X
    if issparse(X):
        X = X.toarray()
    out: dict[str, dict] = {}
    for g in MARKERS:
        if g not in var:
            continue
        gi = var.index(g)
        col = X[:, gi]
        means = np.full(N_BINS, np.nan)
        ci_lo = np.full(N_BINS, np.nan)
        ci_hi = np.full(N_BINS, np.nan)
        for i in range(N_BINS):
            m = (s_sub >= edges[i]) & (s_sub < edges[i + 1])
            if i == N_BINS - 1:
                m |= (s_sub == 1.0)
            if m.sum() < 10:
                continue
            xs = col[m]
            means[i] = xs.mean()
            se = xs.std(ddof=1) / np.sqrt(len(xs))
            ci_lo[i] = means[i] - 1.96 * se
            ci_hi[i] = means[i] + 1.96 * se
        out[g] = {"centers": centers, "means": means,
                   "ci_lo": ci_lo, "ci_hi": ci_hi}
    return out


# ------- Panel A: real-tissue triptych on M4 -------

REPRESENTATIVE_DONOR = "M4"
HE_IMAGE = (DATA / "visium_extracted" / "spatial"
              / f"{REPRESENTATIVE_DONOR}_tissue_hires_image.png")
SCALEFACTORS = (DATA / "visium_extracted" / "spatial"
                  / f"{REPRESENTATIVE_DONOR}_scalefactors_json.json")


def _load_panel_A_data(donor: str = REPRESENTATIVE_DONOR) -> dict:
    """Load spatial coords + s + GLUL for one representative LHD donor."""
    a = sc.read_h5ad(VISIUM)
    samples = a.obs["sample_id"].astype(str).to_numpy()
    s_all = a.obs["s"].to_numpy()
    fib = (a.obs["fibrotic_spot"].astype(bool).to_numpy()
            if "fibrotic_spot" in a.obs.columns
            else np.zeros(a.n_obs, dtype=bool))
    keep = (samples == donor) & ~fib & np.isfinite(s_all)
    sp = a.obsm["spatial"][keep]
    s = s_all[keep]
    var = list(a.var_names)
    glul_i = var.index("GLUL")
    X = a.X[keep][:, glul_i]
    if issparse(X):
        X = X.toarray().ravel()
    glul = np.asarray(X).ravel()
    # H&E hires image + scalefactor for fullres → hires-pixel conversion.
    he_img = None
    he_scale = 1.0
    if HE_IMAGE.exists() and SCALEFACTORS.exists():
        from PIL import Image
        he_img = np.array(Image.open(HE_IMAGE))
        he_scale = json.loads(SCALEFACTORS.read_text())["tissue_hires_scalef"]
    return {"sp": sp, "s": s, "glul": glul, "donor": donor,
            "he_img": he_img, "he_scale": he_scale}


def panel_A(fig, gs_outer) -> None:
    """Three side-by-side sub-panels on a representative LHD donor:
    (1) anatomy with portal/central anchors, (2) framework axis s,
    (3) GLUL (canonical pericentral marker). Designed to span the full
    figure row so each sub-panel has near-square aspect."""
    inner = gs_outer.subgridspec(1, 3, wspace=0.18)
    data = _load_panel_A_data()
    sp, s, glul, donor = data["sp"], data["s"], data["glul"], data["donor"]
    he_img, he_scale = data["he_img"], data["he_scale"]
    spot_size = 3.2

    # Sub-panel 1: histology underlay (real H&E) with the spot grid faintly
    # overlaid so the reader sees both lobular geometry and where Visium
    # measurements sit. Adds tissue context that the s and GLUL panels can't
    # show on their own.
    ax1 = fig.add_subplot(inner[0])
    if he_img is not None:
        # imshow with extent in fullres-pixel coords so spots align directly.
        h, w = he_img.shape[:2]
        ax1.imshow(he_img, extent=(0, w / he_scale, h / he_scale, 0),
                    interpolation="bilinear", zorder=0)
        ax1.scatter(sp[:, 0], sp[:, 1], s=spot_size * 0.6,
                     facecolors="none", edgecolors="#222222",
                     linewidths=0.18, alpha=0.55, zorder=1)
        ax1.set_xlim(sp[:, 0].min() - 200, sp[:, 0].max() + 200)
        ax1.set_ylim(sp[:, 1].max() + 200, sp[:, 1].min() - 200)
    else:
        # Fallback: UMI-density coloring if H&E missing
        ax1.scatter(sp[:, 0], sp[:, 1], s=spot_size, color="#888888",
                     edgecolor="none", linewidths=0)
        ax1.invert_yaxis()
    ax1.set_aspect("equal"); ax1.set_axis_off()
    ax1.set_title(f"histology (H&E), donor {donor}", fontsize=7, pad=2)

    # Sub-panel 2: framework axis s — sequential blue→red diverging map
    # (meaningful midpoint at s=0.5)
    ax2 = fig.add_subplot(inner[1])
    s_cmap = LinearSegmentedColormap.from_list(
        "portal_central", [BLUE_PORTAL, "#9090C0", "#C09090", RED_CENTRAL])
    sc2 = ax2.scatter(sp[:, 0], sp[:, 1], c=s, s=spot_size, cmap=s_cmap,
                       vmin=0, vmax=1, edgecolor="none", linewidths=0)
    ax2.set_aspect("equal"); ax2.invert_yaxis(); ax2.set_axis_off()
    ax2.set_title("framework axis (s = 0 portal, s = 1 central)",
                   fontsize=7, pad=2)
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.040, pad=0.02, aspect=20,
                        shrink=0.7)
    cb2.set_ticks([0, 0.5, 1.0])
    cb2.ax.tick_params(labelsize=5.5)
    cb2.ax.set_yticklabels(["0", "0.5", "1"])
    cb2.outline.set_linewidth(0.4)

    # Sub-panel 3: GLUL — canonical pericentral marker. White → pink → red
    # sequential, matching the central-marker convention from the figure
    # plan. vmax pinned at the 99th percentile so the within-bulk gradient
    # is preserved; capping at 95th was too aggressive (everything central
    # saturated) and capping at the maximum left the bulk pale.
    ax3 = fig.add_subplot(inner[2])
    glul_cmap = LinearSegmentedColormap.from_list(
        "white_red_central",
        ["#FFFFFF", "#F4D6D6", "#E59A9A", "#C45E5E", RED_CENTRAL])
    glul_top = float(np.quantile(glul, 0.99))
    sc3 = ax3.scatter(sp[:, 0], sp[:, 1], c=glul, s=spot_size, cmap=glul_cmap,
                       vmin=0, vmax=glul_top, edgecolor="none", linewidths=0)
    ax3.set_aspect("equal"); ax3.invert_yaxis(); ax3.set_axis_off()
    ax3.set_title("GLUL", fontsize=7, pad=2, fontstyle="italic")
    cb3 = fig.colorbar(sc3, ax=ax3, fraction=0.040, pad=0.02, aspect=20,
                        shrink=0.7)
    cb3.ax.tick_params(labelsize=5.5)
    cb3.set_label("log-norm", fontsize=5.5, labelpad=2)
    cb3.outline.set_linewidth(0.4)


# ------- Panel B: equations -------

def panel_B(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Top equation — smaller, lighter
    ax.text(0.5, 0.86,
             r"$\dfrac{\partial g}{\partial t} + v(s)\,\dfrac{\partial g}{\partial s} = F(g)$",
             ha="center", va="center", fontsize=9, color=CHARCOAL)
    # Down arrow + label
    ax.annotate("", xy=(0.5, 0.48), xytext=(0.5, 0.66),
                 arrowprops=dict(arrowstyle="->", color=CHARCOAL, lw=0.9))
    ax.text(0.56, 0.57, "steady state\n($\\partial g / \\partial t \\approx 0$)",
             ha="left", va="center", fontsize=6, color=LIGHT_GRAY,
             style="italic")
    # Bottom equation — bold but moderate size
    ax.text(0.5, 0.30,
             r"$F(g) = v(s)\,\dfrac{dg}{ds}$",
             ha="center", va="center", fontsize=11, fontweight="bold",
             color=CHARCOAL)
    ax.text(0.5, 0.08,
             "regulatory rate = velocity × spatial gradient",
             ha="center", va="center", fontsize=6.5, color=CHARCOAL,
             style="italic")


# ------- Panel C: pipeline schematic -------

def panel_C(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Two-line box labels with proper padding; arrow labels above each arrow.
    boxes = [
        ("Visium\ntissue", 0.085),
        ("score-\naxis", 0.290),
        ("quantile\nbin", 0.495),
        ("per-gene\nslope", 0.700),
        ("compare\ngradients", 0.905),
    ]
    arrows = ["axis", "binning", "slope", "compare"]
    box_w, box_h = 0.155, 0.55
    y_box = 0.16
    for label, cx in boxes:
        rect = mpatches.FancyBboxPatch(
            (cx - box_w / 2, y_box), box_w, box_h,
            boxstyle="round,pad=0.018", linewidth=0.8,
            facecolor="white", edgecolor=CHARCOAL,
        )
        ax.add_patch(rect)
        ax.text(cx, y_box + box_h / 2, label, ha="center", va="center",
                 fontsize=7, color=CHARCOAL, linespacing=1.05)
    for i, arr in enumerate(arrows):
        x0 = boxes[i][1] + box_w / 2
        x1 = boxes[i + 1][1] - box_w / 2
        ax.annotate("", xy=(x1, y_box + box_h / 2),
                    xytext=(x0, y_box + box_h / 2),
                    arrowprops=dict(arrowstyle="->", color=CHARCOAL, lw=0.9))
        ax.text((x0 + x1) / 2, y_box + box_h + 0.05, arr,
                 ha="center", va="bottom", fontsize=6.5, color=CHARCOAL,
                 style="italic")


# ------- Panel D: marker scoreboard (3×3 grid + recovery matrix) -------

def panel_D_left(fig, gs_outer, bin_data: dict) -> None:
    """3×3 grid of per-marker bin-mean plots. Y-label only on left column,
    x-label only on bottom row."""
    inner = gs_outer.subgridspec(3, 3, hspace=0.55, wspace=0.30)
    # Find global y range per column for shared scale within column
    for k, gene in enumerate(MARKERS):
        row, col = k // 3, k % 3
        ax = fig.add_subplot(inner[row, col])
        d = bin_data.get(gene)
        if d is None:
            ax.text(0.5, 0.5, f"{gene}\nN/A", ha="center", va="center")
            ax.set_axis_off()
            continue
        c = marker_color(gene)
        ax.fill_between(d["centers"], d["ci_lo"], d["ci_hi"], color=c,
                         alpha=0.30, linewidth=0)
        ax.plot(d["centers"], d["means"], color=c, lw=1.6)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.tick_params(axis="both", labelsize=6, length=2)
        ax.grid(False)
        # Title: gene name italic
        ax.set_title(gene, fontsize=7.5, fontstyle="italic", pad=2)
        # Y-label only on leftmost column
        if col == 0:
            ax.set_ylabel("log-norm", fontsize=6.5)
        else:
            ax.set_ylabel("")
        # X tick labels only on bottom row
        if row == 2:
            ax.set_xlabel("s", fontsize=6.5)
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def panel_D_right(ax) -> None:
    """Per-marker × per-donor recovery matrix. White background; full color
    saturation for correct, grayed for incorrect."""
    d = json.loads(TIER0.read_text())
    per = d["per_donor_marker_correct"]
    donors = sorted(per.keys())
    M = np.array([[per[s].get(g, 0) for g in MARKERS] for s in donors])

    ax.set_facecolor("white")
    # Render each cell explicitly with edge boundaries
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            correct = M[i, j] == 1
            face = "white" if correct else "#F1F1F1"
            ax.add_patch(mpatches.Rectangle((j, i), 1, 1, facecolor=face,
                                              edgecolor="#CCCCCC", lw=0.5))
            if correct:
                ax.plot(j + 0.5, i + 0.5, "o",
                         color=marker_color(MARKERS[j]),
                         markersize=8,
                         markeredgecolor=CHARCOAL,
                         markeredgewidth=0.5)
            else:
                ax.plot(j + 0.5, i + 0.5, "x", color=LIGHT_GRAY,
                         markersize=6, mew=0.8)
    ax.set_xlim(0, M.shape[1])
    ax.set_ylim(M.shape[0], 0)  # invert so M1 is at top
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(M.shape[1]) + 0.5)
    ax.set_xticklabels(MARKERS, rotation=55, fontsize=6.5,
                        fontstyle="italic", ha="right")
    ax.set_yticks(np.arange(M.shape[0]) + 0.5)
    ax.set_yticklabels(donors, fontsize=6.5)
    ax.tick_params(axis="both", length=0)
    ax.grid(False)
    # Hide spines
    for sp in ax.spines.values():
        sp.set_visible(False)
    # Right-side single header rather than per-row repetition
    ax.text(M.shape[1] + 0.4, M.shape[0] / 2, "all 9 ✓",
             rotation=270, fontsize=7, color=TEAL_LHD_DARK,
             va="center", ha="center", fontweight="bold")


# ------- Panel E: tier framing -------

def panel_E(ax) -> None:
    tier1 = json.loads(TIER1.read_text())
    cyp_n = tier1["drop_CYP_family"].get("n_correct_pooled", 0)
    cyp_total = tier1["drop_CYP_family"].get("n_evaluated_pooled", 1)
    urea_n = tier1["drop_urea_cycle"].get("n_correct_pooled", 0)
    urea_total = tier1["drop_urea_cycle"].get("n_evaluated_pooled", 1)
    glul_n = tier1["drop_GLUL_alone"].get("n_correct_pooled", 0)
    glul_total = tier1["drop_GLUL_alone"].get("n_evaluated_pooled", 1)
    cross = json.loads(CROSS.read_text())
    mean_r = cross["off_diagonal_mean"]
    min_r = cross["off_diagonal_min"]
    var_b = json.loads(VARIANT_B.read_text())
    n_var_b = sum(1 for g, n in var_b["marker_recovery"]["per_marker_donor_count"].items()
                    if n > 0)

    # Layout: each tier as one row; Tier 1 broken into 3 sub-bars
    rows = [
        {"name": "Tier 0\nmarker scoreboard", "kind": "single",
          "value": 1.0, "annot": "9/9 markers", "color": TIER_COLORS[0]},
        {"name": "Tier 1\nleave-family-out", "kind": "split",
          "values": [(cyp_n / cyp_total, f"{cyp_n}/{cyp_total} CYP"),
                      (urea_n / urea_total, f"{urea_n}/{urea_total} urea"),
                      (glul_n / glul_total, f"{glul_n}/{glul_total} GLUL")],
          "color": TIER_COLORS[1]},
        {"name": "Tier 2\nindependent axis", "kind": "single",
          "value": n_var_b / 9.0,
          "annot": f"{n_var_b}/9 (CPS1 ambiguous in reference panel)",
          "color": TIER_COLORS[2]},
        {"name": "Tier 3\ncross-donor r", "kind": "single",
          "value": min_r,
          "annot": f"mean {mean_r:.2f}, min {min_r:.2f}",
          "color": TIER_COLORS[3]},
    ]

    n_rows = len(rows)
    y_positions = []
    label_positions = []
    cur_y = 0
    bar_h = 0.6
    sub_gap = 0.05
    sub_h = 0.18
    for r in rows:
        if r["kind"] == "single":
            ax.barh(cur_y, r["value"], height=bar_h, color=r["color"],
                     edgecolor=CHARCOAL, linewidth=0.6)
            ax.text(min(r["value"] + 0.012, 1.01), cur_y, r["annot"],
                     va="center", ha="left", fontsize=6.5, color=CHARCOAL)
            y_positions.append(cur_y)
            label_positions.append(cur_y)
            cur_y += 1.0
        else:
            # Tier 1: 3 sub-bars stacked vertically
            sub_top = cur_y - 0.30
            for i, (val, ann) in enumerate(r["values"]):
                yy = sub_top + i * (sub_h + sub_gap)
                ax.barh(yy, val, height=sub_h, color=r["color"],
                         edgecolor=CHARCOAL, linewidth=0.5)
                ax.text(min(val + 0.012, 1.01), yy, ann,
                         va="center", ha="left", fontsize=6.0, color=CHARCOAL)
            label_positions.append(sub_top + (sub_h + sub_gap))
            cur_y += 1.0
    ax.set_yticks(label_positions)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=7)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.5, 0.85, 1.0])
    ax.set_xticklabels(["0", "0.5", "0.85\nthreshold", "1.0"], fontsize=6.5)
    ax.set_xlabel("score / fraction", fontsize=7)
    ax.axvline(0.85, color=LIGHT_GRAY, lw=0.6, linestyle=":", zorder=0)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)


# ------- Panel F: cross-donor correlation matrix -------

def panel_F(ax) -> None:
    d = json.loads(CROSS.read_text())
    M = np.array(d["cross_donor_correlation_matrix"])
    donors = d["donors"]
    # Render as full symmetric matrix (information-complete and visually balanced).
    cmap = LinearSegmentedColormap.from_list(
        "tealwhite", ["#FFFFFF", "#A2D2D2", TEAL_LHD, TEAL_LHD_DARK]
    )
    M_disp = M.copy()
    # Mask diagonal
    diag_mask = np.eye(len(donors), dtype=bool)
    M_disp[diag_mask] = np.nan
    im = ax.imshow(M_disp, cmap=cmap, vmin=0.7, vmax=1.0, aspect="equal",
                    interpolation="nearest")
    for i in range(len(donors)):
        for j in range(len(donors)):
            if i == j:
                # Diagonal: gray, no text
                ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                  facecolor=GRID_GRAY,
                                                  edgecolor="white", lw=0.5))
                continue
            v = M[i, j]
            color = "white" if v > 0.93 else CHARCOAL
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                     color=color)
    ax.set_xticks(range(len(donors)))
    ax.set_xticklabels(donors, fontsize=6.5)
    ax.set_yticks(range(len(donors)))
    ax.set_yticklabels(donors, fontsize=6.5)
    ax.tick_params(axis="both", length=0)
    ax.grid(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, aspect=14)
    cbar.set_label("Pearson r", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.axhline(y=0.85, color=CHARCOAL, lw=0.6, linestyle=":")
    cbar.ax.axhline(y=0.70, color=CHARCOAL, lw=0.6, linestyle=":")
    mean_r = d["off_diagonal_mean"]
    min_r = d["off_diagonal_min"]
    ax.set_title(f"mean r = {mean_r:.3f}, min r = {min_r:.3f} (M3↔M7)",
                  fontsize=7.5, color=CHARCOAL, pad=6)


# ------- main -------

def main() -> None:
    bin_data = load_marker_bin_means()
    fig = plt.figure(figsize=(mm(180), mm(225)))
    # 4-row layout: row 0 hosts the real-tissue triptych (Panel A) spanning
    # the full width. Rows 1-3 carry B+C, D, and E+F as before.
    gs = GridSpec(4, 6, figure=fig, hspace=0.40, wspace=0.55,
                   height_ratios=[1.65, 0.95, 1.45, 1.45],
                   width_ratios=[1, 1, 1.30, 1, 1, 1],
                   left=0.06, right=0.97, bottom=0.05, top=0.97)

    # Row 0: A (full 6 cols, real-tissue triptych)
    panel_A(fig, gs[0, 0:6])
    ax_label_A = fig.add_subplot(gs[0, 0:6])
    ax_label_A.set_axis_off()
    ax_label_A.set_zorder(-1)
    panel_label(ax_label_A, "A", x=-0.01, y=1.0)

    # Row 1: B (cols 0-2, equations) | C (cols 3-5, pipeline)
    ax_B = fig.add_subplot(gs[1, 0:3])
    ax_C = fig.add_subplot(gs[1, 3:6])
    panel_B(ax_B); panel_label(ax_B, "B", x=-0.02, y=1.02)
    panel_C(ax_C); panel_label(ax_C, "C", x=-0.02, y=1.02)

    # Row 2: D-left (3×3 marker grid cols 0-2) | D-right (matrix cols 3-5)
    panel_D_left(fig, gs[2, 0:3], bin_data)
    ax_label_D = fig.add_subplot(gs[2, 0:3])
    ax_label_D.set_axis_off()
    panel_label(ax_label_D, "D", x=-0.025, y=1.02)
    ax_Dr = fig.add_subplot(gs[2, 3:6])
    panel_D_right(ax_Dr)

    # Row 3: E (cols 0-2) | F (cols 3-5)
    ax_E = fig.add_subplot(gs[3, 0:3])
    ax_F = fig.add_subplot(gs[3, 3:6])
    panel_E(ax_E); panel_label(ax_E, "E", x=-0.07, y=1.02)
    panel_F(ax_F); panel_label(ax_F, "F", x=-0.06, y=1.08)

    out = FIGURES / "fig1_axis_validation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] Wrote {out}")


if __name__ == "__main__":
    main()
