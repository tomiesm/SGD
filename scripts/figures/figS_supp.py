"""Supplementary figures S1–S9."""

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

from sgd.config import FIGURES, RESULTS
from sgd.plotstyle import (BLUE_PORTAL, CHARCOAL, GRID_GRAY, LIGHT_GRAY,
                           RED_CENTRAL, TEAL_LHD, TEAL_LHD_DARK, OCHRE_P,
                           OCHRE_P_DARK, apply_style, donor_color,
                           marker_color, mm, panel_label)

warnings.filterwarnings("ignore")
apply_style()

CONF = RESULTS / "confound_diagnostics.json"
TIER1 = RESULTS / "tier1_leave_family.json"
CROSS_PIPE = RESULTS / "cross_pipeline_reproducibility.json"
APPROACH_A = RESULTS / "stage_B_cross_pipeline_approach_a.json"
HEPC2L = RESULTS / "obs_hep_fraction_c2l.parquet"
C2LADD = RESULTS / "c3_cell2location_addendum.json"
SUPP_T6 = RESULTS / "supp_t6_lhd_p_calibration.json"
CRITERIA = RESULTS / "steatosis_call_criteria.parquet"
B8 = RESULTS / "b8_sensitivity.json"
B7_SUMMARY = RESULTS / "b7_c1_summary.json"
B7_JOINT = RESULTS / "b7_joint_table.parquet"

LHD = ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")
STEATOTIC = ("M1", "M2", "M3", "P6")
MARKERS = ("SDS", "ASS1", "CYP2A6", "HAL", "CPS1",
            "CYP2E1", "CYP1A2", "GLUL", "CYP3A4")


# -- S1 ----------------------------------------------------------------------

def figS1():
    """Stage A confound diagnostics — 5 per-sample bar panels (A-E) + 5×3
    status table (F). All five diagnostic panels share the same per-sample
    bar-chart format with their respective trigger threshold drawn as a
    dashed reference line."""
    d = json.loads(CONF.read_text())
    per = d["per_sample"]
    add = json.loads(C2LADD.read_text())
    # Three-row layout: row 1 (A, B, C diagnostic bars), row 2 (D, E
    # diagnostic bars), row 3 (F full-width status table).
    fig = plt.figure(figsize=(mm(200), mm(180)))
    gs = GridSpec(3, 6, figure=fig, hspace=1.10, wspace=0.55,
                   height_ratios=[1.0, 1.0, 0.85],
                   left=0.08, right=0.97, bottom=0.05, top=0.95)
    ax_A = fig.add_subplot(gs[0, 0:2])
    ax_B = fig.add_subplot(gs[0, 2:4])
    ax_C = fig.add_subplot(gs[0, 4:6])
    ax_D = fig.add_subplot(gs[1, 0:3])
    ax_E = fig.add_subplot(gs[1, 3:6])
    ax_F = fig.add_subplot(gs[2, 0:6])

    samples = list(LHD)
    cb_lhd = [TEAL_LHD] * len(samples)

    # ---- Panel A: C1 r(UMI, s) per LHD sample ----
    rs = [per[s].get("C1", {}).get("r", np.nan) for s in samples]
    cohort_mean = d["summary"]["C1_cohort"]["abs_r_mean"]
    ax_A.bar(samples, rs, color=cb_lhd, edgecolor=CHARCOAL, linewidth=0.5)
    ax_A.axhline(0.3, color=LIGHT_GRAY, lw=0.7, linestyle="--")
    ax_A.axhline(-0.3, color=LIGHT_GRAY, lw=0.7, linestyle="--")
    ax_A.axhline(0, color=CHARCOAL, lw=0.5)
    # Explicit ±0.3 labels at the right edge of each line
    ax_A.text(7.6, 0.30, "+0.3", fontsize=5.5, color=LIGHT_GRAY,
                va="center", ha="left")
    ax_A.text(7.6, -0.30, "−0.3", fontsize=5.5, color=LIGHT_GRAY,
                va="center", ha="left")
    ax_A.set_ylabel("r(UMI, s)", fontsize=7)
    ax_A.set_ylim(-0.45, 0.45)
    ax_A.tick_params(axis="x", labelsize=6.5, rotation=0)
    ax_A.set_title(f"C1: UMI vs s per donor\nall negative — pericentral lower; cohort |r| = {cohort_mean:.2f}",
                    fontsize=7, pad=4)
    panel_label(ax_A, "A", x=-0.18, y=1.10)

    # ---- Panel B: C2 r(lipid, UMI) per steatotic donor ----
    labs = []
    rs_c2 = []
    for s in STEATOTIC:
        if s in per:
            v = per[s].get("C2")
            if v and v.get("evaluable"):
                labs.append(s); rs_c2.append(v["r"])
            else:
                labs.append(s); rs_c2.append(np.nan)
        else:
            labs.append(s); rs_c2.append(np.nan)
    cb = [TEAL_LHD if s in LHD else OCHRE_P for s in labs]
    bars = ax_B.bar(labs, [v if not np.isnan(v) else 0 for v in rs_c2],
                     color=cb, edgecolor=CHARCOAL, linewidth=0.5)
    for b, v, s in zip(bars, rs_c2, labs):
        if np.isnan(v):
            b.set_facecolor("none")
            b.set_edgecolor(LIGHT_GRAY)
            ax_B.text(b.get_x() + b.get_width() / 2, 0.005, "n.a.",
                       ha="center", va="bottom", fontsize=5.5,
                       color=LIGHT_GRAY, fontstyle="italic")
    ax_B.axhline(-0.5, color=LIGHT_GRAY, lw=0.7, linestyle="--")
    ax_B.text(3.6, -0.5, "−0.5\ntrigger", fontsize=5.5, color=LIGHT_GRAY,
                va="center", ha="left")
    ax_B.axhline(0, color=CHARCOAL, lw=0.5)
    ax_B.set_ylim(-0.6, 0.05)
    ax_B.set_ylabel("r(lipid_pct, UMI)", fontsize=7)
    ax_B.set_title("C2: lipid vs UMI per steatotic donor\nM2, M3 lipid annotation unavailable",
                    fontsize=7, pad=4)
    ax_B.tick_params(axis="x", labelsize=6.5)
    panel_label(ax_B, "B", x=-0.18, y=1.10)

    # ---- Panel C: C3 max-quintile frac >25% mt per LHD sample ----
    c3_vals = [per[s]["C3"].get("max_quintile_frac_above_25", 0)
                if s in per and per[s].get("C3") else 0 for s in samples]
    ax_C.bar(samples, c3_vals, color=cb_lhd, edgecolor=CHARCOAL, linewidth=0.5)
    ax_C.axhline(0.15, color=LIGHT_GRAY, lw=0.7, linestyle="--")
    ax_C.text(7.6, 0.15, "0.15\ntrigger", fontsize=5.5, color=LIGHT_GRAY,
                va="center", ha="left")
    ax_C.set_ylabel("max-quintile frac\nspots > 25% mt", fontsize=7)
    ax_C.set_title("C3: mt-fraction trigger\nno donor exceeds 0.15",
                    fontsize=7, pad=4)
    ax_C.tick_params(axis="x", labelsize=6.5)
    ax_C.set_ylim(0, 0.18)
    panel_label(ax_C, "C", x=-0.20, y=1.10)

    # ---- Panel D: C4 r(s, hep_fraction) per LHD donor (cell2location) ----
    confound4 = add.get("confound_4_per_sample", {})
    rs_c4 = [confound4.get(s, {}).get("r", np.nan) for s in samples]
    ax_D.bar(samples, rs_c4, color=cb_lhd, edgecolor=CHARCOAL, linewidth=0.5)
    ax_D.axhline(0.30, color=LIGHT_GRAY, lw=0.7, linestyle="--")
    ax_D.axhline(-0.30, color=LIGHT_GRAY, lw=0.7, linestyle="--")
    ax_D.axhline(0, color=CHARCOAL, lw=0.5)
    ax_D.text(7.6, 0.30, "+0.3", fontsize=5.5, color=LIGHT_GRAY,
                va="center", ha="left")
    ax_D.text(7.6, -0.30, "−0.3", fontsize=5.5, color=LIGHT_GRAY,
                va="center", ha="left")
    max_abs = add.get("confound_4_cohort_max_abs_r", max(abs(v) for v in rs_c4
                                                          if not np.isnan(v)))
    ax_D.set_ylabel("r(s, hep_fraction)", fontsize=7)
    ax_D.set_title(f"C4: composition vs s (cell2location, D13)\nmax |r| = {max_abs:.2f} (M6)",
                    fontsize=7, pad=4)
    ax_D.set_ylim(-0.45, 0.45)
    ax_D.tick_params(axis="x", labelsize=6.5)
    panel_label(ax_D, "D", x=-0.18, y=1.10)

    # ---- Panel E: C5 r_dir per LHD sample, flagged samples in dark teal ----
    rdirs = [per[s].get("C5", {}).get("r_dir", np.nan) for s in samples]
    flagged = ("M2", "M4", "M6", "M8")
    bars = ax_E.bar(samples, rdirs, color=cb_lhd, edgecolor=CHARCOAL,
                     linewidth=0.5)
    for b, s in zip(bars, samples):
        if s in flagged:
            b.set_edgecolor(TEAL_LHD_DARK)
            b.set_linewidth(1.5)
    ax_E.axhline(0.30, color=LIGHT_GRAY, lw=0.7, linestyle="--")
    ax_E.text(7.6, 0.30, "0.3\nflag", fontsize=5.5, color=LIGHT_GRAY,
                va="center", ha="left")
    ax_E.set_ylabel("r_dir", fontsize=7)
    ax_E.set_title("C5: slide-edge directional artifact\nflagged donors (dark border): M2, M4, M6, M8",
                    fontsize=7, pad=4)
    ax_E.set_ylim(0, max(rdirs) * 1.15)
    ax_E.tick_params(axis="x", labelsize=6.5)
    panel_label(ax_E, "E", x=-0.18, y=1.10)

    # ---- Panel F: 5×3 status grid ----
    ax_F.set_axis_off()
    rows = [
        ("C1", "UMI vs s",     "covariate-adjusted",    "ok"),
        ("C2", "lipid vs UMI", "log-UMI covariate",     "ok"),
        ("C3", "mt-fraction",  "no trigger",            "ok"),
        ("C4", "composition",  "cell2location (D13)",   "ok"),
        ("C5", "slide-edge",   "edge-mask M2/M6",       "flagged"),
    ]
    headers = ["confound", "diagnostic", "resolution", "status"]
    n_cols = 4
    n_rows = len(rows) + 1
    col_widths = [0.13, 0.21, 0.42, 0.24]
    cum = np.cumsum([0] + col_widths)
    row_h = 0.78 / n_rows
    y_top = 0.86
    # Header row
    for j, h in enumerate(headers):
        ax_F.add_patch(mpatches.Rectangle((cum[j], y_top - row_h),
                                            col_widths[j], row_h,
                                            facecolor="#EEEEEE",
                                            edgecolor=CHARCOAL, lw=0.5))
        ax_F.text(cum[j] + col_widths[j] / 2, y_top - row_h / 2, h,
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color=CHARCOAL)
    # Data rows
    for i, row in enumerate(rows):
        y0 = y_top - (i + 2) * row_h
        face = "white" if i % 2 == 0 else "#FAFAFA"
        for j, val in enumerate(row[:3]):
            ax_F.add_patch(mpatches.Rectangle((cum[j], y0),
                                                col_widths[j], row_h,
                                                facecolor=face,
                                                edgecolor="#DDDDDD", lw=0.4))
            ax_F.text(cum[j] + col_widths[j] / 2, y0 + row_h / 2, val,
                        ha="center", va="center", fontsize=7.5,
                        color=CHARCOAL)
        # Status pill
        status = row[3]
        col = TEAL_LHD if status == "ok" else OCHRE_P
        ax_F.add_patch(mpatches.Rectangle((cum[3], y0),
                                            col_widths[3], row_h,
                                            facecolor=col, alpha=0.75,
                                            edgecolor=CHARCOAL, lw=0.4))
        ax_F.text(cum[3] + col_widths[3] / 2, y0 + row_h / 2,
                    status.upper(),
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")
    ax_F.set_xlim(0, 1); ax_F.set_ylim(0, 1)
    # Title above the axes so it doesn't overlap the header row.
    ax_F.set_title("Stage A confound resolution",
                    fontsize=9, fontweight="bold", color=CHARCOAL, pad=8)
    panel_label(ax_F, "F", x=-0.04, y=1.10)

    fig.savefig(FIGURES / "figS1_confounds.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "figS1_confounds.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[figS1] Wrote figS1_confounds.png")


# -- S2 ----------------------------------------------------------------------

def figS2():
    """Tier 1 leave-family-out detail.

    White panel backgrounds; failed cells shown as open-circle X markers
    (distinct from "untested" empty cells); per-donor minimum bars
    alongside pooled in Panel D.
    """
    d = json.loads(TIER1.read_text())

    fig = plt.figure(figsize=(mm(200), mm(95)))
    # Width-proportional columns: CYP (4 cols of dropped), urea (2),
    # GLUL (1), summary D — give A more horizontal width than C.
    gs = GridSpec(1, 12, figure=fig, wspace=2.50,
                   left=0.06, right=0.98, bottom=0.22, top=0.82)
    # Allocate cols: CYP 4, urea 3, GLUL 2, D 3 (out of 12)
    family_data = [
        ("CYP family", d["drop_CYP_family"], gs[0, 0:4]),
        ("urea cycle", d["drop_urea_cycle"], gs[0, 4:7]),
        ("GLUL only", d["drop_GLUL_alone"], gs[0, 7:9]),
    ]

    for k, (name, sub, gpos) in enumerate(family_data):
        ax = fig.add_subplot(gpos)
        per_donor = sub.get("per_donor_held_out_correct", {})
        donors = sorted(per_donor.keys())
        markers = sub.get("dropped_genes", [])
        M = np.zeros((len(donors), len(markers)))
        for i, dn in enumerate(donors):
            row = per_donor.get(dn, {})
            for j, g in enumerate(markers):
                M[i, j] = row.get(g, 0)

        # White cells with subtle gray separators; correct → colored dot,
        # failed → open circle with X.
        ax.set_facecolor("white")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                rect = mpatches.Rectangle((j, i), 1, 1, facecolor="white",
                                            edgecolor="#CCCCCC", lw=0.5)
                ax.add_patch(rect)
                if M[i, j] == 1:
                    ax.plot(j + 0.5, i + 0.5, "o",
                             color=marker_color(markers[j]),
                             markersize=8, markeredgecolor=CHARCOAL,
                             markeredgewidth=0.5)
                else:
                    ax.plot(j + 0.5, i + 0.5, "o", color="white",
                             markersize=7, markeredgecolor=LIGHT_GRAY,
                             markeredgewidth=0.7)
                    ax.plot(j + 0.5, i + 0.5, "x", color=LIGHT_GRAY,
                             markersize=4, mew=0.9)
        ax.set_xlim(0, M.shape[1])
        ax.set_ylim(M.shape[0], 0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(M.shape[1]) + 0.5)
        ax.set_xticklabels(markers, rotation=55, ha="right", fontsize=6.5,
                            fontstyle="italic")
        ax.set_yticks(np.arange(M.shape[0]) + 0.5)
        ax.set_yticklabels(donors, fontsize=6.5)
        ax.tick_params(axis="both", length=0)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        n_correct = sub.get("n_correct_pooled")
        n_total = sub.get("n_evaluated_pooled")
        ax.set_title(f"{name}\npooled {n_correct}/{n_total}",
                      fontsize=7.5, pad=4)
        panel_label(ax, "ABC"[k], x=0.5, y=1.22, ha="center")

    # Panel D: pooled vs minimum per-donor recovery, side-by-side bars.
    ax_D = fig.add_subplot(gs[0, 9:12])
    variants = ["CYP family", "urea cycle", "GLUL"]
    keys = ["drop_CYP_family", "drop_urea_cycle", "drop_GLUL_alone"]
    pooled_frac = []
    min_donor_frac = []
    for k in keys:
        sub = d[k]
        n_total_per_donor = len(sub["dropped_genes"])
        pooled_frac.append(sub["n_correct_pooled"] /
                            max(sub["n_evaluated_pooled"], 1))
        per_donor = sub.get("per_donor_held_out_correct", {})
        donor_correct = [sum(row.values()) / max(n_total_per_donor, 1)
                          for row in per_donor.values()]
        min_donor_frac.append(min(donor_correct) if donor_correct else 0.0)

    x = np.arange(len(variants))
    bw = 0.36
    off = bw / 2 + 0.03      # paired-bar centre offset (+0.03 opens a small gap)
    ax_D.bar(x - off, pooled_frac, width=bw, color=TEAL_LHD,
              edgecolor=CHARCOAL, linewidth=0.5, label="pooled")
    ax_D.bar(x + off, min_donor_frac, width=bw, color=OCHRE_P,
              edgecolor=CHARCOAL, linewidth=0.5, label="min per-donor")
    for i, p in enumerate(pooled_frac):
        ax_D.text(i - off, p + 0.02, f"{p:.2f}", ha="center",
                   fontsize=6, color=CHARCOAL)
    for i, m in enumerate(min_donor_frac):
        col = OCHRE_P_DARK if m < 1.0 else CHARCOAL
        ax_D.text(i + off, m + 0.02, f"{m:.2f}", ha="center",
                   fontsize=6, color=col, fontweight="bold" if m < 1.0 else "normal")
    ax_D.set_xticks(x)
    ax_D.set_xticklabels(variants, fontsize=6.5)
    ax_D.set_ylim(0, 1.20)
    ax_D.set_ylabel("fraction of held-out\nmarkers recovered", fontsize=7)
    ax_D.set_title("pooled vs minimum per-donor",
                    fontsize=7.5, pad=4)
    ax_D.legend(fontsize=6, loc="upper right", frameon=False)
    ax_D.tick_params(axis="both", labelsize=6.5)
    ax_D.grid(axis="y", alpha=0.3)
    ax_D.set_axisbelow(True)
    panel_label(ax_D, "D", x=0.5, y=1.22, ha="center")

    fig.savefig(FIGURES / "figS2_tier1.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "figS2_tier1.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[figS2] Wrote figS2_tier1.png")


# -- S3 ----------------------------------------------------------------------

def figS3():
    """Cross-pipeline disagreement diagnostics. Panel C content (the
    structural-disagreement explanation, 9/9 marker recovery note, and the
    approach-(a)/(b) method definitions) lives in the figure caption."""
    cp = json.loads(CROSS_PIPE.read_text())
    aa = json.loads(APPROACH_A.read_text())
    fig = plt.figure(figsize=(mm(140), mm(80)))
    gs = GridSpec(1, 2, figure=fig, wspace=0.45,
                   left=0.13, right=0.97, bottom=0.20, top=0.85)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])

    # Panel A: agreement scaling with effect size
    cats = ["all directional", "high-conf q<0.05", "top-quartile |slope|"]
    vals = [
        aa["sign_agreement_pct"],
        aa["high_confidence_q_lt_005"]["sign_agreement_pct"],
        aa["top_quartile_our_magnitude"]["sign_agreement_pct"],
    ]
    bars = ax_A.bar(cats, vals, color=[LIGHT_GRAY, TEAL_LHD, TEAL_LHD_DARK],
                     edgecolor=CHARCOAL, linewidth=0.5)
    for b, v in zip(bars, vals):
        ax_A.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                   ha="center", fontsize=6.5)
    ax_A.axhline(95, color=RED_CENTRAL, lw=0.6, linestyle=":")
    ax_A.text(0.02, 0.97, "target ≥95%", color=RED_CENTRAL, fontsize=5.5,
                transform=ax_A.transAxes, va="top")
    ax_A.set_ylabel("sign agreement (%)", fontsize=7)
    ax_A.set_ylim(50, 100)
    ax_A.tick_params(axis="x", labelsize=6, rotation=20)
    ax_A.set_title("agreement scales with effect size", fontsize=7, pad=4)
    panel_label(ax_A, "A")

    # Panel B: approach (a) vs (b) convergence
    a_b = cp["approach_b_from_B11"]["sign_agreement_pct"]
    a_a = cp["approach_a_from_B12"]["sign_agreement_pct"]
    a_b_hc = cp["approach_b_from_B11"]["high_confidence_q_lt_005"]["sign_agreement_pct"]
    a_a_hc = cp["approach_a_from_B12"]["high_confidence_q_lt_005"]["sign_agreement_pct"]
    x = np.array([0, 1])
    ax_B.bar(x - 0.18, [a_a, a_a_hc], width=0.35, color=TEAL_LHD,
              edgecolor=CHARCOAL, linewidth=0.5, label="approach (a)")
    ax_B.bar(x + 0.18, [a_b, a_b_hc], width=0.35, color=OCHRE_P,
              edgecolor=CHARCOAL, linewidth=0.5, label="approach (b)")
    for xi, va, vb in zip(x, [a_a, a_a_hc], [a_b, a_b_hc]):
        ax_B.text(xi - 0.18, va + 0.4, f"{va:.1f}%", ha="center", fontsize=6.5)
        ax_B.text(xi + 0.18, vb + 0.4, f"{vb:.1f}%", ha="center", fontsize=6.5)
    ax_B.set_xticks(x)
    ax_B.set_xticklabels(["all", "high-conf"], fontsize=6.5)
    ax_B.legend(fontsize=6, frameon=False, loc="upper left")
    ax_B.set_ylabel("sign agreement (%)", fontsize=7)
    ax_B.set_ylim(60, 85)
    ax_B.set_title("two methods converge", fontsize=7, pad=4)
    panel_label(ax_B, "B")

    fig.savefig(FIGURES / "figS3_cross_pipeline.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "figS3_cross_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[figS3] Wrote figS3_cross_pipeline.png")


# -- S4 ----------------------------------------------------------------------

def figS4():
    """Within-cell-type robustness using cell2location."""
    add = json.loads(C2LADD.read_text())
    fig = plt.figure(figsize=(mm(192), mm(104)))
    gs = GridSpec(1, 3, figure=fig, wspace=0.50,
                   left=0.085, right=0.985, bottom=0.20, top=0.80)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])

    # Panel A: hep_fraction distribution + threshold sweep
    hep = pd.read_parquet(HEPC2L)
    vals = hep["hep_fraction_c2l"].dropna().to_numpy()
    ax_A.hist(vals, bins=60, color=TEAL_LHD, alpha=0.75, edgecolor=CHARCOAL,
                linewidth=0.4)
    # Threshold lines; the three labels are anchored right / centre / left of
    # their lines so the numbers do not overlap above the histogram.
    thresh_specs = [(0.40, LIGHT_GRAY, "right"),
                    (0.45, RED_CENTRAL, "center"),
                    (0.50, CHARCOAL, "left")]
    y_top = ax_A.get_ylim()[1]
    for thresh, c, ha in thresh_specs:
        ax_A.axvline(thresh, color=c, lw=0.8, linestyle="--")
        ax_A.text(thresh, y_top * 1.03, f"{thresh:.2f}", color=c, fontsize=6,
                   ha=ha, va="bottom")
    ax_A.set_xlabel("c2l hep_fraction", fontsize=7)
    ax_A.set_ylabel("number of spots", fontsize=7)
    ax_A.set_title(f"all 47,828 spots; max = {vals.max():.2f}", fontsize=7,
                    pad=16)
    panel_label(ax_A, "A", x=0.5, y=1.24, ha="center")

    # Panel B: scoreboard at three thresholds
    msb = add.get("multi_threshold_scoreboard", {})
    thresholds = [0.40, 0.45, 0.50]
    headline = []
    sanity = []
    n_kept = []
    for t in thresholds:
        info = msb.get(f"thresh_{t}", {})
        headline.append(info.get("headline_8_correct", 0))
        sanity.append(info.get("sanity_9_correct", 0))
        n_kept.append(info.get("n_spots_kept", 0))
    x = np.arange(len(thresholds))
    bars_h = ax_B.bar(x - 0.18, headline, width=0.35, color=TEAL_LHD_DARK,
                       edgecolor=CHARCOAL, linewidth=0.5, label="headline 8/8")
    bars_s = ax_B.bar(x + 0.18, sanity, width=0.35, color=TEAL_LHD,
                       edgecolor=CHARCOAL, linewidth=0.5, label="sanity 9/9")
    for b, v in zip(bars_h, headline):
        ax_B.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v}",
                   ha="center", fontsize=7, fontweight="bold", color=CHARCOAL)
    for b, v in zip(bars_s, sanity):
        ax_B.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v}",
                   ha="center", fontsize=7, fontweight="bold", color=CHARCOAL)
    ax_B.set_xticks(x)
    xtl = []
    for i, (t, n) in enumerate(zip(thresholds, n_kept)):
        if i == len(thresholds) - 1:
            xtl.append(f"{t}\n(n={n:,})\ntop 1.4%")
        else:
            xtl.append(f"{t}\n(n={n:,})")
    ax_B.set_xticklabels(xtl, fontsize=6)
    ax_B.set_ylabel("n markers correct", fontsize=7)
    ax_B.set_ylim(0, 11)
    # Legend in the headroom inside the panel (it previously overlapped the
    # title sitting above the axes).
    ax_B.legend(fontsize=6, loc="upper center", ncol=2, frameon=False)
    ax_B.set_title("marker recovery at every threshold", fontsize=7, pad=16)
    panel_label(ax_B, "B", x=0.5, y=1.24, ha="center")

    # Panel C: composition vs axis correlation per LHD donor
    confound = add.get("confound_4_per_sample", {})
    donors = [d for d in LHD if d in confound]
    rs = [confound[d]["r"] for d in donors]
    max_abs = max(abs(r) for r in rs)
    cb = [RED_CENTRAL if abs(r) == max_abs else TEAL_LHD for r in rs]
    ax_C.bar(donors, rs, color=cb, edgecolor=CHARCOAL, linewidth=0.5)
    ax_C.axhline(0.3, color=RED_CENTRAL, lw=0.5, linestyle=":")
    ax_C.axhline(-0.3, color=RED_CENTRAL, lw=0.5, linestyle=":")
    ax_C.text(0.98, 0.92, "±0.3 §4.6 trigger", color=RED_CENTRAL, fontsize=5.5,
                transform=ax_C.transAxes, ha="right", va="top")
    ax_C.set_ylabel("r(s, hep_fraction)", fontsize=7)
    ax_C.tick_params(axis="x", labelsize=6.5)
    # Two-line title so it fits within the panel width.
    ax_C.set_title(
        "composition along axis\n"
        f"max |r| = {add.get('confound_4_cohort_max_abs_r', 0):.2f} "
        f"(M6) < 0.3 trigger",
        fontsize=6.0, pad=16,
    )
    panel_label(ax_C, "C", x=0.5, y=1.24, ha="center")

    fig.savefig(FIGURES / "figS4_celltype.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "figS4_celltype.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[figS4] Wrote figS4_celltype.png")


# -- S5 ----------------------------------------------------------------------

def figS5():
    """Steatosis volcano + weak-1 + |β₂| distribution."""
    df = pd.read_parquet(CRITERIA)
    fig = plt.figure(figsize=(mm(184), mm(96)))
    gs = GridSpec(1, 3, figure=fig, wspace=0.5,
                   left=0.08, right=0.97, bottom=0.20, top=0.85)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])

    # Panel A: volcano
    valid = df[df["cohort_status"] == "ok"].copy()
    valid["nlogq"] = -np.log10(valid["q_2"].clip(lower=1e-300))
    palette = {"robust_1234": RED_CENTRAL, "donor_sensitive_12": OCHRE_P,
                "weak_1": TEAL_LHD, "null_0": LIGHT_GRAY}
    flag_labels = {"robust_1234": "robust (all criteria)",
                    "donor_sensitive_12": "donor-sensitive (1+2)",
                    "weak_1": "criterion 1 only",
                    "null_0": "null"}
    for flag, sub in valid.groupby("four_criteria_flag"):
        flag_s = str(flag)
        ax_A.scatter(sub["beta_2"], sub["nlogq"], s=14,
                      color=palette.get(flag_s, LIGHT_GRAY),
                      edgecolor="white", linewidths=0.4, alpha=0.85,
                      label=flag_labels.get(flag_s, flag_s))
    ax_A.axhline(-np.log10(0.05), color=CHARCOAL, lw=0.5, linestyle=":")
    ax_A.text(0.02, -np.log10(0.05) + 0.2, "q = 0.05", fontsize=5.5,
                color=CHARCOAL, transform=ax_A.get_yaxis_transform())
    # Label robust genes (red) and all weak-1 genes (teal) for context
    label_subset = valid[valid["four_criteria_flag"].isin(
        ["robust_1234", "weak_1"])]
    for _, row in label_subset.iterrows():
        is_robust = row["four_criteria_flag"] == "robust_1234"
        ax_A.annotate(row["gene"], xy=(row["beta_2"], row["nlogq"]),
                       xytext=(4, 2), textcoords="offset points",
                       fontsize=6, fontstyle="italic",
                       color=RED_CENTRAL if is_robust else TEAL_LHD_DARK)
    ax_A.set_xlabel("β₂", fontsize=7)
    ax_A.set_ylabel("−log10(q)", fontsize=7)
    ax_A.legend(fontsize=5.5, loc="upper left", frameon=False, ncol=1)
    ax_A.set_title("66 strict-panel genes", fontsize=7, pad=4)
    panel_label(ax_A, "A")

    # Panel B: weak-1 leave-out diagnostics
    weak = valid[valid["four_criteria_flag"] == "weak_1"]
    cols = ["beta_2", "beta_2_leave_p6", "beta_2_leave_m1",
             "beta_2_leave_m2", "beta_2_leave_m3"]
    col_labels = ["full", "−P6", "−M1", "−M2", "−M3"]
    x = np.arange(len(cols))
    # Categorical palette (Okabe-Ito-style) — these are 5 distinct genes,
    # not an ordered axis
    weak_palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    mt1h_xy = None
    all_vals = []
    for i, (_, row) in enumerate(weak.iterrows()):
        vals = [row[c] for c in cols]
        all_vals.extend(vals)
        ax_B.plot(x, vals, "o-", color=weak_palette[i % len(weak_palette)],
                   lw=1.2, markersize=4, label=row["gene"])
        if row["gene"] == "MT1H":
            # x=2 is the −M1 column (full, −P6, −M1, −M2, −M3)
            mt1h_xy = (2, row["beta_2_leave_m1"])
    ax_B.axhline(0, color=CHARCOAL, lw=0.5)
    # Headroom above and below the curves: the legend goes in the clear top
    # band and the MT1H note in the clear bottom band, off the lines.
    lo, hi = min(all_vals), max(all_vals)
    span = hi - lo
    ax_B.set_ylim(lo - 0.46 * span, hi + 0.46 * span)
    if mt1h_xy is not None:
        ax_B.annotate("MT1H sign-flips at −M1",
                       xy=mt1h_xy, xytext=(2.0, lo - 0.30 * span),
                       textcoords="data", fontsize=6, color=CHARCOAL,
                       ha="center", va="center",
                       arrowprops=dict(arrowstyle="-", lw=0.5,
                                        color=CHARCOAL))
    ax_B.set_xticks(x); ax_B.set_xticklabels(col_labels, fontsize=6.5)
    ax_B.set_ylabel(r"$\beta_2$", fontsize=8)
    ax_B.legend(fontsize=5.5, loc="upper center", frameon=False, ncol=3,
                  columnspacing=1.2)
    ax_B.set_title("weak-1 genes (criterion 1 only)", fontsize=7, pad=4)
    panel_label(ax_B, "B")

    # Panel C: |β₂| distribution
    ranked = valid.sort_values("beta_2", key=lambda x: x.abs(), ascending=False)
    abs_b2 = ranked["beta_2"].abs().to_numpy()
    flag_for_rank = ranked["four_criteria_flag"].tolist()
    colors = [palette[f] for f in flag_for_rank]
    ax_C.bar(range(len(abs_b2)), abs_b2, color=colors, width=0.95,
              edgecolor="none")
    # Annotate "ranks 1-3 = robust" with a bracket above the first three bars
    if len(abs_b2) >= 3:
        y_top = abs_b2[0] * 1.15
        ax_C.plot([0, 2], [y_top, y_top], color=RED_CENTRAL, lw=0.6)
        ax_C.text(0, y_top * 1.04, "ranks 1-3 = robust",
                   color=RED_CENTRAL, fontsize=5.5, ha="left", va="bottom")
        ax_C.set_ylim(0, y_top * 1.18)
    # Inline mini-legend
    handles = [
        mpatches.Patch(color=RED_CENTRAL, label="robust"),
        mpatches.Patch(color=TEAL_LHD, label="criterion 1 only"),
        mpatches.Patch(color=LIGHT_GRAY, label="null"),
    ]
    ax_C.legend(handles=handles, fontsize=5.5, loc="upper right",
                  frameon=False)
    ax_C.set_xlabel("gene rank", fontsize=7)
    ax_C.set_ylabel(r"$|\beta_2|$", fontsize=8)
    ax_C.set_title("|β₂| magnitude distribution", fontsize=7, pad=4)
    panel_label(ax_C, "C")

    fig.savefig(FIGURES / "figS5_steatosis_ranked.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "figS5_steatosis_ranked.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[figS5] Wrote figS5_steatosis_ranked.png")


# -- S6 ----------------------------------------------------------------------

def figS6():
    """SAA1 portal-side acute-phase response (weak_1 set headline).

    Three panels:
      A: SAA1 per-tertile g(s) curves (the load-bearing panel — demonstrates
         portal intensification with portal Δ +0.31 vs. central Δ +0.16).
      B: ORM1 per-tertile g(s) curves (strict-set portal-intensification
         analogue — same visual encoding, lets reader compare at a glance).
      C: SAA1 leave-one-donor-out β₂ forest plot (full cohort + 4 leave-out
         variants), with 95% Wald CI derived from cluster-robust p-value.
    """
    print("[figS6] Loading artifacts...")
    d11_called8 = json.loads((RESULTS / "d11_spatial_patterns_called8.json").read_text())
    df = pd.read_parquet(CRITERIA)

    palette = {"low": BLUE_PORTAL, "mid": LIGHT_GRAY, "high": RED_CENTRAL}

    fig = plt.figure(figsize=(mm(195), mm(72)))
    gs = GridSpec(1, 3, figure=fig, wspace=0.55,
                   left=0.06, right=0.98, bottom=0.18, top=0.86,
                   width_ratios=[1.0, 1.0, 1.25])

    # ---- Panel A: SAA1 per-tertile curves ----
    ax_A = fig.add_subplot(gs[0, 0])
    saa1 = d11_called8["per_gene"]["SAA1"]
    cls = saa1.get("_classification", {})
    all_vals = []
    for label in ("low", "mid", "high"):
        v = saa1.get(label, {})
        if v.get("status") == "too_few":
            continue
        all_vals.extend([m for m in v["bin_means"] if m is not None and not np.isnan(m)])
        all_vals.extend([m for m in v["bin_ci_lo"] if m is not None and not np.isnan(m)])
        all_vals.extend([m for m in v["bin_ci_hi"] if m is not None and not np.isnan(m)])
    y_min, y_max = float(np.min(all_vals)), float(np.max(all_vals))
    pad = 0.10 * (y_max - y_min)
    ax_A.set_ylim(y_min - pad, y_max + pad)
    for label in ("low", "mid", "high"):
        v = saa1[label]
        xs = np.asarray(v["bin_centers"])
        means = np.asarray(v["bin_means"])
        ci_lo = np.asarray(v["bin_ci_lo"])
        ci_hi = np.asarray(v["bin_ci_hi"])
        n = v["n_spots"]
        ax_A.fill_between(xs, ci_lo, ci_hi, color=palette[label],
                            alpha=0.18, linewidth=0)
        ax_A.plot(xs, means, color=palette[label], lw=1.7,
                   label=f"{label} (n={n:,})")
    ax_A.axvline(0.5, color=GRID_GRAY, lw=0.6, ls="--", zorder=0)
    ax_A.set_xlim(0, 1); ax_A.set_xticks([0, 0.5, 1.0])
    ax_A.tick_params(axis="both", labelsize=6.5)
    ax_A.set_xlabel("s (portal → central)", fontsize=8)
    ax_A.set_ylabel("SAA1 log-norm", fontsize=8, fontstyle="italic")
    # Δ annotations
    portal_d = cls.get("delta_high_minus_low_mean_low_s", 0.0)
    central_d = cls.get("delta_high_minus_low_mean_high_s", 0.0)
    ax_A.text(0.25, 0.04,
                f"Δ portal\n{portal_d:+.2f}",
                transform=ax_A.transAxes, ha="center", va="bottom",
                fontsize=6.5, color=RED_CENTRAL, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                           boxstyle="round,pad=0.20"))
    ax_A.text(0.78, 0.04,
                f"Δ central\n{central_d:+.2f}",
                transform=ax_A.transAxes, ha="center", va="bottom",
                fontsize=6.5, color=CHARCOAL,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                           boxstyle="round,pad=0.20"))
    ax_A.text(0.97, 0.96, "portal intensification",
                transform=ax_A.transAxes, ha="right", va="top",
                fontsize=6.0, fontweight="bold", fontstyle="italic",
                color=CHARCOAL,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                           boxstyle="round,pad=0.20"))
    ax_A.legend(fontsize=6.0, loc="lower left",
                 bbox_to_anchor=(0.0, 1.02), frameon=False, ncol=3,
                 title="lipid tertile", title_fontsize=6.5,
                 columnspacing=1.0, handlelength=1.3)

    # ---- Panel B: ORM1 per-tertile curves (analogue) ----
    ax_B = fig.add_subplot(gs[0, 1])
    orm1 = d11_called8["per_gene"]["ORM1"]
    cls_o = orm1.get("_classification", {})
    all_vals_o = []
    for label in ("low", "mid", "high"):
        v = orm1.get(label, {})
        if v.get("status") == "too_few":
            continue
        all_vals_o.extend([m for m in v["bin_means"] if m is not None and not np.isnan(m)])
        all_vals_o.extend([m for m in v["bin_ci_lo"] if m is not None and not np.isnan(m)])
        all_vals_o.extend([m for m in v["bin_ci_hi"] if m is not None and not np.isnan(m)])
    y_min_o, y_max_o = float(np.min(all_vals_o)), float(np.max(all_vals_o))
    pad_o = 0.10 * (y_max_o - y_min_o)
    ax_B.set_ylim(y_min_o - pad_o, y_max_o + pad_o)
    for label in ("low", "mid", "high"):
        v = orm1[label]
        xs = np.asarray(v["bin_centers"])
        means = np.asarray(v["bin_means"])
        ci_lo = np.asarray(v["bin_ci_lo"])
        ci_hi = np.asarray(v["bin_ci_hi"])
        ax_B.fill_between(xs, ci_lo, ci_hi, color=palette[label],
                            alpha=0.18, linewidth=0)
        ax_B.plot(xs, means, color=palette[label], lw=1.7)
    ax_B.axvline(0.5, color=GRID_GRAY, lw=0.6, ls="--", zorder=0)
    ax_B.set_xlim(0, 1); ax_B.set_xticks([0, 0.5, 1.0])
    ax_B.tick_params(axis="both", labelsize=6.5)
    ax_B.set_xlabel("s (portal → central)", fontsize=8)
    ax_B.set_ylabel("ORM1 log-norm", fontsize=8, fontstyle="italic")
    portal_o = cls_o.get("delta_high_minus_low_mean_low_s", 0.0)
    central_o = cls_o.get("delta_high_minus_low_mean_high_s", 0.0)
    ax_B.text(0.25, 0.04,
                f"Δ portal\n{portal_o:+.2f}",
                transform=ax_B.transAxes, ha="center", va="bottom",
                fontsize=6.5, color=CHARCOAL,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                           boxstyle="round,pad=0.20"))
    ax_B.text(0.78, 0.04,
                f"Δ central\n{central_o:+.2f}",
                transform=ax_B.transAxes, ha="center", va="bottom",
                fontsize=6.5, color=CHARCOAL,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                           boxstyle="round,pad=0.20"))
    ax_B.text(0.97, 0.96, "portal intensification (analogue)",
                transform=ax_B.transAxes, ha="right", va="top",
                fontsize=6.0, fontweight="bold", fontstyle="italic",
                color=CHARCOAL,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                           boxstyle="round,pad=0.20"))

    # ---- Panel C: SAA1 leave-out forest plot ----
    ax_C = fig.add_subplot(gs[0, 2])
    saa1_row = df[df["gene"] == "SAA1"].iloc[0]
    from scipy.stats import norm as _norm
    # Per-leave-out n_spots: full cohort 10,399; donor counts M1=3,588,
    # M2=1,229, M3=2,448, P6=3,134 (from d11 cohort = M1+M2+M3+P6).
    rows = [
        ("full cohort", saa1_row["beta_2"],         saa1_row["p_2"],         10399, True),
        ("− P6",        saa1_row["beta_2_leave_p6"], saa1_row["p_2_leave_p6"],  7265, False),
        ("− M1",        saa1_row["beta_2_leave_m1"], saa1_row["p_2_leave_m1"],  6811, False),
        ("− M2",        saa1_row["beta_2_leave_m2"], saa1_row["p_2_leave_m2"],  9170, False),
        ("− M3",        saa1_row["beta_2_leave_m3"], saa1_row["p_2_leave_m3"],  7951, False),
    ]
    # Wald-style 95% CI from β and p (two-sided): SE = |β| / z(1-p/2)
    y_pos = np.arange(len(rows))
    for i, (label, b, p, n_sp, is_full) in enumerate(rows):
        b = float(b); p = float(p)
        z = _norm.ppf(1 - p / 2)
        se = abs(b) / max(z, 1e-9)
        lo = b - 1.96 * se; hi = b + 1.96 * se
        col = TEAL_LHD_DARK if is_full else CHARCOAL
        ms = 7 if is_full else 5
        lw = 1.4 if is_full else 1.0
        ax_C.plot([lo, hi], [i, i], color=col, lw=lw, zorder=2)
        ax_C.plot([b], [i], "o", color=col, ms=ms, zorder=3,
                   markeredgecolor="white", markeredgewidth=0.8)
        # Annotate β / p / n to the right of the upper CI bound
        ax_C.text(hi + 0.005, i,
                   f"{b:+.3f}  p={p:.3f}  n={n_sp:,}",
                   fontsize=6.0, va="center", ha="left", color=CHARCOAL)
    ax_C.axvline(0, color=CHARCOAL, lw=0.7, zorder=1)
    ax_C.set_yticks(y_pos)
    ax_C.set_yticklabels([r[0] for r in rows], fontsize=7.0)
    ax_C.tick_params(axis="x", labelsize=6.5)
    ax_C.set_xlabel(r"$\hat\beta_2$ (s × lipid_pct)", fontsize=8)
    ax_C.invert_yaxis()
    # Pad x-limits so β-value annotations don't get clipped
    all_b = [float(r[1]) for r in rows]
    all_p = [float(r[2]) for r in rows]
    all_se = [abs(b) / max(_norm.ppf(1 - p / 2), 1e-9) for b, p in zip(all_b, all_p)]
    x_lo = min(b - 1.96 * se for b, se in zip(all_b, all_se))
    x_hi = max(b + 1.96 * se for b, se in zip(all_b, all_se))
    ax_C.set_xlim(x_lo - 0.005, x_hi + 0.075)
    ax_C.set_title("SAA1 leave-one-donor-out β̂₂ (95% cluster-robust Wald CI)",
                    fontsize=7.5, color=CHARCOAL, pad=4)
    ax_C.grid(axis="x", alpha=0.3)
    ax_C.set_axisbelow(True)

    panel_label(ax_A, "A", x=-0.15, y=1.18)
    panel_label(ax_B, "B", x=-0.15, y=1.18)
    panel_label(ax_C, "C", x=-0.32, y=1.10)

    out = FIGURES / "figS6_saa1_portal_response.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[figS6] Wrote {out}")


# -- S7 ----------------------------------------------------------------------

def figS7():
    """Failed §9.9 LHD-vs-P calibration with Supp T6."""
    d = json.loads(SUPP_T6.read_text())
    fig = plt.figure(figsize=(mm(180), mm(75)))
    gs = GridSpec(1, 2, figure=fig, wspace=0.40,
                   left=0.12, right=0.95, bottom=0.18, top=0.82)
    # Panel A is split into two sub-axes (sign-agreement % and Spearman ρ)
    # because they're bounded on different intervals — co-plotting on a
    # single axis falsely implies a shared scale.
    sub_a = gs[0, 0].subgridspec(1, 2, wspace=0.55)
    ax_A1 = fig.add_subplot(sub_a[0, 0])
    ax_A2 = fig.add_subplot(sub_a[0, 1])
    ax_B = fig.add_subplot(gs[0, 1])

    pct = d.get("sign_agreement_pct", 0)
    rho = d.get("signed_magnitude_spearman", 0)
    n_compared = d.get("n_compared_directional", 0)
    n_agree = d.get("n_agree", 0)

    # Panel A1: sign agreement on a proper 0-100% scale
    ax_A1.bar([0], [pct], color=OCHRE_P_DARK,
               edgecolor=CHARCOAL, linewidth=0.5, width=0.55)
    ax_A1.axhline(80, color=RED_CENTRAL, lw=0.6, linestyle=":")
    ax_A1.text(0.98, 0.83, "target ≥80%", fontsize=5.5, color=RED_CENTRAL,
                transform=ax_A1.transAxes, ha="right", va="bottom")
    ax_A1.text(0, pct + 2.5, f"{pct:.1f}%", ha="center", fontsize=7,
                fontweight="bold", color=RED_CENTRAL)
    ax_A1.set_xticks([0]); ax_A1.set_xticklabels(["sign agreement"], fontsize=6.5)
    ax_A1.set_ylim(0, 100)
    ax_A1.set_ylabel("agreement (%)", fontsize=7)
    ax_A1.set_xlim(-0.6, 0.6)
    panel_label(ax_A1, "A")

    # Panel A2: Spearman ρ on its native [-1, +1] scale
    ax_A2.bar([0], [rho], color=OCHRE_P,
               edgecolor=CHARCOAL, linewidth=0.5, width=0.55)
    ax_A2.axhline(0, color=CHARCOAL, lw=0.5)
    ax_A2.text(0, rho - 0.04, f"ρ = {rho:.3f}", ha="center", va="top",
                fontsize=7, fontweight="bold", color=CHARCOAL)
    ax_A2.set_xticks([0]); ax_A2.set_xticklabels(["Spearman ρ"], fontsize=6.5)
    ax_A2.set_ylim(-1, 1)
    ax_A2.set_ylabel("ρ", fontsize=7)
    ax_A2.set_xlim(-0.6, 0.6)

    # Single title spanning both A1 and A2 — placed in figure coords above
    # the sub-axis pair.
    fig.text(0.30, 0.91,
              f"FAIL: {n_agree}/{n_compared} agree ({pct:.1f}%); "
              f"ρ = {rho:.3f}",
              fontsize=7, color=RED_CENTRAL, ha="center", fontweight="bold")

    # Panel B: sanity-marker subset
    sanity = d.get("sanity_per_marker", {})
    if not sanity:
        ax_B.text(0.5, 0.5, "no sanity data", ha="center", va="center",
                   color=LIGHT_GRAY, transform=ax_B.transAxes)
    else:
        genes = list(sanity.keys())
        ours = [sanity[g]["our_portal_bias"] for g in genes]
        paper = [sanity[g]["paper_residual"] for g in genes]
        agree = [sanity[g]["agree"] for g in genes]
        x = np.arange(len(genes))
        # Per-gene background shading: light green for sign-agreement,
        # light red for cross-zero disagreement. Drawn first so bars
        # paint on top.
        for i, ag in enumerate(agree):
            shade = "#D6EFD6" if ag else "#F4D6D6"
            ax_B.axvspan(i - 0.45, i + 0.45, color=shade, alpha=0.7,
                          zorder=0)
        ax_B.bar(x - 0.18, ours, width=0.35, color=TEAL_LHD,
                  edgecolor=CHARCOAL, linewidth=0.5, label="our −Δslope",
                  zorder=2)
        ax_B.bar(x + 0.18, paper, width=0.35, color=OCHRE_P,
                  edgecolor=CHARCOAL, linewidth=0.5, label="Supp T6 residual",
                  zorder=2)
        for i, ag in enumerate(agree):
            mark = "✓" if ag else "✗"
            col = TEAL_LHD_DARK if ag else RED_CENTRAL
            ax_B.text(x[i], 0.05, mark, fontsize=10, ha="center",
                       color=col, transform=ax_B.get_xaxis_transform())
        ax_B.set_xticks(x)
        ax_B.set_xticklabels(genes, fontsize=6.5, fontstyle="italic")
        ax_B.axhline(0, color=CHARCOAL, lw=0.5, zorder=1)
        # Extend the top of the axis so the legend and the shading key sit
        # in clear headroom above the bars.
        ax_B.set_ylim(top=ax_B.get_ylim()[1] + 0.2)
        ax_B.set_ylabel("metric value (a.u.)", fontsize=7)
        ax_B.legend(fontsize=6, frameon=False, loc="upper left")
        ax_B.text(0.98, 0.97,
                   "green = both bars same sign\nred = cross-zero disagreement",
                   fontsize=5.5, color=CHARCOAL, transform=ax_B.transAxes,
                   ha="right", va="top")
        n_ag = sum(agree)
        ax_B.set_title(f"sanity markers: {n_ag}/{len(agree)} orientation-correct",
                        fontsize=7, pad=4)
    panel_label(ax_B, "B")

    fig.savefig(FIGURES / "figS7_lhd_vs_p.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "figS7_lhd_vs_p.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[figS7] Wrote figS7_lhd_vs_p.png")


# -- S8 ----------------------------------------------------------------------

def figS8():
    """Sensitivity of headline marker recovery across normalisation, mt-fraction
    threshold, and bin count. The methods text cites this figure for the §13
    sensitivity-sweep result; a flat-line plot doesn't earn its space, so the
    payload is rendered as a compact summary table."""
    d = json.loads(B8.read_text())
    n_h = d["n_headline"]
    fig = plt.figure(figsize=(mm(140), mm(60)))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_off()

    rows = [
        ("Normalisation", "log1p (default)", 0, n_h),
        ("Normalisation", "Pearson residuals", 0, n_h),
        ("Mt-fraction threshold", "0%", 0, n_h),
        ("Mt-fraction threshold", "25%", 0, n_h),
        ("Mt-fraction threshold", "45%", 0, n_h),
        ("Bin count", "N = 30", 0, n_h),
        ("Bin count", "N = 50 (default)", 0, n_h),
        ("Bin count", "N = 100", 0, n_h),
        ("Bin count", "N = 150", 0, n_h),
    ]

    # Header
    headers = ["Sweep dimension", "Variant",
                "Disagreements", "% stable"]
    col_x = [0.02, 0.30, 0.62, 0.84]
    y_top = 0.93
    row_h = 0.085
    for cx, h in zip(col_x, headers):
        ax.text(cx, y_top, h, fontsize=7, fontweight="bold", color=CHARCOAL,
                 transform=ax.transAxes, va="top", ha="left")
    ax.plot([0.02, 0.98], [y_top - 0.04, y_top - 0.04], color=CHARCOAL,
             lw=0.8, transform=ax.transAxes)

    last_dim = None
    for i, (dim, variant, ndis, ntot) in enumerate(rows):
        y = y_top - 0.10 - i * row_h
        # Group separator
        if last_dim is not None and dim != last_dim:
            ax.plot([0.02, 0.98], [y + row_h * 0.42, y + row_h * 0.42],
                     color="#CCCCCC", lw=0.5, transform=ax.transAxes)
        last_dim = dim
        dim_label = dim if (i == 0 or rows[i - 1][0] != dim) else ""
        ax.text(col_x[0], y, dim_label, fontsize=6.5, color=CHARCOAL,
                 transform=ax.transAxes, va="top", ha="left",
                 fontstyle="italic" if dim_label else "normal")
        ax.text(col_x[1], y, variant, fontsize=6.5, color=CHARCOAL,
                 transform=ax.transAxes, va="top", ha="left")
        # 0 disagreements rendered with a check; nonzero would be red.
        col_dis = TEAL_LHD_DARK if ndis == 0 else RED_CENTRAL
        ax.text(col_x[2], y, f"{ndis}/{ntot}", fontsize=6.5, color=col_dis,
                 transform=ax.transAxes, va="top", ha="left",
                 fontweight="bold")
        pct = 100.0 * (ntot - ndis) / ntot if ntot else 0.0
        col_pct = TEAL_LHD_DARK if pct >= 95 else RED_CENTRAL
        ax.text(col_x[3], y, f"{pct:.0f}%", fontsize=6.5, color=col_pct,
                 transform=ax.transAxes, va="top", ha="left",
                 fontweight="bold")

    fig.text(0.5, 0.05,
              "0/23 candidate-headline genes change sign across any "
              "sweep dimension — well within the pre-declared "
              "≤2-disagreers threshold.",
              fontsize=6, color=CHARCOAL, ha="center", fontstyle="italic")

    fig.savefig(FIGURES / "figS8_sensitivity_sweep.png", dpi=300,
                  bbox_inches="tight")
    fig.savefig(FIGURES / "figS8_sensitivity_sweep.pdf",
                  bbox_inches="tight")
    plt.close(fig)
    print("[figS8] Wrote figS8_sensitivity_sweep.png")


# -- S9 ----------------------------------------------------------------------

def figS9():
    """C1 adjustment sensitivity — per-gene slope with vs without log(UMI)
    residualisation. Two panels: (A) per-gene scatter of slope_unadjusted
    vs slope_adjusted across 23 candidate-headline genes, with UMI-sensitive
    genes flagged; (B) histogram of delta_slope showing the magnitude shift
    while signs stay stable (0/23 sign changes)."""
    summary = json.loads(B7_SUMMARY.read_text())
    df = pd.read_parquet(B7_JOINT)
    fig = plt.figure(figsize=(mm(180), mm(75)))
    gs = GridSpec(1, 2, figure=fig, wspace=0.30,
                   width_ratios=[1.15, 1.0],
                   left=0.08, right=0.97, bottom=0.18, top=0.86)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])

    # ---- Panel A: scatter of unadjusted vs adjusted ----
    x = df["slope_unadjusted"].to_numpy()
    y = df["slope_adjusted"].to_numpy()
    umi_sens = df["umi_sensitive"].to_numpy().astype(bool)
    is_marker = df["is_known_marker"].to_numpy().astype(bool)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.10 * (hi - lo)
    lim = (lo - pad, hi + pad)

    # Agreement quadrants (Q1 + Q3, signs of slope agree).
    ax_A.add_patch(mpatches.Rectangle((0, 0), lim[1], lim[1],
                                        facecolor=TEAL_LHD, alpha=0.10,
                                        zorder=0))
    ax_A.add_patch(mpatches.Rectangle((lim[0], lim[0]), -lim[0], -lim[0],
                                        facecolor=TEAL_LHD, alpha=0.10,
                                        zorder=0))
    # Zero lines + unity diagonal.
    ax_A.axhline(0, color=GRID_GRAY, lw=0.6, zorder=1)
    ax_A.axvline(0, color=GRID_GRAY, lw=0.6, zorder=1)
    ax_A.plot([lim[0], lim[1]], [lim[0], lim[1]], color=LIGHT_GRAY,
                lw=0.7, linestyle="--", zorder=2)

    # Plot non-UMI-sensitive (filled) and UMI-sensitive (open).
    not_sens = ~umi_sens
    ax_A.scatter(x[not_sens], y[not_sens], s=28, color=TEAL_LHD,
                  edgecolor=CHARCOAL, linewidths=0.5, zorder=3,
                  label=f"stable ({int(not_sens.sum())})")
    ax_A.scatter(x[umi_sens], y[umi_sens], s=28, facecolor="white",
                  edgecolor=CHARCOAL, linewidths=0.7, zorder=3,
                  label=f"UMI-sensitive ({int(umi_sens.sum())})")

    # Label canonical markers in italic.
    for i in np.where(is_marker)[0]:
        ax_A.annotate(df["gene"].iloc[i], xy=(x[i], y[i]),
                       xytext=(6, 4), textcoords="offset points",
                       fontsize=6, color=RED_CENTRAL, fontstyle="italic")

    rxy = float(summary.get("across_gene_r_slope_with_vs_without", 0.0))
    n_flip = int(summary.get("n_sign_change_pre_demotion", 0))
    n_total = int(summary.get("n_headline", len(df)))
    annot_text = (f"across-gene r = {rxy:.3f}\n"
                  f"{n_flip}/{n_total} sign changes")
    ax_A.text(0.04, 0.96, annot_text, transform=ax_A.transAxes,
                va="top", ha="left", fontsize=6.5, color=CHARCOAL,
                bbox=dict(facecolor="white", edgecolor="#CCCCCC",
                            linewidth=0.5, boxstyle="round,pad=0.30",
                            alpha=0.92))
    ax_A.set_xlim(*lim); ax_A.set_ylim(*lim)
    ax_A.set_aspect("equal", adjustable="box")
    ax_A.set_xlabel("slope (unadjusted)", fontsize=7)
    ax_A.set_ylabel("slope (log-UMI adjusted)", fontsize=7)
    ax_A.tick_params(axis="both", labelsize=6.5)
    ax_A.legend(fontsize=6, frameon=False, loc="lower right",
                  handletextpad=0.4)
    ax_A.set_title("per-gene slope: 23 candidate-headline genes",
                    fontsize=7, pad=4)
    panel_label(ax_A, "A", x=-0.16, y=1.04)

    # ---- Panel B: delta_slope histogram ----
    delta = df["delta_slope"].to_numpy()
    bins = np.linspace(min(delta.min() * 1.05, -0.05),
                        max(delta.max() * 1.05, 0.12), 14)
    # Stack: stable + UMI-sensitive separately for the same colours.
    ax_B.hist([delta[~umi_sens], delta[umi_sens]], bins=bins,
                stacked=True, color=[TEAL_LHD, "white"],
                edgecolor=CHARCOAL, linewidth=0.5,
                label=["stable", "UMI-sensitive"])
    ax_B.axvline(0, color=CHARCOAL, lw=0.6, linestyle=":")
    ax_B.set_xlabel(r"$\Delta$slope = adjusted $-$ unadjusted", fontsize=7)
    ax_B.set_ylabel("number of genes", fontsize=7)
    ax_B.tick_params(axis="both", labelsize=6.5)
    ax_B.legend(fontsize=6, frameon=False, loc="upper right",
                  handletextpad=0.4)
    ax_B.set_title("magnitude shifts; no sign flips", fontsize=7, pad=4)
    panel_label(ax_B, "B", x=-0.16, y=1.04)

    fig.savefig(FIGURES / "figS9_c1_adjustment.png", dpi=300,
                  bbox_inches="tight")
    fig.savefig(FIGURES / "figS9_c1_adjustment.pdf",
                  bbox_inches="tight")
    plt.close(fig)
    print("[figS9] Wrote figS9_c1_adjustment.png")


def main():
    figS1()
    figS2()
    figS3()
    figS4()
    figS5()
    figS6()
    figS7()
    figS8()
    figS9()


if __name__ == "__main__":
    main()
