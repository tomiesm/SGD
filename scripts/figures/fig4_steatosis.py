"""Figure 4 — Steatosis biology with three robust genes (revised v2).

Four panels (A and F removed): B leave-out matrix, C per-gene spatial
patterns, D Supp T11 calibration, E GSEA pathway enrichment. Layout:
B | C (row 1, C wider as centerpiece) and D | E (row 2).
"""

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

from sgd.config import FIGURES, RESULTS, SUPP_T11
from sgd.plotstyle import (BLUE_PORTAL, CHARCOAL, GRID_GRAY, LIGHT_GRAY,
                           RED_CENTRAL, TEAL_LHD, TEAL_LHD_DARK, apply_style,
                           mm, panel_label)

warnings.filterwarnings("ignore")
apply_style()

CRITERIA = RESULTS / "steatosis_call_criteria.parquet"
SUMMARY = RESULTS / "steatosis_call_summary.json"
SPATIAL = RESULTS / "d11_spatial_patterns.json"
SUPP_T11_CALIB = RESULTS / "supp_t11_calibration.json"
GSEA = RESULTS / "d12_gsea_summary.json"

ROBUST_GENES = ("GLUL", "LYZ", "ORM1")


def panel_B(ax) -> None:
    """Leave-out matrix for the 3 robust genes — sign + magnitude.

    Numerical values are placed BELOW each cell (in compact table form)
    so they don't crowd the heat-map cells themselves.
    """
    df = pd.read_parquet(CRITERIA)
    rows = []
    for g in ROBUST_GENES:
        r = df[df["gene"] == g].iloc[0]
        rows.append({
            "full": r["beta_2"],
            "leave_p6": r["beta_2_leave_p6"],
            "leave_m1": r["beta_2_leave_m1"],
            "leave_m2": r["beta_2_leave_m2"],
            "leave_m3": r["beta_2_leave_m3"],
        })
    cols = ["full", "leave_p6", "leave_m1", "leave_m2", "leave_m3"]
    col_labels = ["full", "−P6", "−M1", "−M2", "−M3"]
    M = np.array([[r[c] for c in cols] for r in rows])
    n_rows, n_cols = M.shape
    abs_max = np.max(np.abs(M))

    # Heatmap cells (color only, no in-cell text)
    for i in range(n_rows):
        for j in range(n_cols):
            v = M[i, j]
            base = RED_CENTRAL if v > 0 else BLUE_PORTAL
            sat = min(0.85, 0.30 + 0.55 * abs(v) / abs_max)
            rect = mpatches.Rectangle((j, n_rows - 1 - i), 1, 1, facecolor=base,
                                        alpha=sat, edgecolor="white", lw=0.6)
            ax.add_patch(rect)

    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(reversed(ROBUST_GENES), fontsize=8.5,
                        fontstyle="italic")
    ax.tick_params(axis="both", length=0)
    ax.set_aspect("equal")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("β₂ leave-out matrix", fontsize=8, color=CHARCOAL, pad=4)

    # Compact value table BELOW the heatmap.
    table_y0 = -1.0
    for i in range(n_rows):
        row_idx = n_rows - 1 - i
        gene = ROBUST_GENES[row_idx]
        ax.text(-0.10, table_y0 - row_idx * 0.55, gene, fontsize=6.5,
                 fontstyle="italic", ha="right", va="center", color=CHARCOAL)
        for j in range(n_cols):
            v = M[row_idx, j]
            ax.text(j + 0.5, table_y0 - row_idx * 0.55, f"{v:+.3f}",
                     fontsize=6, ha="center", va="center",
                     color=CHARCOAL)
    ax.text(0.5 * n_cols, table_y0 + 0.50, "β₂ values", fontsize=6,
             ha="center", va="bottom", color=LIGHT_GRAY, fontstyle="italic")
    ax.set_ylim(table_y0 - n_rows * 0.55 - 0.4, n_rows + 0.4)

    # Sign legend below value table
    leg_y = table_y0 - n_rows * 0.55 - 0.20
    leg = [
        mpatches.Patch(color=RED_CENTRAL, alpha=0.7,
                        label="β₂ > 0 (central-amplified)"),
        mpatches.Patch(color=BLUE_PORTAL, alpha=0.7,
                        label="β₂ < 0 (portal-shifted)"),
    ]
    ax.legend(handles=leg, fontsize=6, loc="upper center",
               bbox_to_anchor=(0.5, leg_y / abs(table_y0 - n_rows * 0.55 - 0.2)),
               ncol=1, frameon=False)


def panel_C(fig, gs_outer) -> None:
    """3 stacked subplots — per-bin g(s) by lipid tertile."""
    d = json.loads(SPATIAL.read_text())
    inner = gs_outer.subgridspec(3, 1, hspace=0.55)
    palette = {"low": BLUE_PORTAL, "mid": LIGHT_GRAY, "high": RED_CENTRAL}
    pattern_label = {
        "GLUL": "amplitude shift",
        "LYZ": "flattening",
        "ORM1": "portal intensification",
    }

    for k, gene in enumerate(ROBUST_GENES):
        ax = fig.add_subplot(inner[k, 0])
        gd = d["per_gene"].get(gene, {})

        # Pre-compute y-range across tertiles to set axes properly
        all_vals = []
        for label in ("low", "mid", "high"):
            v = gd.get(label)
            if v and v.get("status") != "too_few":
                all_vals.extend([m for m in v["bin_means"]
                                  if m is not None and not np.isnan(m)])
                all_vals.extend([m for m in v["bin_ci_lo"]
                                  if m is not None and not np.isnan(m)])
                all_vals.extend([m for m in v["bin_ci_hi"]
                                  if m is not None and not np.isnan(m)])
        if all_vals:
            y_min, y_max = float(np.min(all_vals)), float(np.max(all_vals))
            pad = 0.10 * (y_max - y_min)
            ax.set_ylim(y_min - pad, y_max + pad)

        for label in ("low", "mid", "high"):
            v = gd.get(label)
            if not v or v.get("status") == "too_few":
                continue
            xs = np.asarray(v["bin_centers"])
            means = np.asarray(v["bin_means"])
            ci_lo = np.asarray(v["bin_ci_lo"])
            ci_hi = np.asarray(v["bin_ci_hi"])
            n = v.get("n_spots", "?")
            ax.fill_between(xs, ci_lo, ci_hi, color=palette[label],
                             alpha=0.18, linewidth=0)
            ax.plot(xs, means, color=palette[label], lw=1.7,
                    label=f"{label} (n={n:,})")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.tick_params(axis="both", labelsize=6.5)
        ax.set_ylabel(f"{gene}\nlog-norm", fontsize=8, fontstyle="italic")

        # Pattern label — colored to match the lipid-high line (RED_CENTRAL)
        # for prominence
        ax.text(0.97, 0.93, pattern_label[gene], transform=ax.transAxes,
                 ha="right", va="top", fontsize=7.5, fontweight="bold",
                 fontstyle="italic", color=RED_CENTRAL,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                            boxstyle="round,pad=0.30"))
        if k == 2:
            ax.set_xlabel("s (portal → central)", fontsize=8)
        if k == 0:
            ax.legend(fontsize=6.5, loc="upper left", frameon=False, ncol=3,
                       bbox_to_anchor=(0, 1.45),
                       title="lipid tertile",
                       title_fontsize=7)


def panel_D(ax) -> None:
    """Supp T11 calibration — paired bars normalised per side, agree ✓."""
    df = pd.read_parquet(CRITERIA)
    sup = json.loads(SUPP_T11_CALIB.read_text())
    rows = []
    for g in ROBUST_GENES:
        r = df[df["gene"] == g].iloc[0]
        rows.append({"gene": g, "beta_2": r["beta_2"]})

    try:
        t11 = pd.read_excel(
            SUPP_T11,
            sheet_name="Visium spot lipid content", header=1)
        t11.columns = [str(c).strip() for c in t11.columns]
        if "Gene" in t11.columns:
            t11 = t11.set_index("Gene")
        elif "gene" in t11.columns:
            t11 = t11.set_index("gene")
        bin_cols = [c for c in t11.columns if c.startswith("mean lipid bin")]
        for c in bin_cols:
            t11[c] = pd.to_numeric(t11[c], errors="coerce")
        high = t11[[c for c in bin_cols if c.endswith("4") or c.endswith("5")]].mean(axis=1)
        low = t11[[c for c in bin_cols if c.endswith("0") or c.endswith("1")]].mean(axis=1)
        diff = high - low
        diff = diff[~diff.index.duplicated()]
    except Exception:
        diff = None

    y = np.arange(len(ROBUST_GENES))
    bar_h = 0.32
    b2s = np.array([r["beta_2"] for r in rows])
    if diff is not None:
        d_vals = np.array([float(diff.loc[r["gene"]])
                            if r["gene"] in diff.index else np.nan
                            for r in rows])
    else:
        d_vals = np.full(len(rows), np.nan)
    b2_norm = b2s / max(np.max(np.abs(b2s)), 1e-9)
    d_norm = d_vals / np.maximum(np.nanmax(np.abs(d_vals)), 1e-9)

    for i, row in enumerate(rows):
        b2 = b2s[i]
        b2n = b2_norm[i]
        col_b2 = RED_CENTRAL if b2 > 0 else BLUE_PORTAL
        ax.barh(y[i] + bar_h / 2, b2n, height=bar_h, color=col_b2,
                 edgecolor=CHARCOAL, linewidth=0.5)
        if not np.isnan(d_norm[i]):
            col_d = RED_CENTRAL if d_vals[i] > 0 else BLUE_PORTAL
            ax.barh(y[i] - bar_h / 2, d_norm[i], height=bar_h,
                     color=col_d, alpha=0.55, edgecolor=CHARCOAL,
                     linewidth=0.5, hatch="///")
            if np.sign(b2) == np.sign(d_vals[i]):
                ax.text(1.32, y[i], "✓", fontsize=11, va="center", ha="left",
                         color=TEAL_LHD_DARK,
                         transform=ax.get_yaxis_transform())

    # Numerical annotations placed JUST PAST the bar tip on the OUTSIDE
    # (further from zero), so they never overlap the bars themselves.
    for i, row in enumerate(rows):
        b2 = b2s[i]
        b2n = b2_norm[i]
        if b2n >= 0:
            xt_b2 = b2n + 0.05
            ha_b2 = "left"
        else:
            xt_b2 = b2n - 0.05
            ha_b2 = "right"
        ax.text(xt_b2, y[i] + bar_h / 2,
                 f"β₂={b2:+.3f}", fontsize=6, va="center", ha=ha_b2,
                 color=CHARCOAL)
        if not np.isnan(d_vals[i]):
            dn = d_norm[i]
            if dn >= 0:
                xt_d = dn + 0.05
                ha_d = "left"
            else:
                xt_d = dn - 0.05
                ha_d = "right"
            ax.text(xt_d, y[i] - bar_h / 2,
                     f"Δ={d_vals[i]:+.3f}", fontsize=6, va="center", ha=ha_d,
                     color=CHARCOAL, fontstyle="italic")

    ax.axvline(0, color=CHARCOAL, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(ROBUST_GENES, fontsize=8.5, fontstyle="italic")
    ax.set_xlabel("normalised magnitude\n"
                   "top bar: framework β₂ — bottom bar: Supp T11 Δ",
                   fontsize=6.5)
    ax.set_xlim(-1.5, 1.7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_title(
        f"Supp T11: {sup['n_agree']}/{sup['n_compared_directional']} sign agree; "
        f"signed-mag ρ = {sup['signed_magnitude_spearman']:.2f}",
        fontsize=7.5, color=CHARCOAL, pad=4)
    leg_handles = [
        mpatches.Patch(facecolor="white", edgecolor=CHARCOAL,
                        label="framework β₂  (top bar)"),
        mpatches.Patch(facecolor="white", edgecolor=CHARCOAL, hatch="///",
                        alpha=0.5, label="Supp T11 Δ  (bottom bar)"),
    ]
    ax.legend(handles=leg_handles, fontsize=6.5, loc="upper center",
               bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)


def panel_E(ax) -> None:
    """GSEA prerank bars — full library names, q-color encoding."""
    d = json.loads(GSEA.read_text())
    rows = []
    rea = d.get("prerank_per_library", {}).get("Reactome_2022", {})
    for t in rea.get("top_terms", [])[:5]:
        rows.append({
            "term": t.get("Term", "?").split(" R-HSA")[0][:36],
            "nes": t.get("NES"),
            "q": t.get("FDR q-val"),
            "library": "Reactome",
        })
    hall = d.get("prerank_per_library", {}).get("MSigDB_Hallmark_2020", {})
    for t in hall.get("top_terms", [])[:3]:
        rows.append({
            "term": t.get("Term", "?")[:36],
            "nes": t.get("NES"),
            "q": t.get("FDR q-val"),
            "library": "Hallmark",
        })
    rows = [r for r in rows if r["nes"] is not None]
    rows = sorted(rows, key=lambda r: r["nes"])

    y = np.arange(len(rows))
    nes = np.array([r["nes"] for r in rows])
    qs = np.array([r["q"] if r["q"] is not None else 1.0 for r in rows])

    def col_for(q):
        if q < 0.05:
            return TEAL_LHD_DARK
        if q < 0.25:
            return TEAL_LHD
        return LIGHT_GRAY
    bar_colors = [col_for(q) for q in qs]
    ax.barh(y, nes, color=bar_colors, edgecolor=CHARCOAL, linewidth=0.5,
             height=0.7)
    for i, r in enumerate(rows):
        q = r["q"]
        suffix = f"q={q:.2f}" if q is not None else ""
        ax.text(0 + 0.04 * np.sign(nes[i]), y[i], suffix,
                 fontsize=5.5, va="center",
                 ha="left" if nes[i] > 0 else "right", color=CHARCOAL)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['term']} [{r['library']}]" for r in rows],
                        fontsize=6.5)
    ax.axvline(0, color=CHARCOAL, lw=0.7)
    ax.set_xlabel("NES\n(− portal-shifting tail  /  + central-shifting tail)",
                   fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)

    leg = [
        mpatches.Patch(color=TEAL_LHD_DARK, label="FDR q < 0.05"),
        mpatches.Patch(color=TEAL_LHD, label="0.05 ≤ q < 0.25"),
        mpatches.Patch(color=LIGHT_GRAY, label="q ≥ 0.25"),
    ]
    ax.legend(handles=leg, fontsize=6.5, loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, -0.27), ncol=3)
    ax.set_title("GSEA prerank on β₂  (n=66, directional)",
                  fontsize=7.5, color=CHARCOAL, pad=4)
    pad = (np.max(np.abs(nes)) + 0.1) * 1.05
    ax.set_xlim(-pad, pad)


def main() -> None:
    fig = plt.figure(figsize=(mm(220), mm(160)))
    # Independent GridSpecs per row. Row 2 uses left/right margins that
    # leave a large gap in the middle so Panel D's long pathway-name
    # y-tick labels don't crash into Panel C's annotations.
    gs_top = GridSpec(1, 6, figure=fig, wspace=1.10,
                       left=0.07, right=0.97, bottom=0.55, top=0.93)

    # Row 2: place C and D as two separate axes with explicit positions
    # to control the gap precisely.
    # Panel C left half (x: 0.07–0.40), Panel D right half (x: 0.55–0.97).
    # The large gap in the middle (0.40–0.55) absorbs Panel D's long
    # pathway-name y-tick labels.

    ax_B = fig.add_subplot(gs_top[0, 0:2])
    panel_B(ax_B)
    panel_label(ax_B, "A", x=-0.30, y=1.12)

    panel_C(fig, gs_top[0, 2:6])
    ax_label_C = fig.add_subplot(gs_top[0, 2:6])
    ax_label_C.set_axis_off()
    panel_label(ax_label_C, "B", x=-0.06, y=1.05)

    ax_D = fig.add_axes((0.07, 0.10, 0.30, 0.36))
    panel_D(ax_D)
    panel_label(ax_D, "C", x=-0.18, y=1.10)

    ax_E = fig.add_axes((0.62, 0.10, 0.34, 0.36))
    panel_E(ax_E)
    panel_label(ax_E, "D", x=-0.55, y=1.10)

    out = FIGURES / "fig4_steatosis.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] Wrote {out}")


if __name__ == "__main__":
    main()
