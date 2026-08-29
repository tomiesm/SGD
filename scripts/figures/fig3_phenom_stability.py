"""Figure 3 — Phenomenological coefficient stability (revised v2).

Five panels: A rank saturation, B descriptive bootstrap stability,
C exhaustive donor-partition disagreement, D observed platform-split W entries,
E numerical nullspace dimension.
Layout: 3 panels top (A, B, C) + 2 panels bottom (D, E).
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

from sgd.config import FIGURES, RESULTS
from sgd.plotstyle import (CHARCOAL, GRID_GRAY, LIGHT_GRAY, RED_CENTRAL,
                           TEAL_LHD, TEAL_LHD_DARK, apply_style, mm,
                           panel_label)

warnings.filterwarnings("ignore")
apply_style()

PHENOM = RESULTS / "phenom_stability.json"


def load_grids():
    d = json.loads(PHENOM.read_text())
    sv = d["standard_visium_grid"]
    hd = d["hd_grid"]
    sv_rows = []
    for k, v in sv.items():
        if not isinstance(v, dict) or v.get("status") != "ok":
            continue
        sv_rows.append({
            "N": int(k), "platform": "Visium",
            "panel_size": v["n_genes"],
            "rank": v["effective_rank"],
            "nullspace": v.get("nullspace_dimensionality", v["n_genes"] - v["effective_rank"]),
            "frac_stable": v["frac_W_stable_z_gt_196"],
        })
    hd_rows = []
    for k, v in hd.items():
        if not isinstance(v, dict) or v.get("status") != "ok":
            continue
        hd_rows.append({
            "N": int(k), "platform": "Visium HD",
            "panel_size": v["n_genes"],
            "rank": v["effective_rank"],
            "nullspace": v.get("nullspace_dimensionality", v["n_genes"] - v["effective_rank"]),
            "frac_stable": v["frac_W_stable_z_gt_196"],
        })
    return d, pd.DataFrame(sv_rows), pd.DataFrame(hd_rows)


def panel_A(ax, sv_df, hd_df) -> None:
    """Effective rank vs N — log x; ceilings labeled clearly."""
    sv = sv_df.sort_values("N")
    hd = hd_df.sort_values("N")
    ax.plot(sv["N"], sv["rank"], "o-", color=TEAL_LHD_DARK, lw=1.6,
             markersize=5, label="Visium (p=66)")
    ax.plot(hd["N"], hd["rank"], "s-", color=TEAL_LHD, lw=1.6,
             markersize=5, label="Visium HD (p=55)")
    ax.axhline(66, color=TEAL_LHD_DARK, linestyle=":", lw=0.7, alpha=0.7)
    ax.axhline(55, color=TEAL_LHD, linestyle=":", lw=0.7, alpha=0.7)
    # Keep the Visium label at left; place the HD label above the corresponding
    # line at the right, clear of both trajectories.
    ax.text(28, 66.8, "Visium ceiling p=66", color=TEAL_LHD_DARK,
             fontsize=5.5, va="bottom", ha="left", style="italic")
    ax.text(650, 55.8, "HD ceiling p=55", color=TEAL_LHD,
             fontsize=5.5, va="bottom", ha="right", style="italic")
    ax.set_xscale("log")
    ax.set_xlabel("bin count N", fontsize=7)
    ax.set_ylabel("effective rank", fontsize=7)
    # Thinned ticks to avoid 100/150/200 collisions on log axis
    ax.set_xticks([30, 50, 100, 200, 500])
    ax.set_xticklabels(["30", "50", "100", "200", "500"], fontsize=6.5)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlim(25, 700)
    ax.set_ylim(20, 78)
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)


def panel_B(ax, sv_df, hd_df) -> None:
    """Descriptive fraction of entries large relative to bootstrap spread."""
    sv = sv_df.sort_values("N")
    hd = hd_df.sort_values("N")
    ax.plot(sv["N"], sv["frac_stable"], "o-", color=TEAL_LHD_DARK, lw=1.5,
             markersize=5, label="Visium")
    ax.plot(hd["N"], hd["frac_stable"], "s-", color=TEAL_LHD, lw=1.5,
             markersize=5, label="Visium HD")
    ax.set_xscale("log")
    ax.set_xlabel("bin count N", fontsize=7)
    ax.set_ylabel("fraction of W entries\nwith |z| > 1.96", fontsize=7)
    ax.set_xticks([30, 50, 100, 200, 500])
    ax.set_xticklabels(["30", "50", "100", "200", "500"], fontsize=6.5)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlim(25, 700)
    ax.set_ylim(0, max(0.20, 1.25 * pd.concat([sv, hd])["frac_stable"].max()))
    ax.legend(fontsize=6.5, loc="upper left", frameon=False)
    ax.text(0.98, 0.96,
             r"descriptive threshold:" "\n" r"$|\bar W|/SD_{boot}>1.96$",
             transform=ax.transAxes, fontsize=6.5, va="top", ha="right",
             color=CHARCOAL, fontstyle="italic",
             bbox=dict(facecolor="white", edgecolor="#CCCCCC",
                        linewidth=0.5, boxstyle="round,pad=0.25"))


def panel_C(ax, d_full) -> None:
    """Donor-split Frob ratio — narrow y-range to emphasise flatness."""
    grid = d_full["donor_split_stability_grid"]
    rows = sorted([(int(k), v) for k, v in grid.items()])
    Ns = [r[0] for r in rows]
    means = [r[1]["mean_frobenius_ratio"] for r in rows]
    p25 = [r[1]["p25"] for r in rows]
    p75 = [r[1]["p75"] for r in rows]
    ax.fill_between(Ns, p25, p75, color=TEAL_LHD, alpha=0.25, linewidth=0,
                     label="IQR")
    ax.plot(Ns, means, "o-", color=TEAL_LHD_DARK, lw=1.6, markersize=5,
             label=f"mean (all {d_full['n_donor_splits']} unique splits)")
    ax.axhline(1.0, color=CHARCOAL, lw=0.6, linestyle=":")
    ax.text(155, 1.025, "difference = mean matrix norm",
             fontsize=5.5, color=LIGHT_GRAY, ha="right", va="bottom",
             style="italic")
    ax.set_xticks([30, 50, 100, 150])
    ax.set_xticklabels(["30", "50", "100", "150"], fontsize=6.5)
    ax.set_xlim(20, 165)
    ax.set_xlabel("bin count N", fontsize=7)
    ax.set_ylabel(r"$2\Vert W_a-W_b\Vert/(\Vert W_a\Vert+\Vert W_b\Vert)$", fontsize=7)
    # Tighter y range — emphasises the flat ≈1.4 result
    ax.set_ylim(0.95, 1.55)
    ax.legend(fontsize=6.5, loc="upper right", frameon=False)


def panel_D(ax, d_full) -> None:
    """Observed platform-split entry correlations in both matched donors."""
    platform = d_full["platform_split_M2_M6"]
    plotted: list[tuple[str, float, int]] = []
    all_values: list[np.ndarray] = []
    colours = {"M2": CHARCOAL, "M6": TEAL_LHD}
    for donor in ("M2", "M6"):
        info = platform.get(donor, {})
        if info.get("status") != "ok":
            continue
        x = np.asarray(info["W_visium_entries"], dtype=float)
        y = np.asarray(info["W_hd_entries"], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        # Standardise within donor and platform for readable common limits;
        # rank correlation is invariant to this transformation. Every point is
        # an observed fitted entry, not a cloud simulated to match a target rho.
        x = (x - x.mean()) / max(x.std(), 1e-12)
        y = (y - y.mean()) / max(y.std(), 1e-12)
        ax.scatter(x, y, s=2.0, alpha=0.22, color=colours[donor],
                   edgecolor="none", label=donor)
        all_values.extend((x, y))
        plotted.append((donor, float(info["spearman_rho_W_entries"]), len(x)))
    ax.axhline(0, color=GRID_GRAY, lw=0.6)
    ax.axvline(0, color=GRID_GRAY, lw=0.6)
    lim = max(2.5, float(np.percentile(np.abs(np.concatenate(all_values)), 99)))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("standardised W entry — Visium", fontsize=7)
    ax.set_ylabel("standardised W entry — Visium HD", fontsize=7)
    n_total = sum(n for _, _, n in plotted)
    annotation = "\n".join(
        f"{donor}: Spearman $\\rho$ = {rho:.3f}"
        for donor, rho, _ in plotted
    )
    # Keep the numerical summary outside the data cloud.  Earlier versions
    # placed an opaque annotation box inside the axes, obscuring observations.
    ax.set_title(
        f"matched-donor platform splits (n = {n_total:,} observed entries)\n"
        + annotation.replace("\n", "   "),
        fontsize=7, pad=4, linespacing=1.25,
    )
    ax.legend(fontsize=6.2, loc="upper center", frameon=False, ncol=2,
              bbox_to_anchor=(0.5, -0.24), markerscale=3,
              handletextpad=0.3, columnspacing=1.2)


def panel_E(ax, sv_df, hd_df) -> None:
    """Nullspace dimension — grouped x-axis under Visium / Visium HD brackets."""
    sv = sv_df.sort_values("N").reset_index(drop=True)
    hd = hd_df.sort_values("N").reset_index(drop=True)
    Ns = list(sv["N"]) + list(hd["N"])
    nulls = list(sv["nullspace"]) + list(hd["nullspace"])
    n_sv = len(sv)
    n_hd = len(hd)
    colors = [TEAL_LHD_DARK] * n_sv + [TEAL_LHD] * n_hd

    x_pos = np.arange(len(Ns))
    bars = ax.bar(x_pos, nulls, color=colors, edgecolor=CHARCOAL,
                   linewidth=0.6)
    for b, v in zip(bars, nulls):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{int(v)}",
                 ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(n) for n in Ns], fontsize=6.5)
    ax.set_ylabel("nullspace dimension\n(p − effective rank)", fontsize=7)
    # Platform brackets already label the grouped x axis; a second xlabel
    # collides with those labels in the journal-width layout.
    ax.set_xlabel("")
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, max(nulls) + 8)

    # Group brackets at bottom with platform labels
    y_bracket = -0.15
    # Visium bracket
    ax.annotate("", xy=(n_sv - 0.7, y_bracket), xytext=(0 - 0.3, y_bracket),
                 xycoords=("data", "axes fraction"),
                 textcoords=("data", "axes fraction"),
                 arrowprops=dict(arrowstyle="-",
                                   connectionstyle="bar,fraction=-0.10",
                                   color=TEAL_LHD_DARK, lw=0.8))
    ax.text((n_sv - 1) / 2, y_bracket - 0.06, "Visium (p=66)",
             transform=ax.get_xaxis_transform(),
             fontsize=6.5, color=TEAL_LHD_DARK,
             ha="center", va="top", fontweight="bold")
    # Visium HD bracket
    ax.annotate("", xy=(len(Ns) - 1 + 0.3, y_bracket),
                 xytext=(n_sv - 0.3, y_bracket),
                 xycoords=("data", "axes fraction"),
                 textcoords=("data", "axes fraction"),
                 arrowprops=dict(arrowstyle="-",
                                   connectionstyle="bar,fraction=-0.10",
                                   color=TEAL_LHD, lw=0.8))
    ax.text((n_sv - 1 + len(Ns) - 1) / 2 + 0.5, y_bracket - 0.06,
             "Visium HD (p=55)",
             transform=ax.get_xaxis_transform(),
             fontsize=6.5, color=TEAL_LHD, ha="center", va="top",
             fontweight="bold")

    ax.set_title("Nullspace dimension", fontsize=7.5, pad=4)


def main() -> None:
    d_full, sv_df, hd_df = load_grids()

    fig = plt.figure(figsize=(mm(190), mm(135)))
    # 3 panels top + 2 panels bottom; 6-col grid for fine layout.
    gs = GridSpec(2, 6, figure=fig, hspace=1.05, wspace=1.05,
                   height_ratios=[1.0, 1.0],
                   left=0.07, right=0.97, bottom=0.13, top=0.94)

    ax_A = fig.add_subplot(gs[0, 0:2])
    ax_B = fig.add_subplot(gs[0, 2:4])
    ax_C = fig.add_subplot(gs[0, 4:6])
    ax_D = fig.add_subplot(gs[1, 0:3])
    ax_E = fig.add_subplot(gs[1, 3:6])

    panel_A(ax_A, sv_df, hd_df)
    panel_B(ax_B, sv_df, hd_df)
    panel_C(ax_C, d_full)
    panel_D(ax_D, d_full)
    panel_E(ax_E, sv_df, hd_df)

    panel_label(ax_A, "A", x=-0.20, y=1.05)
    panel_label(ax_B, "B", x=-0.20, y=1.05)
    panel_label(ax_C, "C", x=-0.20, y=1.05)
    panel_label(ax_D, "D", x=-0.09, y=1.13)
    panel_label(ax_E, "E", x=-0.13, y=1.05)

    out = FIGURES / "fig3_phenom_stability.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] Wrote {out}")


if __name__ == "__main__":
    main()
