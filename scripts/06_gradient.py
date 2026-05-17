"""Compute the per-gene spatial gradient over each axis variant and bin count.

Purpose: for each axis variant in {s, s_alt, s_v_C} (and s_v_B if evaluable) and
each bin count in the sweep, compute the bin-mean dg/ds, the OLS slope β, and the
monotonicity percentage per gene — the workhorse gradient primitive Stage B uses.

Reads:
  - ``results/visium_human.h5ad``

Writes:
  - ``results/per_gene_gradient.parquet`` (cohort + per-donor rows; strict +
    relaxed panels; bin-count sweep N∈{30,50,100,150})

Feeds: the per-gene gradient primitive Methods §4.5; consumed by steps 07, 08,
09, and Fig 2B/2C. B1 monotonicity 37/66 lives here.

Note: step 05 does not write ``obs.s_v_B``, so the column is genuinely
absent when this script runs — the ``s_v_B``-not-all-NaN gate below evaluates to
False and the s_v_B axis is not added (matches legacy B6, which ran before B11).

Absorbs legacy ``stage_B/B6_per_gene_gradient.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/sgd package importable without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed

from sgd.config import RESULTS, LHD_SAMPLES
from sgd.gradient import (N_BINS_DEFAULT, N_BINS_SWEEP, SIGMA_BINS,
                          per_gene_gradient, quantile_bin_per_donor_pool)
from sgd.panels import analysis_gene_panel, lhd_analysis_mask

warnings.filterwarnings("ignore")
np.random.seed(0)

VISIUM = RESULTS / "visium_human.h5ad"
OUT = RESULTS / "per_gene_gradient.parquet"
N_JOBS = 16

AXES_TO_RUN = ("s", "s_alt", "s_v_C")  # s_v_B added if not all-NaN


def run_one(adata_lhd: ad.AnnData, axis: str, n_bins: int) -> pd.DataFrame:
    """Cohort-pooled per-gene gradient for one (axis, n_bins) combo."""
    Gs, bc, cpb, _ = quantile_bin_per_donor_pool(
        adata_lhd, s_col=axis, n_bins=n_bins, sigma=SIGMA_BINS)
    if np.isnan(bc).all() or len(bc) < 3:
        return pd.DataFrame()
    res = per_gene_gradient(Gs, bc)
    df = pd.DataFrame({
        "gene": adata_lhd.var_names,
        "axis": axis,
        "n_bins": n_bins,
        "bin_mean_dgds": res["bin_mean_dgds"],
        "slope": res["slope"],
        "monotonicity": res["monotonicity"],
        "n_bins_actual": np.full(adata_lhd.n_vars, len(bc)),
        "cpb_min": int(cpb.min()) if len(cpb) else 0,
    })
    return df


def main() -> None:
    print(f"[B6] Loading {VISIUM}")
    adata = sc.read_h5ad(VISIUM)
    panel_strict = analysis_gene_panel(adata, mode="strict")
    panel_relaxed = analysis_gene_panel(adata, mode="relaxed")
    print(f"[B6]   strict panel: {int(panel_strict.sum())} genes; "
          f"relaxed panel: {int(panel_relaxed.sum())} genes")
    spot_mask = lhd_analysis_mask(adata)
    sub = adata[spot_mask, panel_strict].copy()
    print(f"[B6]   LHD ∧ ¬fibrotic spots: {sub.n_obs} × strict-panel genes: {sub.n_vars}")

    axes = list(AXES_TO_RUN)
    if "s_v_B" in sub.obs.columns and not sub.obs["s_v_B"].isna().all():
        axes.append("s_v_B")
    print(f"[B6] Axes: {axes}")
    print(f"[B6] Bin counts: {N_BINS_SWEEP}")

    grid = [(axis, nb) for axis in axes for nb in N_BINS_SWEEP]
    print(f"[B6] Running {len(grid)} (axis, n_bins) cells in parallel (n_jobs={N_JOBS})...")
    parts = Parallel(n_jobs=min(N_JOBS, len(grid)), backend="loky", verbose=5)(
        delayed(run_one)(sub, axis, nb) for axis, nb in grid
    )
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df["panel"] = "strict"
    print(f"[B6] Aggregated rows (strict panel): {len(df)}")

    # Relaxed-panel secondary computation at default n_bins on axis s only.
    sub_rel = adata[spot_mask, panel_relaxed].copy()
    print(f"[B6] Relaxed panel cohort run: {sub_rel.n_obs} × {sub_rel.n_vars}")
    rel_part = run_one(sub_rel, axis="s", n_bins=N_BINS_DEFAULT)
    if not rel_part.empty:
        rel_part["panel"] = "relaxed"
        df = pd.concat([df, rel_part], ignore_index=True)

    # --- Per-donor estimates (default n_bins only) for B9.
    # Run the per-gene gradient on each donor in isolation. The pooling
    # function above takes a multi-donor AnnData; for per-donor we just
    # subset.
    print("[B6] Per-donor pipeline at default n_bins (for B9)...")
    samples_arr = sub.obs["sample_id"].astype(str).to_numpy()
    donor_rows = []
    for d in sorted(np.unique(samples_arr)):
        if d not in LHD_SAMPLES:
            continue
        m = samples_arr == d
        sub_d = sub[m].copy()
        Gs, bc, _, _ = quantile_bin_per_donor_pool(
            sub_d, s_col="s", n_bins=N_BINS_DEFAULT, sigma=SIGMA_BINS)
        if np.isnan(bc).all() or len(bc) < 3:
            continue
        res = per_gene_gradient(Gs, bc)
        donor_rows.append(pd.DataFrame({
            "gene": sub.var_names,
            "axis": "s",
            "n_bins": N_BINS_DEFAULT,
            "donor": d,
            "bin_mean_dgds": res["bin_mean_dgds"],
            "slope": res["slope"],
            "monotonicity": res["monotonicity"],
            "panel": "strict",
        }))
    per_donor_df = pd.concat(donor_rows, ignore_index=True) if donor_rows else pd.DataFrame()

    # Save both as a partitioned parquet table.
    df["scope"] = "cohort"
    if not per_donor_df.empty:
        per_donor_df["scope"] = "per_donor"
        df = pd.concat([df, per_donor_df], ignore_index=True)
    df.to_parquet(OUT, compression="zstd", index=False)
    print(f"[B6] Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB; rows={len(df)})")


if __name__ == "__main__":
    main()
