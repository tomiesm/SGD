"""
09 — W-matrix phenomenological-coefficient stability (Methods §4.7).

Absorbs the legacy C1_phenom_stability script in full — all six
W-stability diagnostics. The analysis body is copied verbatim; the only
changes are import rewiring to ``src/sgd/`` and hard-coded paths replaced
by ``sgd.config`` constants.

Six diagnostics evaluated at each bin count:
  Standard Visium: N_bins ∈ {30, 50, 100, 150} on the strict 66-gene panel.
  Visium HD: N_bins ∈ {200, 500} on M2 + M6 path-A pre-segmented, on the
    HD ∩ strict-66 = 55-gene panel.

  1. Effective rank of the binned design matrix.
  2. Block bootstrap of Ridge-fit W (200 resamples, λ from LOOCV CV).
  3. Donor-split stability (1000 4-vs-4 splits) — standard Visium only.
  4. Platform-split stability (Visium-vs-HD on M2+M6 at N=200, 55-gene panel).
  5. Gene-label permutation baseline (negative control for #2).
  6. Nullspace dimensionality from #1's SVD.

Reads:
  RESULTS/visium_human.h5ad
  RESULTS/visium_hd.h5ad

Writes:
  RESULTS/phenom_stability.json

Feeds: the entire W-matrix story §4.7 — all of R3.1-R3.7 and all five
       Fig 3 panels read this one artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/sgd package importable without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from sgd.config import RESULTS
from sgd.confounds import edge_spots_mask
from sgd.gradient import SIGMA_BINS, per_gene_gradient, quantile_bin_per_donor_pool
from sgd.panels import (CENTRAL_MARKERS, PORTAL_MARKERS, _gene_score,
                        analysis_gene_panel, lhd_analysis_mask)
from sgd.wmatrix import (N_BINS_HD, N_BINS_STANDARD, RIDGE_LAMBDA_GRID,
                         apply_donor_aware_model_A, autocorr_length,
                         block_bootstrap_W, donor_split_W_distance,
                         effective_rank, fit_W_ridge, hd_path_a_donor_subset,
                         per_cell_log_count_residuals,
                         select_ridge_lambda_loocv)

warnings.filterwarnings("ignore")
np.random.seed(0)

# --- Paths (legacy hard-coded DATA/* -> sgd.config constants) ---
VISIUM = RESULTS / "visium_human.h5ad"
VISIUM_HD = RESULTS / "visium_hd.h5ad"
OUT = RESULTS / "phenom_stability.json"

# --- Constants (verbatim from C1_phenom_stability.py) ---
N_JOBS = 16
N_BLOCK_BOOTSTRAP = 200
N_PERMUTATION = 100
N_DONOR_SPLITS = 1000


def standard_visium_panel(adata: ad.AnnData):
    """
    LHD ∧ ¬fibrotic, strict 66-gene panel, **donor-aware model-A
    residualised** X (donor + log(total_umi_raw) per Stage B B7).
    Required for parity with HD-side per-cell log_count residualisation
    (otherwise platform-split W comparisons mostly compare preprocessing
    scales).
    """
    panel = analysis_gene_panel(adata, mode="strict")
    spot_mask = lhd_analysis_mask(adata)
    sub = adata[spot_mask, panel].copy()
    return apply_donor_aware_model_A(sub, donor_col="donor_id"), panel


def grid_cell(sub: ad.AnnData, axis: str, n_bins: int) -> dict:
    """Compute diagnostics 1, 2, 5, 6 for one (axis, n_bins) cell."""
    Gs, bc, cpb, _ = quantile_bin_per_donor_pool(
        sub, s_col=axis, n_bins=n_bins, sigma=SIGMA_BINS)
    if np.isnan(bc).all() or len(bc) < 3:
        return {"status": "binning_failed"}
    # Drop NaN bin centres (quantile-edge collapse) — propagates through
    # bc.mean() to slope denominator otherwise.
    valid_bin = ~np.isnan(bc)
    if valid_bin.sum() < 3:
        return {"status": "too_few_valid_bins", "n_valid": int(valid_bin.sum())}
    Gs = Gs[valid_bin]
    bc = bc[valid_bin]
    cpb = cpb[valid_bin]
    grad = per_gene_gradient(Gs, bc)
    dgds = grad["dgds_per_bin"]
    n_genes = Gs.shape[1]

    # Diagnostic 1: effective rank.
    eff_rank = effective_rank(Gs)
    # Diagnostic 6: nullspace dimensionality = panel_size − effective rank.
    nullspace_dim = int(n_genes - eff_rank)

    # Ridge λ via LOOCV (only meaningful at N_bins ≥ 4).
    if Gs.shape[0] < 4:
        return {"status": "too_few_bins", "n_bins_actual": int(Gs.shape[0])}
    best_lam, mse_grid = select_ridge_lambda_loocv(Gs, dgds)

    # Diagnostic 2: block bootstrap of W.
    block_lags = [autocorr_length(dgds[:, gi]) for gi in range(min(n_genes, 50))]
    block_size = max(1, int(np.median(block_lags) * 1.5))
    W_mean, W_std = block_bootstrap_W(
        Gs, dgds, lam=best_lam, n_boot=N_BLOCK_BOOTSTRAP,
        block_size=block_size, rng_seed=42)
    z = np.abs(W_mean) / (W_std + 1e-12)
    n_stable = int((z > 1.96).sum())
    frac_stable = float(n_stable / max(1, z.size))

    # Diagnostic 5: gene-label permutation baseline.
    rng = np.random.RandomState(7)
    perm_frac_stable = []
    for _ in range(N_PERMUTATION):
        perm = rng.permutation(n_genes)
        Wp_mean, Wp_std = block_bootstrap_W(
            Gs, dgds[:, perm], lam=best_lam, n_boot=20,
            block_size=block_size, rng_seed=int(rng.randint(0, 2**31 - 1)))
        zp = np.abs(Wp_mean) / (Wp_std + 1e-12)
        perm_frac_stable.append(float((zp > 1.96).sum() / max(1, zp.size)))

    return {
        "status": "ok",
        "n_bins_target": n_bins,
        "n_bins_actual": int(Gs.shape[0]),
        "n_genes": int(n_genes),
        "ridge_lambda_loocv": float(best_lam),
        "ridge_mse_grid": [float(m) for m in mse_grid],
        "effective_rank": int(eff_rank),
        "nullspace_dimensionality": nullspace_dim,
        "block_bootstrap_block_size": int(block_size),
        "frac_W_stable_z_gt_196": float(frac_stable),
        "n_W_stable_z_gt_196": n_stable,
        "permutation_frac_stable_mean": float(np.mean(perm_frac_stable)),
        "permutation_frac_stable_p95": float(np.percentile(perm_frac_stable, 95)),
        "min_cells_per_bin": int(cpb.min()) if len(cpb) else 0,
    }


def platform_split(adata_visium: ad.AnnData, adata_hd: ad.AnnData,
                    sample: str, n_bins: int = 200) -> dict:
    """
    Diagnostic 4: platform split on one donor, fitting W on each platform's
    binned data over the HD ∩ strict-66 = 55-gene panel.
    """
    panel_strict = analysis_gene_panel(adata_visium, mode="strict")
    strict_genes = list(adata_visium.var_names[panel_strict])

    # HD-side subset. Build axis on the FULL gene set (all 9 markers
    # present per Stage A) BEFORE restricting to the strict-66 panel —
    # the panel doesn't contain all 9 axis markers (only 3 of 9 in strict).
    sub_hd_full = hd_path_a_donor_subset(adata_hd, sample)
    if sub_hd_full.n_obs < n_bins * 2:
        return {"status": "insufficient_HD_data", "n_cells": int(sub_hd_full.n_obs)}
    P = _gene_score(sub_hd_full, PORTAL_MARKERS)
    C = _gene_score(sub_hd_full, CENTRAL_MARKERS)
    if np.isnan(P).all() or np.isnan(C).all():
        return {"status": "HD_axis_markers_absent"}
    Pz = (P - np.nanmean(P)) / (np.nanstd(P) + 1e-9)
    Cz = (C - np.nanmean(C)) / (np.nanstd(C) + 1e-9)
    delta = Cz - Pz
    ranks = pd.Series(delta).rank(method="average").to_numpy() - 1
    s_hd = ranks / max(1, len(ranks) - 1)
    cy = _gene_score(sub_hd_full, ("CYP2E1", "CYP1A2"))
    if not np.isnan(cy).all():
        if cy[s_hd > 0.5].mean() < cy[s_hd <= 0.5].mean():
            s_hd = 1.0 - s_hd
    sub_hd_full.obs["s"] = s_hd
    # Now restrict to the panel intersection for binning.
    hd_genes_in_strict = [g for g in strict_genes if g in sub_hd_full.var_names]
    if len(hd_genes_in_strict) < 5:
        return {"status": "too_few_panel_genes", "n_genes": len(hd_genes_in_strict)}
    sub_hd = sub_hd_full[:, hd_genes_in_strict].copy()

    # HD: harmony-corrected per Stage A A3 — use as-is (no per-cell residualisation).
    Gs_hd, bc_hd, _, _ = quantile_bin_per_donor_pool(
        sub_hd, s_col="s", n_bins=n_bins, sigma=SIGMA_BINS,
        sample_col="sample_id" if "sample_id" in sub_hd.obs.columns else "donor_id")
    valid_bin_hd = ~np.isnan(bc_hd)
    if valid_bin_hd.sum() < 3:
        return {"status": "HD_too_few_valid_bins"}
    Gs_hd = Gs_hd[valid_bin_hd]
    bc_hd = bc_hd[valid_bin_hd]
    grad_hd = per_gene_gradient(Gs_hd, bc_hd)

    # Visium-standard side on the same panel restricted to common genes,
    # for this donor only. Apply C5 edge mask (per Stage B forward-flag for
    # M2/M6 cross-platform work) and donor-aware model-A residualisation
    # for parity with HD-side per-cell log_count residualisation.
    sample_mask = (adata_visium.obs["sample_id"].astype(str) == sample) \
                   & lhd_analysis_mask(adata_visium)
    sub_v_full = adata_visium[sample_mask, :].copy()
    coords = np.asarray(sub_v_full.obsm["spatial"])
    edge = edge_spots_mask(coords, edge_frac=0.05)
    keep_v = ~edge
    sub_v_keep = sub_v_full[keep_v].copy()
    sub_v_keep = sub_v_keep[:, hd_genes_in_strict].copy()
    sub_v_resid = apply_donor_aware_model_A(sub_v_keep, donor_col="donor_id")
    Gs_v, bc_v, _, _ = quantile_bin_per_donor_pool(
        sub_v_resid, s_col="s", n_bins=n_bins, sigma=SIGMA_BINS)
    valid_bin_v = ~np.isnan(bc_v)
    if valid_bin_v.sum() < 3:
        return {"status": "Visium_too_few_valid_bins"}
    Gs_v = Gs_v[valid_bin_v]
    bc_v = bc_v[valid_bin_v]
    grad_v = per_gene_gradient(Gs_v, bc_v)

    # Ridge fits, same λ on both for parity (chosen on Visium-standard side).
    lam, _ = select_ridge_lambda_loocv(Gs_v, grad_v["dgds_per_bin"])
    W_v, _ = fit_W_ridge(Gs_v, grad_v["dgds_per_bin"], lam)
    W_hd, _ = fit_W_ridge(Gs_hd, grad_hd["dgds_per_bin"], lam)

    # Frobenius distance and per-entry Spearman.
    from scipy.stats import spearmanr
    frob = float(np.linalg.norm(W_v - W_hd) / max(1e-12, np.linalg.norm(W_v)))
    rho, _ = spearmanr(W_v.flatten(), W_hd.flatten())
    return {
        "status": "ok",
        "sample": sample,
        "n_genes": len(hd_genes_in_strict),
        "n_bins_visium_actual": int(Gs_v.shape[0]),
        "n_bins_hd_actual": int(Gs_hd.shape[0]),
        "ridge_lambda_used": float(lam),
        "frobenius_ratio": frob,
        "spearman_rho_W_entries": float(rho),
    }


def _lam_grid_floats():
    return RIDGE_LAMBDA_GRID


def main() -> None:
    print(f"[C1] Loading {VISIUM}")
    adata = sc.read_h5ad(VISIUM)
    sub_std, panel = standard_visium_panel(adata)
    print(f"[C1]   Standard Visium: {sub_std.n_obs} spots × {sub_std.n_vars} strict-panel genes")

    # --- Diagnostics 1, 2, 5, 6 across standard-Visium bin counts ---
    print(f"[C1] Standard-Visium grid (n_bins ∈ {N_BINS_STANDARD})...")
    grid_std = {}
    for nb in N_BINS_STANDARD:
        print(f"[C1]   N_bins = {nb}")
        grid_std[nb] = grid_cell(sub_std, axis="s", n_bins=nb)

    # --- Diagnostic 3: donor-split stability across all standard-Visium bin counts ---
    # Required by the §14 #4 within-standard sub-gate: trend across N_bins.
    print(f"[C1] Donor-split stability ({N_DONOR_SPLITS} splits) "
          f"across N_bins ∈ {N_BINS_STANDARD}...")
    panel_mask = analysis_gene_panel(adata, mode="strict")
    spot_mask = lhd_analysis_mask(adata)
    # Donor-split needs raw obs columns + the residualised X. Build once
    # against the residualised standard-panel AnnData and pass it in.
    sub_lhd = adata[spot_mask].copy()
    sub_lhd_strict = sub_lhd[:, panel_mask].copy()
    sub_lhd_resid = apply_donor_aware_model_A(sub_lhd_strict, donor_col="donor_id")
    donor_split_grid: dict[int, dict] = {}
    for nb in N_BINS_STANDARD:
        print(f"[C1]   N_bins = {nb}")
        # The donor_split_W_distance helper was written to subset by panel_mask
        # internally; we pass an already-strict AnnData via a trivial all-True mask.
        all_true = np.ones(sub_lhd_resid.n_vars, dtype=bool)
        donor_split_grid[nb] = donor_split_W_distance(
            sub_lhd_resid, donor_col="sample_id", s_col="s",
            panel_mask=all_true, n_bins=nb, n_splits=N_DONOR_SPLITS, rng_seed=0)
        print(f"[C1]     mean Frob ratio: "
              f"{donor_split_grid[nb].get('mean_frobenius_ratio')}")
    donor_split = donor_split_grid.get(50, {})  # backward-compatible header

    # --- Diagnostics 1, 2, 5, 6 for HD ---
    grid_hd = {}
    if VISIUM_HD.exists():
        print(f"[C1] Loading {VISIUM_HD}")
        adata_hd = sc.read_h5ad(VISIUM_HD)
        # Subset to path-A pre-segmented for HD diagnostics.
        path_col = (adata_hd.obs.get("platform_path", "").astype(str)
                    if "platform_path" in adata_hd.obs.columns else None)
        if path_col is not None:
            mask_a = (path_col == "hd_segmented").to_numpy()
            adata_hd_a = adata_hd[mask_a].copy()
            print(f"[C1]   HD path A (M2/M6 segmented): {adata_hd_a.n_obs} cells × "
                  f"{adata_hd_a.n_vars} genes")
            # HD axis: same §5.1 recipe per donor across all path-A cells.
            samples_arr = adata_hd_a.obs.get("sample_id", adata_hd_a.obs.get("sample", "")).astype(str).to_numpy()
            adata_hd_a.obs["sample_id"] = samples_arr
            s_hd_full = np.full(adata_hd_a.n_obs, np.nan)
            for sample in np.unique(samples_arr):
                m = samples_arr == sample
                sub = adata_hd_a[m]
                P = _gene_score(sub, PORTAL_MARKERS)
                C = _gene_score(sub, CENTRAL_MARKERS)
                if np.isnan(P).all() or np.isnan(C).all():
                    continue
                Pz = (P - np.nanmean(P)) / (np.nanstd(P) + 1e-9)
                Cz = (C - np.nanmean(C)) / (np.nanstd(C) + 1e-9)
                delta = Cz - Pz
                ranks = pd.Series(delta).rank(method="average").to_numpy() - 1
                s_i = ranks / max(1, len(ranks) - 1)
                cy = _gene_score(sub, ("CYP2E1", "CYP1A2"))
                if not np.isnan(cy).all():
                    if cy[s_i > 0.5].mean() < cy[s_i <= 0.5].mean():
                        s_i = 1.0 - s_i
                s_hd_full[m] = s_i
            adata_hd_a.obs["s"] = s_hd_full
            # HD ∩ strict-66
            hd_panel_genes = [g for g in adata.var_names[panel_mask] if g in adata_hd_a.var_names]
            sub_hd_panel = adata_hd_a[:, hd_panel_genes].copy()
            print(f"[C1]   HD strict ∩ panel: {sub_hd_panel.n_vars} genes")
            for nb in N_BINS_HD:
                print(f"[C1]   HD N_bins = {nb}")
                grid_hd[nb] = grid_cell(sub_hd_panel, axis="s", n_bins=nb)
            # --- Diagnostic 4: platform split on M2 and M6 ---
            # Drop platform-split to N=50 (the §7.1 standard-Visium default)
            # because M6 has only 1338 spots → ~1148 after edge mask, which
            # doesn't quantile-bin cleanly at N=200. Both platforms binned at
            # the same N=50 for parity.
            print("[C1] Platform-split (Visium vs HD) on M2 and M6, N=50, 55-gene panel...")
            platform = {}
            for sample in ("M2", "M6"):
                platform[sample] = platform_split(adata, adata_hd, sample, n_bins=50)
                print(f"[C1]   {sample}: status={platform[sample].get('status')}; "
                      f"frob={platform[sample].get('frobenius_ratio')}; "
                      f"spearman={platform[sample].get('spearman_rho_W_entries')}")
        else:
            platform = {"status": "no_platform_path_obs"}
    else:
        platform = {"status": "visium_hd_h5ad_missing"}
        grid_hd = {}

    # --- Within-standard / within-HD trend gates ---
    def trend_check(grid: dict, key: str, want_increasing: bool) -> dict:
        """Are values monotone (or trending) in the right direction across bin counts?"""
        ks = sorted(int(k) for k in grid.keys() if grid[k].get("status") == "ok")
        if len(ks) < 2:
            return {"status": "insufficient_data"}
        vals = [grid[k][key] for k in ks]
        if want_increasing:
            monotone = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        else:
            monotone = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
        return {"bin_counts": ks, "values": [float(v) for v in vals],
                "monotone": bool(monotone)}

    # Donor-split distance should DECREASE with N_bins.
    donor_split_trend_vals: list[float] = []
    donor_split_trend_bins: list[int] = []
    for nb in N_BINS_STANDARD:
        v = donor_split_grid[nb].get("mean_frobenius_ratio")
        if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
            donor_split_trend_vals.append(float(v))
            donor_split_trend_bins.append(nb)
    if len(donor_split_trend_vals) >= 2:
        donor_split_decreases = all(
            donor_split_trend_vals[i] >= donor_split_trend_vals[i + 1]
            for i in range(len(donor_split_trend_vals) - 1)
        )
    else:
        donor_split_decreases = None
    trends_std = {
        "effective_rank_increases": trend_check(grid_std, "effective_rank", True),
        "frac_W_stable_increases": trend_check(grid_std, "frac_W_stable_z_gt_196", True),
        "donor_split_distance_decreases": {
            "bin_counts": donor_split_trend_bins,
            "values": donor_split_trend_vals,
            "monotone": (bool(donor_split_decreases) if donor_split_decreases is not None
                         else None),
        },
    }
    trends_hd = {
        "effective_rank_increases": trend_check(grid_hd, "effective_rank", True),
        "frac_W_stable_increases": trend_check(grid_hd, "frac_W_stable_z_gt_196", True),
    } if grid_hd else {}

    out = {
        "schema": "stage_C_phenom_stability_v1",
        "n_block_bootstrap": N_BLOCK_BOOTSTRAP,
        "n_donor_splits": N_DONOR_SPLITS,
        "n_permutation": N_PERMUTATION,
        "ridge_lambda_grid": list(_lam_grid_floats()),
        "panel_strict_size": int(sub_std.n_vars),
        "standard_visium_grid": {str(k): v for k, v in grid_std.items()},
        "hd_grid": {str(k): v for k, v in grid_hd.items()},
        "donor_split_stability": donor_split,  # legacy: N=50 only
        "donor_split_stability_grid": {str(k): v for k, v in donor_split_grid.items()},
        "platform_split_M2_M6": platform,
        "trends_within_standard_visium": trends_std,
        "trends_within_HD": trends_hd,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"[C1] Wrote {OUT}")


if __name__ == "__main__":
    main()
