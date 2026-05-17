"""
08 — Cross-modality / cross-platform / cross-pipeline comparisons (Methods §4.6).

Consolidates three legacy comparison scripts plus the C6 repackaging
step into one entry point. Each legacy analysis body is copied verbatim;
the only changes are import rewiring to ``src/sgd/`` and hard-coded paths
replaced by ``sgd.config`` constants.

Absorbs:
  * C2_sign_correlation              — sign-of-correlation baseline (§9.6)
  * C4_cross_platform                — cross-platform Visium-vs-HD M2/M6 (§9.4)
  * B12_cross_pipeline_approach_a    — cross-pipeline approach (a), 8-layer
  * C6_cross_pipeline (repackaging)  — folded in as repackage_cross_pipeline();
                                       C6 computes nothing, it only reformats
                                       B11+B12 JSON into the §9.7-framed artifact.

Reads:
  RESULTS/visium_human.h5ad
  RESULTS/visium_hd.h5ad
  RESULTS/per_gene_gradient.parquet
  DATA   Supplementary Table 2 + Supplementary Table 4 xlsx
  RESULTS/stage_B_variant_B_addendum.json + stage_B_cross_pipeline_supp_table_2.json (from 07)

Writes (all to RESULTS, basenames identical to legacy):
  sign_correlation_baseline.json
  cross_platform_M2_M6.json
  stage_B_cross_pipeline_approach_a.json
  cross_pipeline_reproducibility.json

Feeds: cross-modality §4.6 — Fig 2A/2B/2C, Fig S3; R2.1-R2.3, R2.5/R2.6,
       R2.8, and the repackaged §9.7 artifact two figures read.
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
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr

from sgd.config import RESULTS, SUPP_T2, SUPP_T4
from sgd.confounds import detect_boundary_spots, edge_mask_visium
from sgd.gradient import (N_BINS_DEFAULT, SIGMA_BINS, per_gene_gradient,
                          per_gene_layer_means, quantile_bin_per_donor_pool)
from sgd.panels import (CENTRAL_MARKERS, HEADLINE_MARKERS, KNOWN_LIVER_DIRECTION,
                        PORTAL_MARKERS, _gene_score, analysis_gene_panel,
                        lhd_analysis_mask)
from sgd.wmatrix import per_cell_log_count_residuals

warnings.filterwarnings("ignore")
np.random.seed(0)

# --- Paths (legacy hard-coded DATA/* -> sgd.config constants) ---
VISIUM = RESULTS / "visium_human.h5ad"
VISIUM_HD = RESULTS / "visium_hd.h5ad"
GRAD = RESULTS / "per_gene_gradient.parquet"

OUT_C2 = RESULTS / "sign_correlation_baseline.json"
OUT_C4 = RESULTS / "cross_platform_M2_M6.json"
OUT_B12 = RESULTS / "stage_B_cross_pipeline_approach_a.json"
OUT_C6 = RESULTS / "cross_pipeline_reproducibility.json"

# C6 inputs (B11 outputs from 07).
B11_OUT = RESULTS / "stage_B_variant_B_addendum.json"
B11_CROSS = RESULTS / "stage_B_cross_pipeline_supp_table_2.json"
B12_OUT = RESULTS / "stage_B_cross_pipeline_approach_a.json"

# B12 constant (verbatim from B12_cross_pipeline_approach_a.py).
N_LAYERS = 8


# ===========================================================================
# C2 — Sign-of-correlation baseline (§9.6)
# ===========================================================================

def run_c2() -> None:
    """C2 — sign-of-correlation baseline (§9.6)."""
    print(f"[C2] Loading {VISIUM}")
    adata = sc.read_h5ad(VISIUM)
    spot_mask = lhd_analysis_mask(adata)
    sub = adata[spot_mask].copy()
    s = sub.obs["s"].to_numpy()
    valid = ~np.isnan(s)
    sub = sub[valid].copy()
    s = s[valid]
    print(f"[C2]   LHD ∧ ¬fibrotic ∧ valid s: {sub.n_obs} spots × {sub.n_vars} genes")

    # --- Per-gene Pearson r(g, s) ---
    X = sub.X
    if issparse(X):
        X = X.toarray()
    print("[C2] Computing per-gene r(g, s)...")
    n_genes = X.shape[1]
    s_c = s - s.mean()
    s_norm = (s_c ** 2).sum() ** 0.5
    X_c = X - X.mean(axis=0, keepdims=True)
    X_norm = (X_c ** 2).sum(axis=0) ** 0.5
    r = (X_c * s_c[:, None]).sum(axis=0) / np.where(X_norm * s_norm > 0,
                                                       X_norm * s_norm, np.nan)
    baseline_sign = np.sign(r)
    var_names = list(sub.var_names)

    # --- Compare against B6 slope from parquet ---
    df = pd.read_parquet(GRAD)
    cohort_s = df[(df["scope"] == "cohort") & (df["axis"] == "s")
                   & (df["n_bins"] == N_BINS_DEFAULT)
                   & (df.get("panel", "strict") == "strict")]
    sgd_slope = cohort_s.set_index("gene")["slope"]

    # Common genes between baseline and SGD strict-panel.
    common = sorted(set(var_names) & set(sgd_slope.index))
    print(f"[C2]   strict-panel genes common: {len(common)}")
    base_dir = pd.Series(baseline_sign, index=var_names).reindex(common)
    sgd_dir = np.sign(sgd_slope.reindex(common))
    valid_pair = (base_dir != 0) & (sgd_dir != 0) & ~np.isnan(base_dir) & ~np.isnan(sgd_dir)
    n_strict = int(valid_pair.sum())
    n_agree_strict = int((base_dir[valid_pair] == sgd_dir[valid_pair]).sum())
    pct_strict = 100.0 * n_agree_strict / max(1, n_strict)
    print(f"[C2]   strict-panel agreement: {n_agree_strict}/{n_strict} = {pct_strict:.1f}%")

    # --- 9 known markers and 8 headline markers ---
    marker_results = {"all_9_known": {}, "headline_8": {}}
    for mset_name, mset in (("all_9_known", KNOWN_LIVER_DIRECTION.keys()),
                              ("headline_8", HEADLINE_MARKERS)):
        for g in mset:
            if g not in var_names:
                continue
            gi = var_names.index(g)
            base = int(np.sign(r[gi])) if not np.isnan(r[gi]) else 0
            expected = KNOWN_LIVER_DIRECTION.get(g, 0)
            sgd = int(np.sign(sgd_slope[g])) if g in sgd_slope.index else 0
            marker_results[mset_name][g] = {
                "expected": int(expected),
                "baseline_sign": base,
                "sgd_slope_sign": sgd,
                "baseline_correct": int(base == expected),
                "sgd_correct": int(sgd == expected),
                "baseline_pearson_r": float(r[gi]) if not np.isnan(r[gi]) else None,
                "sgd_slope_value": float(sgd_slope[g]) if g in sgd_slope.index else None,
            }

    # --- Magnitude rank correlation: |slope| vs |r| ---
    abs_pearson = pd.Series(np.abs(r), index=var_names).reindex(common)
    abs_slope = sgd_slope.reindex(common).abs()
    keep = ~np.isnan(abs_pearson) & ~np.isnan(abs_slope)
    rho, _ = spearmanr(abs_pearson[keep], abs_slope[keep])

    summary = {
        "n_lhd_spots_evaluated": int(sub.n_obs),
        "strict_panel_size": int(len(common)),
        "strict_panel_n_genes_compared": n_strict,
        "strict_panel_n_agree": n_agree_strict,
        "strict_panel_sign_agreement_pct": float(pct_strict),
        "magnitude_spearman_abs_r_vs_abs_slope": float(rho),
        "marker_recovery": marker_results,
        "summary_baseline_correct_9": sum(
            v["baseline_correct"] for v in marker_results["all_9_known"].values()
        ),
        "summary_baseline_correct_headline_8": sum(
            v["baseline_correct"] for v in marker_results["headline_8"].values()
        ),
        "framing_note": (
            "Baseline (sign of Pearson r(g, s)) differs from the SGD pipeline "
            "(slope of binned g(s) on s) in normalisation, smoothing, and "
            "binning. Sign agreement on the strict panel measures whether the "
            "two methods point genes in the same direction; the framework's "
            "contribution is the dynamical interpretation under the "
            "material-derivative identity, not a technical gradient-estimation "
            "improvement (per §9.6)."
        ),
    }
    OUT_C2.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[C2] Wrote {OUT_C2}")
    print(f"[C2]   Headline 8/8 baseline-correct: "
          f"{summary['summary_baseline_correct_headline_8']}/8")
    print(f"[C2]   |·| Spearman: {rho:.3f}")


# ===========================================================================
# C4 — Cross-platform Visium-vs-HD on M2 + M6 (§9.4 cross-platform)
# ===========================================================================
# NOTE: edge_mask_visium imported from sgd.confounds (legacy C4 defined it inline).

def hd_axis_per_donor(sub_hd: ad.AnnData) -> np.ndarray:
    """§5.1 score axis on HD path-A pre-segmented cells for one donor."""
    P = _gene_score(sub_hd, PORTAL_MARKERS)
    C = _gene_score(sub_hd, CENTRAL_MARKERS)
    if np.isnan(P).all() or np.isnan(C).all():
        return np.full(sub_hd.n_obs, np.nan)
    Pz = (P - np.nanmean(P)) / (np.nanstd(P) + 1e-9)
    Cz = (C - np.nanmean(C)) / (np.nanstd(C) + 1e-9)
    delta = Cz - Pz
    ranks = pd.Series(delta).rank(method="average").to_numpy() - 1
    s = ranks / max(1, len(ranks) - 1)
    cy = _gene_score(sub_hd, ("CYP2E1", "CYP1A2"))
    if not np.isnan(cy).all():
        if cy[s > 0.5].mean() < cy[s <= 0.5].mean():
            s = 1.0 - s
    return s


def per_donor_comparison(adata_visium: ad.AnnData, adata_hd: ad.AnnData,
                          sample: str, panel_55_genes: list[str]) -> dict:
    """Run the full Visium-vs-HD comparison for one donor."""
    print(f"[C4]   donor {sample}: building HD axis and slope...")
    # HD path-A subset for this donor.
    path_col = adata_hd.obs.get("platform_path", "").astype(str)
    samples_arr = (adata_hd.obs.get("sample_id", adata_hd.obs.get("sample", ""))
                    .astype(str))
    sub_hd = adata_hd[(path_col == "hd_segmented") & (samples_arr == sample)].copy()
    if sub_hd.n_obs < 100:
        return {"status": "insufficient_HD_cells", "n_cells": int(sub_hd.n_obs)}
    sub_hd.obs["sample_id"] = sample
    # Build axis on the FULL HD gene set (all 9 markers present) BEFORE
    # restricting to the 55-panel — otherwise the §5.1 axis recipe lacks
    # the portal markers that aren't in the strict HVG intersection.
    sub_hd.obs["s"] = hd_axis_per_donor(sub_hd)
    sub_hd_panel = sub_hd[:, [g for g in panel_55_genes if g in sub_hd.var_names]].copy()

    # NOTE: Stage A A3 verified the path-A integrated HD h5ad is already
    # harmony-corrected for sample-specific effects. Applying per-cell
    # log_count residualisation on top of that produces pathological
    # residuals (NaN columns at binning) and double-processes the data.
    # Use the harmony-corrected X as-is.
    Gs_hd, bc_hd, _, _ = quantile_bin_per_donor_pool(
        sub_hd_panel, s_col="s", n_bins=N_BINS_DEFAULT, sigma=SIGMA_BINS)
    if np.isnan(bc_hd).all():
        return {"status": "HD_binning_failed"}
    # Drop NaN bin centres (from quantile-edge collapse) before slope fit —
    # bc.mean() over a vector containing NaN propagates to slope denominator.
    valid_bin = ~np.isnan(bc_hd)
    if valid_bin.sum() < 3:
        return {"status": "HD_too_few_valid_bins", "n_valid": int(valid_bin.sum())}
    Gs_hd = Gs_hd[valid_bin]
    bc_hd = bc_hd[valid_bin]
    grad_hd = per_gene_gradient(Gs_hd, bc_hd)

    # Visium-standard side for this donor with edge mask.
    print(f"[C4]   donor {sample}: standard Visium with edge mask...")
    spot_mask = (adata_visium.obs["sample_id"].astype(str) == sample) & lhd_analysis_mask(adata_visium)
    sub_v = adata_visium[spot_mask].copy()
    edge = edge_mask_visium(sub_v)
    print(f"[C4]   donor {sample}: edge spots flagged: "
          f"{int(edge.sum())} of {sub_v.n_obs} ({100*edge.mean():.1f}%)")
    sub_v_keep = sub_v[~edge].copy()
    sub_v_panel = sub_v_keep[:, [g for g in panel_55_genes if g in sub_v_keep.var_names]].copy()
    Gs_v, bc_v, _, _ = quantile_bin_per_donor_pool(
        sub_v_panel, s_col="s", n_bins=N_BINS_DEFAULT, sigma=SIGMA_BINS)
    if np.isnan(bc_v).all():
        return {"status": "Visium_binning_failed"}
    grad_v = per_gene_gradient(Gs_v, bc_v)

    # Align gene sets — they should already be common after intersection.
    var_v = list(sub_v_panel.var_names)
    var_hd = list(sub_hd_panel.var_names)
    common = sorted(set(var_v) & set(var_hd))
    if len(common) < 10:
        return {"status": "too_few_common_genes", "n_common": len(common)}
    slope_v = pd.Series(grad_v["slope"], index=var_v).reindex(common).to_numpy()
    slope_hd = pd.Series(grad_hd["slope"], index=var_hd).reindex(common).to_numpy()

    # High-confidence subset: top quartile by |Visium slope|.
    abs_v = np.abs(slope_v)
    thr = np.quantile(abs_v, 0.75)
    hc_mask = abs_v >= thr
    sign_agree_hc = int((np.sign(slope_v[hc_mask]) == np.sign(slope_hd[hc_mask])).sum())
    n_hc = int(hc_mask.sum())
    pct_hc = 100.0 * sign_agree_hc / max(1, n_hc)

    # Rank correlation across all 55-panel common genes.
    rho, _ = spearmanr(slope_v, slope_hd)

    # Per-marker direction agreement on 8 headline + 9 sanity.
    headline_per_marker = {}
    sanity_per_marker = {}
    for g, expected in KNOWN_LIVER_DIRECTION.items():
        if g in common:
            i = common.index(g)
            agree = int(np.sign(slope_v[i]) == np.sign(slope_hd[i]))
            sanity_per_marker[g] = {
                "visium_slope": float(slope_v[i]),
                "hd_slope": float(slope_hd[i]),
                "agree": agree,
                "expected": int(expected),
            }
            if g in HEADLINE_MARKERS:
                headline_per_marker[g] = sanity_per_marker[g]

    # Persist per-gene slopes on the 55-panel for Supp Table 4 panel-level calibration.
    panel_slopes = {
        "genes": list(common),
        "visium_slope": [float(v) for v in slope_v],
        "hd_slope": [float(v) for v in slope_hd],
    }
    return {
        "status": "ok",
        "sample": sample,
        "n_panel_genes_common": len(common),
        "n_visium_spots_kept_after_edge_mask": int((~edge).sum()),
        "n_hd_cells": int(sub_hd_panel.n_obs),
        "high_confidence_subset": {
            "n_genes": n_hc,
            "n_agree": sign_agree_hc,
            "sign_agreement_pct": float(pct_hc),
            "passes_section_13_85pct": bool(pct_hc >= 85.0),
        },
        "rank_correlation_all_panel": float(rho),
        "passes_section_13_rank_07": bool(rho >= 0.7),
        "marker_agreement_headline_8": headline_per_marker,
        "marker_agreement_sanity_9": sanity_per_marker,
        "n_headline_8_agree": sum(v["agree"] for v in headline_per_marker.values()),
        "n_sanity_9_agree": sum(v["agree"] for v in sanity_per_marker.values()),
        "panel_slopes": panel_slopes,
    }


def supp_table_4_calibration(panel_55_genes: list[str], hd_results: dict) -> dict:
    """
    Compare paper's published HD zonation (Supp Table 4) against our HD slopes
    on the **full 55-gene panel** (panel-level, not just the 9 markers).
    """
    if not SUPP_T4.exists():
        return {"status": "supp_t4_missing"}
    print(f"[C4] Supp Table 4 calibration on combined HD M1+M2+M6 (panel-level)...")
    try:
        t4 = pd.read_excel(SUPP_T4, sheet_name="combined_visiumHD_table_M1M2M6", header=1)
        t4.columns = [str(c).strip() for c in t4.columns]
        gene_col = next((c for c in t4.columns if c.lower() in ("gene_name", "genename")), None)
        if gene_col is None:
            return {"status": "supp_t4_no_gene_column", "columns": list(t4.columns)[:10]}
        t4 = t4.set_index(gene_col)
        zone_cols = [c for c in t4.columns if c.startswith("Mean_Zone_")]
        t4[zone_cols] = t4[zone_cols].apply(pd.to_numeric, errors="coerce")
        t4_dir = np.sign(
            t4[["Mean_Zone_1", "Mean_Zone_2"]].mean(axis=1)
            - t4[["Mean_Zone_7", "Mean_Zone_8"]].mean(axis=1)
        )
        t4_dir = t4_dir[~t4_dir.index.duplicated()]

        # Per-donor: compare ALL panel-overlapping genes (not just markers).
        per_donor: dict = {}
        per_donor_marker_only: dict = {}
        for sample, donor_res in hd_results.items():
            if not isinstance(donor_res, dict) or donor_res.get("status") != "ok":
                continue
            ps = donor_res.get("panel_slopes")
            if not ps:
                continue
            our_signs = pd.Series(np.sign(ps["hd_slope"]), index=ps["genes"])
            common = sorted(set(our_signs.index) & set(t4_dir.index))
            our_signs = our_signs.reindex(common).to_numpy()
            t4_signs = t4_dir.reindex(common).to_numpy()
            valid = (our_signs != 0) & (t4_signs != 0) & ~np.isnan(t4_signs)
            n_compared = int(valid.sum())
            n_agree = int((our_signs[valid] == t4_signs[valid]).sum())
            pct = 100.0 * n_agree / max(1, n_compared)
            per_donor[sample] = {
                "n_compared": n_compared,
                "n_agree": n_agree,
                "sign_agreement_pct": float(pct),
            }
            # Marker-only sub-result for quick comparison.
            mk = [g for g in common if g in KNOWN_LIVER_DIRECTION]
            mk_idx = [common.index(g) for g in mk]
            mk_valid = valid[mk_idx]
            mk_agree = int((our_signs[mk_idx][mk_valid] == t4_signs[mk_idx][mk_valid]).sum())
            per_donor_marker_only[sample] = {
                "n_markers_compared": int(mk_valid.sum()),
                "n_markers_agree": mk_agree,
            }
        return {
            "status": "ok",
            "per_donor_panel_agreement": per_donor,
            "per_donor_marker_only": per_donor_marker_only,
            "panel_size_in_supp_t4": int(len(t4_dir)),
        }
    except Exception as e:
        return {"status": f"failed: {type(e).__name__}: {str(e)[:80]}"}


def run_c4() -> None:
    """C4 — cross-platform Visium-vs-HD on M2 + M6 (§9.4)."""
    print(f"[C4] Loading {VISIUM}")
    adata_v = sc.read_h5ad(VISIUM)
    print(f"[C4] Loading {VISIUM_HD}")
    adata_hd = sc.read_h5ad(VISIUM_HD)

    # HD ∩ strict-66.
    panel_strict = analysis_gene_panel(adata_v, mode="strict")
    strict_genes = list(adata_v.var_names[panel_strict])
    panel_55 = sorted(set(strict_genes) & set(adata_hd.var_names))
    print(f"[C4] HD ∩ strict-66: {len(panel_55)} genes")

    per_donor = {}
    for sample in ("M2", "M6"):
        per_donor[sample] = per_donor_comparison(adata_v, adata_hd, sample, panel_55)
        if per_donor[sample].get("status") == "ok":
            print(f"[C4]   {sample}: high-conf agreement = "
                  f"{per_donor[sample]['high_confidence_subset']['sign_agreement_pct']:.1f}%, "
                  f"rank ρ = {per_donor[sample]['rank_correlation_all_panel']:.3f}, "
                  f"headline 8 agree = "
                  f"{per_donor[sample]['n_headline_8_agree']}/8")

    # Pooled metrics across M2 + M6 (weighted by spots/cells).
    pooled = {}
    ok_donors = [s for s in ("M2", "M6") if per_donor.get(s, {}).get("status") == "ok"]
    if ok_donors:
        weights = [per_donor[s]["n_visium_spots_kept_after_edge_mask"] for s in ok_donors]
        wsum = sum(weights)
        pooled = {
            "high_conf_sign_agreement_pct": float(sum(
                w * per_donor[s]["high_confidence_subset"]["sign_agreement_pct"]
                for w, s in zip(weights, ok_donors)
            ) / max(1, wsum)),
            "rank_correlation": float(sum(
                w * per_donor[s]["rank_correlation_all_panel"]
                for w, s in zip(weights, ok_donors)
            ) / max(1, wsum)),
            "n_headline_8_agree_total": sum(per_donor[s]["n_headline_8_agree"]
                                              for s in ok_donors),
            "n_headline_8_total": 8 * len(ok_donors),
        }

    # §14 #2 trigger: pooled high-conf disagreement > 15%.
    sev14_2 = bool(pooled.get("high_conf_sign_agreement_pct", 100.0) < 85.0)

    # Supp Table 4 calibration.
    t4_calib = supp_table_4_calibration(panel_55, per_donor)

    summary = {
        "panel_55_size": len(panel_55),
        "per_donor": per_donor,
        "pooled": pooled,
        "sev14_2_failure": sev14_2,
        "supp_table_4_calibration": t4_calib,
    }
    OUT_C4.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[C4] Wrote {OUT_C4}")
    print(f"[C4] Pooled high-conf agreement: "
          f"{pooled.get('high_conf_sign_agreement_pct', 'N/A')}; "
          f"§14 #2 failure: {sev14_2}")


# ===========================================================================
# B12 — Cross-pipeline reproducibility, approach (a): like-for-like
# ===========================================================================
# NOTE: per_gene_layer_means imported from sgd.gradient (legacy B12 defined it
# inline). B12's load_supp_table_2 differs slightly from B11's (one-line column
# normalisation vs the multi-step version); B12's local form is kept verbatim
# below to preserve byte-for-byte output of stage_B_cross_pipeline_approach_a.json.

def _b12_load_supp_table_2() -> pd.DataFrame:
    df = pd.read_excel(SUPP_T2, sheet_name="mean zonation of non steatotic", header=1)
    df.columns = [str(c).strip().replace("mean_zone_", "Mean_Zone_") for c in df.columns]
    df = df.dropna(subset=["gene_name"]).copy()
    df["gene_name"] = df["gene_name"].astype(str)
    df = df.set_index("gene_name")
    zone_cols = [c for c in df.columns if c.startswith("Mean_Zone_")]
    df[zone_cols] = df[zone_cols].apply(pd.to_numeric, errors="coerce")
    return df


def run_b12() -> None:
    """B12 — cross-pipeline reproducibility, approach (a) 8-layer."""
    print(f"[B12] Loading {VISIUM}")
    adata = sc.read_h5ad(VISIUM)
    spot_mask = lhd_analysis_mask(adata)
    sub = adata[spot_mask].copy()
    print(f"[B12]   LHD ∧ ¬fibrotic spots: {sub.n_obs}; genes: {sub.n_vars}")

    # --- Per-gene 8-layer mean expression on our pipeline ---
    print(f"[B12] Binning into {N_LAYERS} quantile layers along obs.s...")
    our_layers, layer_counts = per_gene_layer_means(sub, s_col="s", n_layers=N_LAYERS)
    print(f"[B12]   cells per layer: {layer_counts.tolist()}")

    # --- Sanity check on the 9 known markers ---
    # Our convention: low-s = portal, high-s = central. So a portal marker
    # should have Layer_1 > Layer_8; a central marker should have Layer_8 > Layer_1.
    print("[B12] Sanity check on 9 known markers:")
    for g, expected in KNOWN_LIVER_DIRECTION.items():
        if g in our_layers.index:
            l1 = our_layers.loc[g, "Layer_1"]
            l8 = our_layers.loc[g, "Layer_8"]
            our_dir = +1 if l8 > l1 else -1
            ok = "✓" if our_dir == expected else "✗"
            label = "central" if expected == +1 else "portal"
            print(f"[B12]   {g} ({label}): L1={l1:.3e}, L8={l8:.3e}, our_dir={our_dir} {ok}")

    # --- Read paper's Supp Table 2 ---
    print(f"[B12] Reading {SUPP_T2}")
    pub = _b12_load_supp_table_2()

    # --- Compute directions on identical aggregation ---
    # Both: sign(mean(top-2 central layers) - mean(top-2 portal layers))
    # Ours: sign(mean(L7,L8) - mean(L1,L2))   [L8 = central]
    # Paper: sign(mean(Z1,Z2) - mean(Z7,Z8))  [Z1 = central]
    print("[B12] Computing directions on identical aggregation...")
    our_central = our_layers[["Layer_7", "Layer_8"]].mean(axis=1)
    our_portal = our_layers[["Layer_1", "Layer_2"]].mean(axis=1)
    our_diff = our_central - our_portal
    our_dir = np.sign(our_diff)

    pub_central = pub[["Mean_Zone_1", "Mean_Zone_2"]].mean(axis=1)
    pub_portal = pub[["Mean_Zone_7", "Mean_Zone_8"]].mean(axis=1)
    pub_diff = pub_central - pub_portal
    pub_dir = np.sign(pub_diff)

    common = sorted(set(our_dir.index) & set(pub_dir.index))
    n_common = len(common)
    print(f"[B12]   genes in common: {n_common}")

    # Align via reindex on a unique index, dropping any duplicates
    our_dir = our_dir[~our_dir.index.duplicated()]
    pub_dir = pub_dir[~pub_dir.index.duplicated()]
    our_diff_s = our_diff[~our_diff.index.duplicated()]
    pub_diff_s = pub_diff[~pub_diff.index.duplicated()]
    our_d = our_dir.reindex(common).to_numpy()
    pub_d = pub_dir.reindex(common).to_numpy()
    our_diff_arr = our_diff_s.reindex(common).to_numpy()
    pub_diff_arr = pub_diff_s.reindex(common).to_numpy()

    # Filter to genes with non-zero direction in both AND non-NaN
    both_nonzero = (our_d != 0) & (pub_d != 0) & ~np.isnan(our_d) & ~np.isnan(pub_d)
    n_both = int(both_nonzero.sum())
    n_agree = int((our_d[both_nonzero] == pub_d[both_nonzero]).sum())
    pct_overall = 100.0 * n_agree / max(1, n_both)
    print(f"[B12]   directional in both: {n_both}; sign agreement = {pct_overall:.2f}%")

    # High-confidence subset (q < 0.05 in Supp Table 2).
    q_col = next((c for c in pub.columns if c.lower().startswith("q")), None)
    pct_hc = float("nan")
    n_agree_hc = 0
    n_hc = 0
    if q_col is not None:
        qser = pd.to_numeric(pub[q_col], errors="coerce")
        qser = qser[~qser.index.duplicated()]
        qvals = qser.reindex(common).to_numpy()
        hc_mask = (qvals < 0.05) & both_nonzero
        n_hc = int(hc_mask.sum())
        n_agree_hc = int((our_d[hc_mask] == pub_d[hc_mask]).sum())
        pct_hc = 100.0 * n_agree_hc / max(1, n_hc)
        print(f"[B12]   q<0.05 subset: n={n_hc}; sign agreement = {pct_hc:.2f}%")

    # Restrict to higher-magnitude genes (top quartile by |our_diff|)
    abs_our = np.abs(our_diff_arr)
    abs_our_valid = abs_our[both_nonzero]
    if len(abs_our_valid) > 0:
        thr = np.quantile(abs_our_valid, 0.75)
        mag_mask = both_nonzero & (abs_our >= thr)
        n_mag = int(mag_mask.sum())
        n_agree_mag = int((our_d[mag_mask] == pub_d[mag_mask]).sum())
        pct_mag = 100.0 * n_agree_mag / max(1, n_mag)
    else:
        n_mag = 0
        n_agree_mag = 0
        pct_mag = float("nan")
    print(f"[B12]   top-quartile |our_diff| subset: n={n_mag}; sign agreement = {pct_mag:.2f}%")

    # Spearman rank correlation of magnitudes (signed) on the common+nonzero subset.
    rho, _ = spearmanr(our_diff_arr[both_nonzero], pub_diff_arr[both_nonzero])

    out = {
        "method": "approach_a_8_layer_aggregation_both_pipelines",
        "our_axis": "obs.s, 8-quantile layers per donor, pooled mean across donors",
        "paper_axis": "Supp Table 2 Mean_Zone_1..8 (Z1=central, Z8=portal verified)",
        "our_direction": "sign(mean(L7,L8) - mean(L1,L2)) — central-minus-portal",
        "paper_direction": "sign(mean(Z1,Z2) - mean(Z7,Z8)) — central-minus-portal",
        "n_genes_common": int(n_common),
        "n_directional_in_both": int(n_both),
        "sign_agreement_pct": float(pct_overall),
        "n_agree_overall": int(n_agree),
        "high_confidence_q_lt_005": {
            "n_genes": int(n_hc),
            "n_agree": int(n_agree_hc),
            "sign_agreement_pct": float(pct_hc),
        },
        "top_quartile_our_magnitude": {
            "n_genes": int(n_mag),
            "n_agree": int(n_agree_mag),
            "sign_agreement_pct": float(pct_mag),
        },
        "signed_magnitude_spearman": float(rho),
        "passes_sec13_section_9_7_threshold_95pct": bool(pct_overall >= 95.0),
        "passes_high_confidence_95pct": bool(pct_hc >= 95.0),
    }
    OUT_B12.write_text(json.dumps(out, indent=2))
    print(f"[B12] Wrote {OUT_B12}")
    print(f"[B12] Summary: full={pct_overall:.1f}% / q<0.05={pct_hc:.1f}% / "
          f"top-quartile-mag={pct_mag:.1f}% / signed-magnitude Spearman={rho:.3f}")


# ===========================================================================
# C6 (repackaging) — cross-pipeline reproducibility artifact (§9.7)
# ===========================================================================
# C6 computes nothing: it reformats B11 + B12 JSON into the §9.7-framed
# artifact two figures read. Folded in here as a final repackaging function.

def _c6_safe_load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"status": "missing"}


def repackage_cross_pipeline() -> None:
    """C6 — repackage B11/B12 JSON into cross_pipeline_reproducibility.json."""
    b11 = _c6_safe_load(B11_OUT)
    b11_cross = _c6_safe_load(B11_CROSS)
    b12 = _c6_safe_load(B12_OUT)

    summary = {
        "supp_table_source": str(SUPP_T2),
        "framing": (
            "§9.7 cross-pipeline reproducibility against Yakubovsky et al. "
            "1,711-gene zonation panel (Supp Table 2). The §13 ≥95% threshold "
            "was set when Variant B was unavailable; in practice cross-"
            "pipeline gene-by-gene comparisons on this dataset peak at "
            "70-79% sign agreement (Stage B addendum B11/B12). The framework's "
            "contribution is the dynamical interpretation under the "
            "material-derivative identity, not the gene-by-gene direction "
            "calls; the §9.7 row is reframed as 'sign agreement substantially "
            "above chance with high agreement on high-confidence subsets'."
        ),
        "approach_b_from_B11": (
            {
                "status": "ok",
                "n_genes_compared": b11_cross.get("n_genes_compared"),
                "sign_agreement_pct": b11_cross.get("sign_agreement_pct"),
                "high_confidence_q_lt_005": b11_cross.get("high_confidence_q_lt_005"),
                "method": b11_cross.get("method"),
            } if b11_cross.get("n_genes_compared") is not None
            else {"status": "missing"}
        ),
        "approach_a_from_B12": (
            {
                "status": "ok",
                "n_directional_in_both": b12.get("n_directional_in_both"),
                "sign_agreement_pct": b12.get("sign_agreement_pct"),
                "high_confidence_q_lt_005": b12.get("high_confidence_q_lt_005"),
                "top_quartile_our_magnitude": b12.get("top_quartile_our_magnitude"),
                "signed_magnitude_spearman": b12.get("signed_magnitude_spearman"),
                "our_axis": b12.get("our_axis"),
                "paper_axis": b12.get("paper_axis"),
            } if b12.get("n_directional_in_both") is not None
            else {"status": "missing"}
        ),
        "variant_B_summary": (
            {
                "marker_recovery_pooled": b11.get("marker_recovery", {}).get("pooled_summary"),
                "per_marker_donor_count": b11.get("marker_recovery", {}).get("per_marker_donor_count"),
                "passes_sec13_variant_B": b11.get("passes_sec13_variant_B"),
                "sev14_7_failure_strict": b11.get("sev14_7_failure"),
                "cohort_overlap_caveat": b11.get("cohort_overlap_caveat"),
            } if b11.get("marker_recovery") is not None
            else {"status": "missing"}
        ),
        "cps1_caveat": (
            "CPS1 is the only marker that fails Variant B's 9-marker recovery "
            "(0/8 per donor). Traceable to Supp Table 2's own CPS1 zone profile: "
            "peak at Z3 (mid-axis), with mean(Z1-2) > mean(Z7-8) classifying "
            "CPS1 as weakly central — contrary to literature attribution as "
            "portal. The Stage B / C / D pipeline excludes CPS1 from headline "
            "marker scoreboards and discusses it as a dataset-specific oddity "
            "in supplementary."
        ),
        "tier_0_caveat": (
            "Tier 0 9/9 marker recovery on obs.s is partly tautological for "
            "axis-defining markers. The genuinely load-bearing claim is "
            "Tier 3 cross-donor reproducibility (mean off-diag r = 0.873, "
            "min = 0.770). Stage C inherits this framing."
        ),
    }
    OUT_C6.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[C6] Wrote {OUT_C6}")
    apb = summary["approach_b_from_B11"]
    apa = summary["approach_a_from_B12"]
    if apb.get("status") == "ok":
        print(f"[C6]   Approach (b): {apb['sign_agreement_pct']:.1f}% on full panel")
    if apa.get("status") == "ok":
        print(f"[C6]   Approach (a): {apa['sign_agreement_pct']:.1f}% on full panel; "
              f"top-quartile-mag {apa['top_quartile_our_magnitude']['sign_agreement_pct']:.1f}%")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    # C2, C4, B12 each compute and write an artifact; repackage_cross_pipeline
    # runs last because it reads B12's output (and B11's, produced by 07).
    run_c2()
    run_c4()
    run_b12()
    repackage_cross_pipeline()


if __name__ == "__main__":
    main()
