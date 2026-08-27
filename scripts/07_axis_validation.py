"""
07 — Spatial-axis validation (four-tier validation, Methods §4.4).

Consolidates the five legacy Stage-B axis-validation scripts into one
entry point. Each legacy script's analysis body is copied verbatim; the
only changes are import rewiring to ``src/sgd/`` and hard-coded paths
replaced by ``sgd.config`` constants.

Absorbs (run in this order):
  * B3_tier0_tier1   — Tier 0 marker scoreboard + Tier 1 leave-family-out
  * B7_c1_and_edge   — C1 covariate adjustment + edge concentration +
                       headline-candidate freeze (writes headline_candidates.json)
  * B8_sensitivity   — analyst-DOF sensitivity sweep (reads headline_candidates.json)
  * B9_cross_donor   — Tier 3 cross-donor reproducibility (reads headline_candidates.json)
  * B11_variant_B_addendum — Variant-B axis + Tier 2 + cross-pipeline (b)

Run order: B7 must run before B8/B9 (it writes ``headline_candidates.json``
which both read). B11 is independent and runs last; it is the SOLE writer
of ``obs.s_v_B`` into ``visium_human.h5ad``.

Reads:
  RESULTS/visium_human.h5ad
  RESULTS/per_gene_gradient.parquet
  DATA   Supplementary Table 2 xlsx

Writes (all to RESULTS, basenames identical to legacy):
  tier0_scoreboard.json, tier1_leave_family.json
  headline_candidates.json, b7_joint_table.parquet, b7_c1_summary.json
  b8_sensitivity.json
  cross_donor_correlation.json
  stage_B_variant_B_addendum.json, stage_B_cross_pipeline_supp_table_2.json
  stage_B_cross_pipeline_supp_table_2_genes.parquet
  RESULTS/visium_human.h5ad updated in place with obs.s_v_B

Feeds: four-tier validation Methods §4.4 — Fig 1D/1E/1F, Fig S2/S8/S9;
       cross-pipeline approach (b) (R2.4).
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

from sgd.axis import (build_axis_from_panel, build_score_axis,
                      derive_directions, load_supp_table_2)
from sgd.config import RESULTS, SUPP_T2
from sgd.confounds import (donor_aware_residualize, per_gene_edge_concentration,
                           select_donor_aware_model)
from sgd.gradient import (N_BINS_DEFAULT, N_BINS_SWEEP, SIGMA_BINS,
                          per_gene_gradient, quantile_bin_per_donor_pool)
from sgd.panels import (CENTRAL_MARKERS, HEADLINE_MARKERS, KNOWN_LIVER_DIRECTION,
                        PORTAL_MARKERS, _gene_score, analysis_gene_panel,
                        cohort_mask, lhd_analysis_mask)
from sgd.config import LHD_SAMPLES

warnings.filterwarnings("ignore")
np.random.seed(0)

# --- Paths (legacy hard-coded DATA/* -> sgd.config constants) ---
VISIUM = RESULTS / "visium_human.h5ad"
GRAD = RESULTS / "per_gene_gradient.parquet"

OUT_T0 = RESULTS / "tier0_scoreboard.json"
OUT_T1 = RESULTS / "tier1_leave_family.json"
OUT_HEADLINE = RESULTS / "headline_candidates.json"
OUT_JOINT = RESULTS / "b7_joint_table.parquet"
OUT_SUMMARY = RESULTS / "b7_c1_summary.json"
OUT_B8 = RESULTS / "b8_sensitivity.json"
OUT_B9 = RESULTS / "cross_donor_correlation.json"
OUT_ADDENDUM = RESULTS / "stage_B_variant_B_addendum.json"
OUT_CROSS_PIPELINE = RESULTS / "stage_B_cross_pipeline_supp_table_2.json"
OUT_CROSS_PIPELINE_GENES = RESULTS / "stage_B_cross_pipeline_supp_table_2_genes.parquet"

# --- B3/B4 constants (verbatim from B3_tier0_tier1.py) ---
CYP_FAMILY = ("CYP2A6", "CYP2E1", "CYP1A2", "CYP3A4")
UREA_FAMILY = ("CPS1", "ASS1")
GLUL_ONLY = ("GLUL",)

LEAVE_OUT_VARIANTS = {
    "drop_CYP_family": CYP_FAMILY,
    "drop_urea_cycle": UREA_FAMILY,
    "drop_GLUL_alone": GLUL_ONLY,
}

# --- B7 constants (verbatim from B7_c1_and_edge.py) ---
N_TOP_HEADLINE = 20
N_BIN_BOOTSTRAP = 200
EDGE_FRAC = 0.05
EDGE_SCORE_THRESHOLD = 1.0

# --- B9 cohort partitions (verbatim from legacy stage_B/utils.py;
#     not exposed by sgd.config so defined here as the legacy values) ---
LHD_NON_STEATOTIC = ("M4", "M5", "M6", "M7", "M8")
C5_FLAGGED_LHD = ("M2", "M4", "M6", "M8")    # from Stage A confound diagnostics
C5_CLEAN_LHD = ("M1", "M3", "M5", "M7")

# --- B11 constants (verbatim from B11_variant_B_addendum.py) ---
SUPP_DIR = SUPP_T2.parent  # legacy: DATA/41586_..._ESM/2025-01-01424E-s1


# ===========================================================================
# B3 + B4 — Tier 0 marker scoreboard and Tier 1 leave-family-out
# ===========================================================================

def slope_signs_pooled(adata_lhd: ad.AnnData, s_col: str,
                        gene_subset: list[str] | None = None) -> dict[str, int]:
    """
    Pooled-cohort marker recovery using the **same gradient statistic
    as B6** (slope β from per_gene_gradient, on the strict analysis
    panel binned at default n_bins). For each KNOWN_LIVER marker:
    1 if sign(slope) matches the literature-expected sign, else 0.
    """
    Gs, bc, _, _ = quantile_bin_per_donor_pool(
        adata_lhd, s_col=s_col, n_bins=N_BINS_DEFAULT, sigma=SIGMA_BINS)
    if np.isnan(bc).all() or len(bc) < 3:
        return {}
    res = per_gene_gradient(Gs, bc)
    var_names = list(adata_lhd.var_names)
    out: dict[str, int] = {}
    targets = gene_subset if gene_subset is not None else list(KNOWN_LIVER_DIRECTION)
    for g in targets:
        expected = KNOWN_LIVER_DIRECTION.get(g, 0)
        if g not in var_names or expected == 0:
            continue
        slope_g = res["slope"][var_names.index(g)]
        out[g] = int(np.sign(slope_g) == expected)
    return out


def slope_signs_per_donor(adata_lhd: ad.AnnData, s_col: str,
                           gene_subset: list[str] | None = None
                           ) -> dict[str, dict[str, int]]:
    """Per-donor variant of slope_signs_pooled — one slope per donor per marker."""
    out: dict[str, dict[str, int]] = {}
    samples_arr = adata_lhd.obs["sample_id"].astype(str).to_numpy()
    for sample in sorted(np.unique(samples_arr)):
        m = samples_arr == sample
        if m.sum() < 30:
            continue
        sub_d = adata_lhd[m].copy()
        out[str(sample)] = slope_signs_pooled(sub_d, s_col, gene_subset)
    return out


def run_b3_b4() -> None:
    """B3 + B4 — Tier 0 marker scoreboard and Tier 1 leave-family-out."""
    print(f"[B3/B4] Loading {VISIUM}")
    adata = sc.read_h5ad(VISIUM)
    spot_mask = lhd_analysis_mask(adata)
    # Tier 0 / Tier 1 do NOT restrict to the strict HVG panel — the 9 known
    # markers are literature-defined ground truth and may not all be HVGs in
    # an 8-way intersection of 2,000-gene per-sample HVG sets. Strict-panel
    # filtering is correct for B6/B7/B9 but wrong here. We use all genes that
    # are detected at all in the LHD cohort (the post-A2 filtered gene set).
    sub = adata[spot_mask].copy()
    if "s" not in sub.obs.columns:
        raise RuntimeError("[B3/B4] obs.s missing — run B2 first")
    panel_strict = analysis_gene_panel(adata, mode="strict")
    n_markers_in_strict = sum(1 for g in KNOWN_LIVER_DIRECTION
                                if g in adata.var_names[panel_strict])
    print(f"[B3/B4]   LHD ∧ ¬fibrotic spots: {sub.n_obs}; "
          f"genes available: {sub.n_vars} (full LHD-detected set)")
    print(f"[B3/B4]   For reference: {n_markers_in_strict}/9 markers are also in the strict HVG panel.")

    # ============== Tier 0 ==============
    print("[B3] Tier 0 marker scoreboard on obs.s (slope-based, B6 statistic)")
    pooled = slope_signs_pooled(sub, "s")
    per_donor = slope_signs_per_donor(sub, "s")

    pooled_correct = sum(pooled.values())
    pooled_total = len(pooled)
    per_marker_donor_count: dict[str, int] = {}
    for g in KNOWN_LIVER_DIRECTION:
        if g not in pooled:
            continue
        per_marker_donor_count[g] = sum(per_donor[d].get(g, 0) for d in per_donor)

    n_donors_lhd = len(per_donor)
    gate_pooled = pooled_correct == pooled_total                   # §14 #1: <8/9 fails
    gate_per_marker = all(c >= max(7, n_donors_lhd - 1)
                           for c in per_marker_donor_count.values())  # ≥7/8
    sev14_1 = (pooled_correct < 8) or any(c < 6 for c in per_marker_donor_count.values())

    t0 = {
        "axis_used": "obs.s",
        "n_donors_lhd": n_donors_lhd,
        "pooled_marker_correct": pooled,
        "pooled_summary": f"{pooled_correct}/{pooled_total}",
        "per_donor_marker_correct": per_donor,
        "per_marker_donor_count": per_marker_donor_count,
        "gate_pooled_9_of_9": bool(gate_pooled),
        "gate_per_marker_min_7_of_8": bool(gate_per_marker),
        "sev14_1_failure": bool(sev14_1),
    }
    OUT_T0.write_text(json.dumps(t0, indent=2, default=str))
    print(f"[B3]   pooled: {pooled_correct}/{pooled_total}; "
          f"per-marker min count: {min(per_marker_donor_count.values()) if per_marker_donor_count else 'NA'}")
    print(f"[B3]   §14 #1 failure: {sev14_1}")
    print(f"[B3] Wrote {OUT_T0}")

    # ============== Tier 1 ==============
    print("[B4] Tier 1 leave-family-out variants")
    t1: dict[str, dict] = {}
    for variant, dropped in LEAVE_OUT_VARIANTS.items():
        portal_kept = tuple(g for g in PORTAL_MARKERS if g not in dropped)
        central_kept = tuple(g for g in CENTRAL_MARKERS if g not in dropped)
        if not portal_kept or not central_kept:
            t1[variant] = {"status": "skipped",
                            "reason": "all genes from one direction dropped"}
            continue

        s_v, _, diag = build_score_axis(sub, portal_kept, central_kept)
        sub_v = sub.copy()
        sub_v.obs["_s_v"] = s_v
        # Held-out family marker recovery via the same slope-based statistic.
        held_out = [g for g in dropped if g in sub.var_names]
        held_out_signs = slope_signs_pooled(sub_v, "_s_v", gene_subset=held_out)
        per_donor_held = slope_signs_per_donor(sub_v, "_s_v", gene_subset=held_out)

        t1[variant] = {
            "dropped_genes": list(dropped),
            "axis_diagnostics_per_sample": diag,
            "held_out_marker_pooled_correct": held_out_signs,
            "per_donor_held_out_correct": per_donor_held,
            "n_correct_pooled": int(sum(held_out_signs.values())),
            "n_evaluated_pooled": int(len(held_out_signs)),
            "statistic": "slope_sign_vs_expected",
        }
        print(f"[B4]   {variant}: pooled held-out correct "
              f"{t1[variant]['n_correct_pooled']}/{t1[variant]['n_evaluated_pooled']}")

    OUT_T1.write_text(json.dumps(t1, indent=2, default=str))
    print(f"[B4] Wrote {OUT_T1}")


# ===========================================================================
# B7 — C1 covariate adjustment + per-gene edge-concentration + joint flag table
# ===========================================================================

def freeze_headline_set(grad_df: pd.DataFrame, var_names: pd.Index) -> tuple[list[str], list[str]]:
    """
    9 known markers + top-N highest-magnitude expressed by |slope| on the
    unadjusted s axis at default n_bins, restricted to the **strict
    analysis panel** (the rows B6 wrote with panel='strict').
    """
    cohort = grad_df[(grad_df["scope"] == "cohort") & (grad_df["axis"] == "s")
                      & (grad_df["n_bins"] == N_BINS_DEFAULT)
                      & (grad_df.get("panel", "strict") == "strict")]
    cohort = cohort.set_index("gene")
    markers = [g for g in KNOWN_LIVER_DIRECTION if g in cohort.index]
    candidates = cohort["slope"].abs().sort_values(ascending=False)
    candidates = [g for g in candidates.index if g not in markers]
    top_n = candidates[:N_TOP_HEADLINE]
    return markers, list(top_n)


def per_gene_noise_envelope(
    adata_lhd: ad.AnnData, gene_idx: np.ndarray, n_boot: int = N_BIN_BOOTSTRAP,
) -> np.ndarray:
    """
    95th-percentile bin-bootstrap |Δ slope| per gene. Resamples bins
    with replacement at default n_bins; computes slope each time;
    reports the 95th percentile of |Δ| (vs the full-bin baseline) per
    gene — the per-gene noise envelope used in step 3.
    """
    Gs, bc, _, _ = quantile_bin_per_donor_pool(
        adata_lhd, s_col="s", n_bins=N_BINS_DEFAULT, sigma=SIGMA_BINS)
    full_slope = per_gene_gradient(Gs, bc)["slope"][gene_idx]
    rng = np.random.RandomState(0)
    n_bins = len(bc)
    deltas = np.zeros((n_boot, len(gene_idx)))
    for b in range(n_boot):
        idx = rng.choice(n_bins, size=n_bins, replace=True)
        Gs_b = Gs[idx]
        bc_b = bc[idx]
        # Need monotone bin centres for slope; sort.
        order = np.argsort(bc_b)
        bc_s = bc_b[order]
        Gs_s = Gs_b[order]
        slope_b = per_gene_gradient(Gs_s, bc_s)["slope"][gene_idx]
        deltas[b] = np.abs(slope_b - full_slope)
    return np.percentile(deltas, 95, axis=0).astype(np.float32)


def run_b7() -> None:
    """B7 — C1 covariate adjustment + edge concentration + headline freeze."""
    print(f"[B7] Loading {VISIUM}")
    adata = sc.read_h5ad(VISIUM)
    panel_strict = analysis_gene_panel(adata, mode="strict")
    spot_mask = lhd_analysis_mask(adata)
    sub = adata[spot_mask, panel_strict].copy()
    print(f"[B7]   LHD ∧ ¬fibrotic spots: {sub.n_obs}; strict panel: {sub.n_vars} genes")

    grad_df = pd.read_parquet(GRAD)
    markers, top_n = freeze_headline_set(grad_df, sub.var_names)
    # Keep only headline genes that survived the strict panel.
    headline = [g for g in (list(markers) + list(top_n)) if g in sub.var_names]
    headline_idx = np.array([list(sub.var_names).index(g) for g in headline])
    print(f"[B7] Frozen headline: {len(markers)} markers + {len(top_n)} top — "
          f"{len(headline)} total after strict-panel filter")
    OUT_HEADLINE.write_text(json.dumps({
        "markers": markers, "top_magnitude": top_n, "all": headline,
        "n_bins_used_for_freezing": N_BINS_DEFAULT, "axis_used": "s",
    }, indent=2))

    # --- Step 1: donor-aware residualisation ---
    print("[B7] Step 1: donor-aware log(UMI) residualisation")
    log_umi = np.log1p(sub.obs["total_umi_raw"].to_numpy().astype(float))
    donor_ids = sub.obs["donor_id"].astype(str).to_numpy()
    X = sub.X
    chosen_model = select_donor_aware_model(X, donor_ids, log_umi, headline_idx)
    print(f"[B7]   model selected: {chosen_model}")
    resid, info = donor_aware_residualize(X, donor_ids, log_umi, model=chosen_model)
    sub_adj = sub.copy()
    sub_adj.X = resid

    # Recompute gradient on residuals at default n_bins.
    Gs_adj, bc_adj, _, _ = quantile_bin_per_donor_pool(
        sub_adj, s_col="s", n_bins=N_BINS_DEFAULT, sigma=SIGMA_BINS)
    res_adj = per_gene_gradient(Gs_adj, bc_adj)
    Gs_un, bc_un, _, _ = quantile_bin_per_donor_pool(
        sub, s_col="s", n_bins=N_BINS_DEFAULT, sigma=SIGMA_BINS)
    res_un = per_gene_gradient(Gs_un, bc_un)

    slope_un = res_un["slope"]
    slope_adj = res_adj["slope"]
    bin_mean_un = res_un["bin_mean_dgds"]
    bin_mean_adj = res_adj["bin_mean_dgds"]

    # Across-gene Pearson r on candidate-headline slope vector (with vs without).
    from scipy.stats import pearsonr
    r_slope, _ = pearsonr(slope_un[headline_idx], slope_adj[headline_idx])

    # --- Per-gene noise envelope on candidate-headline (200 bin-bootstrap) ---
    print(f"[B7]   per-gene noise envelope ({N_BIN_BOOTSTRAP} bin-bootstraps)")
    noise = per_gene_noise_envelope(sub, headline_idx, n_boot=N_BIN_BOOTSTRAP)

    # Per-gene flags.
    delta_slope = slope_adj[headline_idx] - slope_un[headline_idx]
    umi_sensitive = np.abs(delta_slope) > noise
    sign_change = (np.sign(slope_un[headline_idx]) != np.sign(slope_adj[headline_idx]))

    # --- Step 2: edge-concentration ---
    print("[B7] Step 2: per-gene edge-concentration scores")
    edge_score, edge_meta = per_gene_edge_concentration(
        sub, gene_idx=headline_idx, edge_frac=EDGE_FRAC)
    edge_concentrated = edge_score > EDGE_SCORE_THRESHOLD

    # --- Step 3: 2×2 joint flag table ---
    quadrants = []
    for us, ec in zip(umi_sensitive, edge_concentrated):
        if us and ec:
            quadrants.append("demoted_full_caveat")
        elif us:
            quadrants.append("headline_with_C1_caveat")
        elif ec:
            quadrants.append("headline_with_edge_caveat")
        else:
            quadrants.append("headline_clean")
    quadrants = np.array(quadrants)

    table = pd.DataFrame({
        "gene": headline,
        "is_known_marker": [g in KNOWN_LIVER_DIRECTION for g in headline],
        "slope_unadjusted": slope_un[headline_idx],
        "slope_adjusted": slope_adj[headline_idx],
        "bin_mean_unadjusted": bin_mean_un[headline_idx],
        "bin_mean_adjusted": bin_mean_adj[headline_idx],
        "delta_slope": delta_slope,
        "noise_envelope_p95": noise,
        "umi_sensitive": umi_sensitive,
        "sign_change_under_adjustment": sign_change,
        "edge_concentration_score": edge_score,
        "edge_concentrated": edge_concentrated,
        "quadrant": quadrants,
    })
    table.to_parquet(OUT_JOINT, compression="zstd", index=False)
    print(f"[B7] Wrote {OUT_JOINT}")

    # §14 #8: evaluated on candidate-headline set BEFORE demotion.
    n_sign_change = int(sign_change.sum())
    sev14_8_pre = bool(n_sign_change > 0)
    # After demotion, ask whether the surviving non-demoted candidates
    # all stay sign-stable.
    surviving = quadrants != "demoted_full_caveat"
    n_sign_change_post = int(sign_change[surviving].sum())
    sev14_8_post_demotion_passes = bool(n_sign_change_post == 0)

    summary = {
        "model_chosen": chosen_model,
        "model_A_mean_R2_headline": float(np.mean(info["r2"][headline_idx]))
            if chosen_model == "A" else None,
        "across_gene_r_slope_with_vs_without": float(r_slope),
        "n_headline": int(len(headline)),
        "n_umi_sensitive": int(umi_sensitive.sum()),
        "n_edge_concentrated": int(edge_concentrated.sum()),
        "n_sign_change_pre_demotion": n_sign_change,
        "n_sign_change_post_demotion": n_sign_change_post,
        "quadrant_counts": {
            q: int((quadrants == q).sum())
            for q in ("headline_clean", "headline_with_C1_caveat",
                       "headline_with_edge_caveat", "demoted_full_caveat")
        },
        "sev14_8_pre_demotion_failure": sev14_8_pre,
        "sev14_8_post_demotion_passes": sev14_8_post_demotion_passes,
        "edge_per_sample_meta": edge_meta["per_sample"],
        "noise_envelope_n_bootstrap": N_BIN_BOOTSTRAP,
        "edge_score_threshold": EDGE_SCORE_THRESHOLD,
        "edge_frac": EDGE_FRAC,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[B7] Wrote {OUT_SUMMARY}")
    print(f"[B7] Quadrant counts: {summary['quadrant_counts']}")
    print(f"[B7] §14 #8 pre-demotion: {sev14_8_pre}; post-demotion passes: {sev14_8_post_demotion_passes}")


# ===========================================================================
# B8 — Stage B sensitivity sweeps (frozen headline list)
# ===========================================================================

def signs_for(adata_lhd: ad.AnnData, headline_idx: np.ndarray, axis: str = "s",
              n_bins: int = N_BINS_DEFAULT) -> np.ndarray:
    Gs, bc, _, _ = quantile_bin_per_donor_pool(
        adata_lhd, s_col=axis, n_bins=n_bins, sigma=SIGMA_BINS)
    if np.isnan(bc).all() or len(bc) < 3:
        return np.full(len(headline_idx), np.nan)
    res = per_gene_gradient(Gs, bc)
    return np.sign(res["slope"][headline_idx])


def run_b8() -> None:
    """B8 — Stage B sensitivity sweeps (frozen headline list)."""
    print(f"[B8] Loading {VISIUM}")
    adata = sc.read_h5ad(VISIUM)
    panel_strict = analysis_gene_panel(adata, mode="strict")
    spot_mask = lhd_analysis_mask(adata)
    sub = adata[spot_mask, panel_strict].copy()

    headline = json.loads(OUT_HEADLINE.read_text())["all"]
    # Drop any headline genes that aren't in the strict panel (shouldn't happen
    # because B7 already filtered, but defensive).
    headline = [g for g in headline if g in sub.var_names]
    headline_idx = np.array([list(sub.var_names).index(g) for g in headline])
    n_h = len(headline)
    print(f"[B8] LHD ∧ ¬fibrotic spots: {sub.n_obs}; "
          f"strict panel: {sub.n_vars} genes; headline frozen ({n_h} genes)")

    sign_baseline = signs_for(sub, headline_idx)

    # ---- Dimension 1: normalisation ----
    print("[B8] Dim 1: normalisation (per-donor median log1p vs Pearson residuals)")
    # Default is what's already in adata.X. For Pearson residuals, work
    # from raw counts in layers["counts"] then z-residualise per gene
    # against the analytic Pearson residual model (NB).
    sub_pr = sub.copy()
    sub_pr.X = sub_pr.layers["counts"].copy()
    try:
        sc.experimental.pp.normalize_pearson_residuals(sub_pr)
        sign_pr = signs_for(sub_pr, headline_idx)
        n_disagree_norm = int((sign_baseline != sign_pr).sum())
        norm_status = "ok"
    except Exception as e:
        sign_pr = np.full(n_h, np.nan)
        n_disagree_norm = -1
        norm_status = f"failed: {type(e).__name__}: {str(e)[:80]}"
    print(f"[B8]   Pearson residuals: {norm_status}; disagreements = {n_disagree_norm}")

    # ---- Dimension 2: mt-filter (cheap no-op given Stage A C3) ----
    print("[B8] Dim 2: mt-filter sweep (0/25/45)")
    pct_mt = sub.obs["pct_counts_mt"].to_numpy()
    sign_mt = {}
    n_disagree_mt = {}
    for thr in (0.0, 25.0, 45.0):
        if thr == 0.0:
            keep = np.ones(sub.n_obs, dtype=bool)
        else:
            keep = pct_mt < thr
        sub_thr = sub[keep].copy()
        sign_t = signs_for(sub_thr, headline_idx)
        sign_mt[thr] = sign_t
        n_disagree_mt[thr] = int((sign_baseline != sign_t).sum())
    print(f"[B8]   mt sweep disagreements: {n_disagree_mt}")
    # Stability across mt levels (use levels 0/25/45 jointly):
    mt_unstable = np.zeros(n_h, dtype=bool)
    for i in range(n_h):
        signs = [sign_mt[thr][i] for thr in (0.0, 25.0, 45.0)]
        if not all(s == signs[0] for s in signs):
            mt_unstable[i] = True
    n_disagree_mt_dim = int(mt_unstable.sum())

    # ---- Dimension 3: bin count (read from B6 parquet) ----
    print("[B8] Dim 3: bin-count sweep (read from B6 parquet, strict panel)")
    df = pd.read_parquet(GRAD)
    cohort_s = df[(df["scope"] == "cohort") & (df["axis"] == "s")
                   & (df.get("panel", "strict") == "strict")]
    bc_signs: dict[int, np.ndarray] = {}
    for nb in N_BINS_SWEEP:
        sub_df = cohort_s[cohort_s["n_bins"] == nb].drop_duplicates(subset="gene").set_index("gene")
        if sub_df.empty:
            continue
        # Use .at to force a scalar; missing gene → NaN sign.
        signs = []
        for g in headline:
            if g in sub_df.index:
                signs.append(float(np.sign(sub_df.at[g, "slope"])))
            else:
                signs.append(np.nan)
        bc_signs[nb] = np.array(signs)
    bc_unstable = np.zeros(n_h, dtype=bool)
    for i in range(n_h):
        signs = [bc_signs[nb][i] for nb in bc_signs]
        if signs and not all(s == signs[0] for s in signs):
            bc_unstable[i] = True
    n_disagree_bc_dim = int(bc_unstable.sum())
    print(f"[B8]   bin-count disagreements: {n_disagree_bc_dim}")

    # ---- Joint cross-dimension stability (informational) ----
    norm_unstable = (sign_pr != sign_baseline) if not np.isnan(sign_pr).any() else np.zeros(n_h, dtype=bool)
    joint_unstable = np.array(norm_unstable, dtype=bool) | mt_unstable | bc_unstable
    n_joint = int(joint_unstable.sum())

    out = {
        "n_headline": n_h,
        "thresholds": {"per_dim_max_disagreements": 2,
                       "joint_informational_only": True},
        "dim1_normalisation": {
            "status": norm_status,
            "n_sign_disagreements": n_disagree_norm,
            "passes": (n_disagree_norm >= 0 and n_disagree_norm <= 2),
        },
        "dim2_mt_filter": {
            "n_disagree_per_threshold": {str(k): v for k, v in n_disagree_mt.items()},
            "n_unstable_across_levels": n_disagree_mt_dim,
            "passes": n_disagree_mt_dim <= 2,
        },
        "dim3_bin_count": {
            "levels": list(N_BINS_SWEEP),
            "n_unstable": n_disagree_bc_dim,
            "passes": n_disagree_bc_dim <= 2,
        },
        "joint_cross_dimension_unstable": n_joint,
    }
    OUT_B8.write_text(json.dumps(out, indent=2, default=str))
    print(f"[B8] Wrote {OUT_B8}")
    print(f"[B8] Per-dim pass: norm={out['dim1_normalisation']['passes']} "
          f"mt={out['dim2_mt_filter']['passes']} bc={out['dim3_bin_count']['passes']} "
          f"; joint unstable={n_joint}")


# ===========================================================================
# B9 — Tier 3 cross-donor reproducibility + 3-cohort sub-check
# ===========================================================================

def cohort_slopes(adata: ad.AnnData, samples: list[str],
                  panel_mask: np.ndarray, exclude_fibrotic: bool = True
                  ) -> tuple[np.ndarray, list[str]]:
    """
    Per-gene slope on the strict analysis panel (panel_mask) for cohort `samples`,
    spots restricted to LHD ∩ samples ∩ ¬fibrotic_spot.
    """
    spot = cohort_mask(adata, samples)
    if exclude_fibrotic and "fibrotic_spot" in adata.obs.columns:
        spot = spot & ~adata.obs["fibrotic_spot"].astype(bool).to_numpy()
    sub = adata[spot, panel_mask].copy()
    Gs, bc, _, _ = quantile_bin_per_donor_pool(
        sub, s_col="s", n_bins=N_BINS_DEFAULT, sigma=SIGMA_BINS)
    if np.isnan(bc).all() or len(bc) < 3:
        return np.full(sub.n_vars, np.nan), list(sub.var_names)
    return per_gene_gradient(Gs, bc)["slope"], list(sub.var_names)


def run_b9() -> None:
    """B9 — Tier 3 cross-donor reproducibility + 3-cohort sub-check."""
    from scipy.stats import pearsonr
    print(f"[B9] Loading {VISIUM}")
    adata = sc.read_h5ad(VISIUM)
    panel_strict = analysis_gene_panel(adata, mode="strict")
    panel_var_names = adata.var_names[panel_strict]
    headline = json.loads(OUT_HEADLINE.read_text())["all"]
    headline_in_panel = [g for g in headline if g in set(panel_var_names)]
    headline_idx = np.array([list(panel_var_names).index(g) for g in headline_in_panel])
    print(f"[B9] Strict panel: {len(panel_var_names)} genes; "
          f"headline in panel: {len(headline_in_panel)}/{len(headline)}")

    # --- Per-donor matrix (read from B6 parquet, strict-panel rows) ---
    df = pd.read_parquet(GRAD)
    pd_rows = df[(df["scope"] == "per_donor") & (df["axis"] == "s")
                  & (df["n_bins"] == N_BINS_DEFAULT)
                  & (df.get("panel", "strict") == "strict")]
    if pd_rows.empty:
        raise RuntimeError("[B9] no per-donor strict-panel rows in B6 parquet — re-run B6")
    donors = sorted(pd_rows["donor"].unique().tolist())
    print(f"[B9] Per-donor donors: {donors}")

    def correlation_summary(rows: pd.DataFrame, donor_order: list[str],
                            genes: list[str] | None = None) -> dict:
        """Build a donor-by-donor slope correlation summary from B6 rows."""
        common = sorted(set.intersection(*[
            set(rows[rows["donor"] == d]["gene"].tolist()) for d in donor_order
        ]))
        if genes is not None:
            common = sorted(set(common) & set(genes))
        if not common:
            raise RuntimeError("[B9] no genes common to every requested donor")
        matrix = np.zeros((len(donor_order), len(common)))
        for di, donor in enumerate(donor_order):
            donor_rows = rows[rows["donor"] == donor].set_index("gene").loc[common]
            matrix[di] = donor_rows["slope"].to_numpy()
        cm_local = np.corrcoef(matrix)
        off = cm_local[~np.eye(len(donor_order), dtype=bool)]
        return {
            "donors": donor_order,
            "gene_set_size": len(common),
            "genes": common,
            "cross_donor_correlation_matrix": cm_local.tolist(),
            "off_diagonal_mean": float(off.mean()),
            "off_diagonal_min": float(off.min()),
            "off_diagonal_max": float(off.max()),
        }

    strict_summary = correlation_summary(pd_rows, donors)
    gene_set = strict_summary.pop("genes")
    cm = np.asarray(strict_summary["cross_donor_correlation_matrix"])
    mean_off = strict_summary["off_diagonal_mean"]
    min_off = strict_summary["off_diagonal_min"]
    print(f"[B9] Per-donor gene set (strict-panel ∩ all donors): {len(gene_set)} genes")
    sev14_5 = bool(min_off < 0.70)

    print(f"[B9] Cross-donor off-diagonal r: mean={mean_off:.3f}, min={min_off:.3f}")
    print(f"[B9] §13: mean ≥ 0.85? {mean_off >= 0.85}; min ≥ 0.70? {min_off >= 0.70}")

    # --- Three-cohort sub-check ---
    print("[B9] Three-cohort sub-check (LHD-8 / LHD-5 non-steatotic / LHD-4 C5-clean)")
    slope_full, gene_names = cohort_slopes(adata, list(LHD_SAMPLES), panel_strict)
    slope_ns, _ = cohort_slopes(adata, list(LHD_NON_STEATOTIC), panel_strict)
    slope_c5c, _ = cohort_slopes(adata, list(C5_CLEAN_LHD), panel_strict)

    # Across-gene slope correlation.
    valid = ~np.isnan(slope_full) & ~np.isnan(slope_ns) & ~np.isnan(slope_c5c)
    r_full_ns, _ = pearsonr(slope_full[valid], slope_ns[valid])
    r_full_c5c, _ = pearsonr(slope_full[valid], slope_c5c[valid])

    # Headline sign disagreement vs full.
    full_signs = np.sign(slope_full[headline_idx])
    ns_signs = np.sign(slope_ns[headline_idx])
    c5c_signs = np.sign(slope_c5c[headline_idx])
    n_disagree_ns = int((full_signs != ns_signs).sum())
    n_disagree_c5c = int((full_signs != c5c_signs).sum())

    # Broad-panel reproducibility and its top cohort-slope quartile. These were
    # previously reported in the manuscript but produced only by an external
    # ad-hoc script.
    relaxed_rows = df[(df["scope"] == "per_donor") & (df["axis"] == "s")
                      & (df["n_bins"] == N_BINS_DEFAULT)
                      & (df.get("panel", "strict") == "relaxed")]
    if relaxed_rows.empty:
        raise RuntimeError("[B9] no per-donor relaxed-panel rows in B6 parquet — re-run B6")
    relaxed_donors = sorted(relaxed_rows["donor"].unique().tolist())
    relaxed_summary = correlation_summary(relaxed_rows, relaxed_donors)
    relaxed_summary.pop("genes")
    cohort_relaxed = df[(df["scope"] == "cohort") & (df["axis"] == "s")
                        & (df["n_bins"] == N_BINS_DEFAULT)
                        & (df.get("panel", "strict") == "relaxed")].copy()
    threshold = float(cohort_relaxed["slope"].abs().quantile(0.75))
    top_genes = cohort_relaxed.loc[
        cohort_relaxed["slope"].abs() >= threshold, "gene"
    ].astype(str).tolist()
    top_summary = correlation_summary(relaxed_rows, relaxed_donors, top_genes)
    top_summary.pop("genes")
    top_summary["threshold_abs_cohort_slope"] = threshold
    relaxed_summary["top_quartile_by_abs_cohort_slope"] = top_summary
    print(f"[B9] Relaxed panel: n={relaxed_summary['gene_set_size']}, "
          f"mean r={relaxed_summary['off_diagonal_mean']:.3f}, "
          f"min r={relaxed_summary['off_diagonal_min']:.3f}")
    print(f"[B9] Relaxed top quartile: n={top_summary['gene_set_size']}, "
          f"mean r={top_summary['off_diagonal_mean']:.3f}, "
          f"min r={top_summary['off_diagonal_min']:.3f}")

    summary = {
        **strict_summary,
        "passes_section13_mean_85": bool(mean_off >= 0.85),
        "passes_section13_min_70": bool(min_off >= 0.70),
        "sev14_5_failure": sev14_5,
        "relaxed_panel": relaxed_summary,
        "three_cohort_sensitivity": {
            "n_headline": int(len(headline_idx)),
            "across_gene_r_full_vs_LHD5_nonsteatotic": float(r_full_ns),
            "across_gene_r_full_vs_LHD4_C5clean": float(r_full_c5c),
            "n_headline_sign_disagreements_LHD5_vs_full": n_disagree_ns,
            "n_headline_sign_disagreements_LHD4C5clean_vs_full": n_disagree_c5c,
            "passes_section_3_3_LHD5": bool(n_disagree_ns <= 2),
            "passes_stageA_C5_subcohort": bool(n_disagree_c5c <= 2),
        },
    }
    OUT_B9.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[B9] Wrote {OUT_B9}")
    print(f"[B9] LHD-8 vs LHD-5: r={r_full_ns:.3f}, sign disagreements={n_disagree_ns}")
    print(f"[B9] LHD-8 vs LHD-4 C5-clean: r={r_full_c5c:.3f}, sign disagreements={n_disagree_c5c}")


# ===========================================================================
# B11 — Stage B Variant B addendum
# ===========================================================================
# NOTE: build_axis_from_panel, load_supp_table_2, derive_directions are
# imported from sgd.axis (legacy B11 defined them inline).

def run_b11() -> None:
    """B11 — Variant-B axis + Tier 2 + cross-pipeline approach (b).
    SOLE writer of obs.s_v_B into visium_human.h5ad."""
    print(f"[B11] Loading {VISIUM}")
    adata = sc.read_h5ad(VISIUM)

    # --- Read Supplementary Table 2 ---
    print(f"[B11] Reading {SUPP_T2}")
    panel_df = load_supp_table_2()
    print(f"[B11]   panel size (raw): {len(panel_df)} genes")
    directions = derive_directions(panel_df)
    n_central = sum(1 for v in directions.values() if v == +1)
    n_portal = sum(1 for v in directions.values() if v == -1)
    n_undirected = sum(1 for v in directions.values() if v == 0)
    print(f"[B11]   directional genes: {n_central} central, {n_portal} portal, "
          f"{n_undirected} undirected/below-threshold")

    # --- Restrict to leave-all-test-markers-out subset ---
    test_markers = set(KNOWN_LIVER_DIRECTION)
    central_panel = [g for g, v in directions.items() if v == +1 and g not in test_markers]
    portal_panel = [g for g, v in directions.items() if v == -1 and g not in test_markers]
    print(f"[B11]   leave-test-markers-out panel: "
          f"{len(portal_panel)} portal + {len(central_panel)} central")

    # Restrict to genes present in our Visium panel.
    visium_genes = set(adata.var_names)
    portal_in_visium = [g for g in portal_panel if g in visium_genes]
    central_in_visium = [g for g in central_panel if g in visium_genes]
    print(f"[B11]   panel ∩ Visium: "
          f"{len(portal_in_visium)} portal + {len(central_in_visium)} central")

    # --- Build s_v_B on LHD ∧ ¬fibrotic spots ---
    spot_mask = lhd_analysis_mask(adata)
    sub = adata[spot_mask].copy()
    print(f"[B11] Building s_v_B on {sub.n_obs} LHD ∧ ¬fibrotic spots...")
    s_v_B_lhd, diag = build_axis_from_panel(sub, portal_in_visium, central_in_visium)

    # Write back to full adata (NaN outside LHD ∧ ¬fibrotic).
    s_v_B_full = np.full(adata.n_obs, np.nan)
    s_v_B_full[spot_mask] = s_v_B_lhd
    adata.obs["s_v_B"] = s_v_B_full

    print("[B11] Per-sample axis diagnostics:")
    for sid, d in diag.items():
        if d.get("status") == "ok":
            flip = " (flipped)" if d.get("orientation_flipped") else ""
            print(f"[B11]   {sid}: n={d['n_spots']}{flip}")
        else:
            print(f"[B11]   {sid}: {d.get('status')}")

    # --- Tier 0 marker recovery on s_v_B (literature markers) ---
    sub_full_genes = adata[spot_mask].copy()  # full LHD-detected gene set
    # Per-donor and pooled slope-based marker scoreboard on s_v_B.
    print("[B11] Computing 9-marker recovery on s_v_B...")
    Gs, bc, _, _ = quantile_bin_per_donor_pool(
        sub_full_genes, s_col="s_v_B", n_bins=N_BINS_DEFAULT, sigma=SIGMA_BINS)
    res = per_gene_gradient(Gs, bc)
    var_names = list(sub_full_genes.var_names)

    pooled_recovery: dict[str, int] = {}
    for g, expected in KNOWN_LIVER_DIRECTION.items():
        if g not in var_names:
            continue
        slope_g = res["slope"][var_names.index(g)]
        pooled_recovery[g] = int(np.sign(slope_g) == expected)

    # Per-donor.
    per_donor_recovery: dict[str, dict[str, int]] = {}
    samples_arr = sub_full_genes.obs["sample_id"].astype(str).to_numpy()
    for sample in sorted(np.unique(samples_arr)):
        if sample not in LHD_SAMPLES:
            continue
        m = samples_arr == sample
        sub_d = sub_full_genes[m].copy()
        Gs_d, bc_d, _, _ = quantile_bin_per_donor_pool(
            sub_d, s_col="s_v_B", n_bins=N_BINS_DEFAULT, sigma=SIGMA_BINS)
        if np.isnan(bc_d).all() or len(bc_d) < 3:
            continue
        res_d = per_gene_gradient(Gs_d, bc_d)
        var_names_d = list(sub_d.var_names)
        donor_recovery: dict[str, int] = {}
        for g, expected in KNOWN_LIVER_DIRECTION.items():
            if g not in var_names_d:
                continue
            slope_g = res_d["slope"][var_names_d.index(g)]
            donor_recovery[g] = int(np.sign(slope_g) == expected)
        per_donor_recovery[str(sample)] = donor_recovery

    n_correct_pooled = sum(pooled_recovery.values())
    n_total_pooled = len(pooled_recovery)
    per_marker_donor_count = {
        g: sum(per_donor_recovery[d].get(g, 0) for d in per_donor_recovery)
        for g in KNOWN_LIVER_DIRECTION if g in pooled_recovery
    }
    n_donors_lhd = len(per_donor_recovery)
    sev14_7_failure = (n_correct_pooled < n_total_pooled) or any(
        c < 7 for c in per_marker_donor_count.values()
    )

    print(f"[B11] Pooled 9-marker recovery on s_v_B: "
          f"{n_correct_pooled}/{n_total_pooled}")
    print(f"[B11] Per-marker per-donor counts (out of {n_donors_lhd}):")
    for g, c in per_marker_donor_count.items():
        print(f"[B11]   {g}: {c}/{n_donors_lhd}")
    print(f"[B11] §14 #7 failure: {sev14_7_failure}")

    # --- §9.7 cross-pipeline reproducibility on the published directional table ---
    # Approach (b): aggregate Supp Table 2 directions vs our slope signs on
    # the same genes from B6's parquet (axis = s, n_bins = 50, panel = strict)
    # plus the relaxed-panel rows for genes outside strict-66.
    print("[B11] Computing §9.7 cross-pipeline reproducibility (Supp Table 2 vs our slopes)...")
    grad_df = pd.read_parquet(GRAD)
    # Use the relaxed-panel cohort row at default n_bins for maximum gene coverage.
    rows = grad_df[(grad_df["scope"] == "cohort") & (grad_df["axis"] == "s")
                    & (grad_df["n_bins"] == N_BINS_DEFAULT)
                    & (grad_df.get("panel", "strict") == "relaxed")]
    if rows.empty:
        rows = grad_df[(grad_df["scope"] == "cohort") & (grad_df["axis"] == "s")
                        & (grad_df["n_bins"] == N_BINS_DEFAULT)]
    our_slope = rows.drop_duplicates("gene").set_index("gene")["slope"]

    # Restrict to genes with a defined direction in Supp Table 2 AND present in our slope vector.
    common = set(our_slope.index) & {g for g, v in directions.items() if v != 0}
    common = {
        g for g in common
        if np.isfinite(our_slope[g]) and int(np.sign(our_slope[g])) != 0
    }
    pub_dir = {g: directions[g] for g in common}
    our_dir = {g: int(np.sign(our_slope[g])) for g in common}
    n_compared = len(common)
    n_agree = sum(1 for g in common if our_dir[g] == pub_dir[g])
    sign_agreement_pct = 100.0 * n_agree / max(1, n_compared)
    # Restrict to high-confidence subset (q < 0.05) — Supp Table 2 has q-values.
    high_conf_genes = set()
    if any(c.lower().startswith("q") for c in panel_df.columns):
        q_col = next(c for c in panel_df.columns if c.lower().startswith("q"))
        try:
            qvals = pd.to_numeric(panel_df[q_col], errors="coerce")
            high_conf_genes = set(panel_df.index[qvals < 0.05])
        except Exception:
            pass
    common_hc = common & high_conf_genes
    n_agree_hc = sum(1 for g in common_hc if our_dir[g] == pub_dir[g])
    sign_agreement_hc = 100.0 * n_agree_hc / max(1, len(common_hc))

    print(f"[B11]   published-q directional overlap: {n_compared} genes compared, "
          f"sign agreement = {sign_agreement_pct:.2f}%")
    print(f"[B11]   high-confidence (q<0.05) subset: {len(common_hc)} genes, "
          f"sign agreement = {sign_agreement_hc:.2f}%")

    # Persist the exact gene-level data behind the primary Figure 2B panel.
    # The figure must not reconstruct a subtly different intersection from the
    # source spreadsheet. In particular, the published direction call also
    # applies the source table's minimum-expression rule, which excludes one
    # q<0.05 row that a raw non-zero layer difference alone would retain.
    panel_unique = panel_df[~panel_df.index.duplicated(keep="first")]
    published_diff = (
        panel_unique[["Mean_Zone_1", "Mean_Zone_2"]].mean(axis=1)
        - panel_unique[["Mean_Zone_7", "Mean_Zone_8"]].mean(axis=1)
    )
    cross_genes = sorted(common_hc)
    cross_gene_table = pd.DataFrame({
        "gene": cross_genes,
        "framework_slope": our_slope.reindex(cross_genes).to_numpy(),
        "published_central_minus_portal": published_diff.reindex(cross_genes).to_numpy(),
        "framework_direction": [our_dir[g] for g in cross_genes],
        "published_direction": [pub_dir[g] for g in cross_genes],
    })
    cross_gene_table["sign_agreement"] = (
        cross_gene_table["framework_direction"]
        == cross_gene_table["published_direction"]
    )
    cross_gene_table.to_parquet(
        OUT_CROSS_PIPELINE_GENES, compression="zstd", index=False
    )

    # --- Save addendum + cross-pipeline JSON ---
    addendum = {
        "supp_table_source": str(SUPP_T2),
        "panel_size_raw": int(len(panel_df)),
        "directional_panel_size": {"central": n_central, "portal": n_portal,
                                    "undirected": n_undirected},
        "leave_test_markers_out": {
            "n_portal": len(portal_panel),
            "n_central": len(central_panel),
            "n_portal_intersected_with_visium": len(portal_in_visium),
            "n_central_intersected_with_visium": len(central_in_visium),
        },
        "axis_per_sample": diag,
        "marker_recovery": {
            "pooled": pooled_recovery,
            "pooled_summary": f"{n_correct_pooled}/{n_total_pooled}",
            "per_donor": per_donor_recovery,
            "per_marker_donor_count": per_marker_donor_count,
            "n_donors_lhd": n_donors_lhd,
        },
        "sev14_7_failure": bool(sev14_7_failure),
        "passes_sec13_variant_B": bool(
            n_correct_pooled == n_total_pooled
            and all(c >= 7 for c in per_marker_donor_count.values())
        ),
        "cohort_overlap_caveat": (
            "Supp Table 2's panel was constructed on the paper's non-steatotic 5 "
            "(M4-M8), a subset of our LHD cohort. Variant B is therefore "
            "different-pipeline + partially-overlapping-cohort, not "
            "different-pipeline + independent-cohort."
        ),
    }
    OUT_ADDENDUM.write_text(json.dumps(addendum, indent=2, default=str))
    print(f"[B11] Wrote {OUT_ADDENDUM}")

    cross_pipeline = {
        "supp_table_source": str(SUPP_T2),
        "method": "approach_b_aggregate_directions_then_compare_signs",
        "panel_size": int(len(panel_df)),
        "n_with_direction": int(n_central + n_portal),
        "our_panel_used": "relaxed (≥6/8 LHD donors detected)",
        "n_bins": int(N_BINS_DEFAULT),
        "high_confidence_definition": (
            "q<0.05 in Yakubovsky Supplementary Table 2, intersected with "
            "genes having a non-zero published direction after the source "
            "minimum-expression rule and a non-zero framework direction"
        ),
        "n_genes_compared": int(n_compared),
        "n_sign_agreement": int(n_agree),
        "sign_agreement_pct": float(sign_agreement_pct),
        "high_confidence_q_lt_005": {
            "n_genes_compared": int(len(common_hc)),
            "n_sign_agreement": int(n_agree_hc),
            "sign_agreement_pct": float(sign_agreement_hc),
        },
        "passes_prespecified_80pct": bool(sign_agreement_pct >= 80.0),
        "cohort_overlap_caveat": addendum["cohort_overlap_caveat"],
    }
    OUT_CROSS_PIPELINE.write_text(json.dumps(cross_pipeline, indent=2))
    print(f"[B11] Wrote {OUT_CROSS_PIPELINE}")
    print(f"[B11] Wrote {OUT_CROSS_PIPELINE_GENES}")

    # --- Save updated h5ad with non-NaN obs.s_v_B ---
    adata.write_h5ad(VISIUM, compression="gzip")
    print(f"[B11] Updated {VISIUM} with obs.s_v_B")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    # Run order: B3/B4 (Tier 0/1), then B7 (writes headline_candidates.json),
    # then B8 + B9 (both read it), then B11 last (writes obs.s_v_B).
    run_b3_b4()
    run_b7()
    run_b8()
    run_b9()
    run_b11()


if __name__ == "__main__":
    main()
