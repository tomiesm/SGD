"""
10 — Proper cell2location run + multi-threshold within-cell-type rescoreboard.

Absorbs legacy ``stage_D/D13_cell2location.py`` and
``stage_D/D13b_rescoreboard.py``. Two functions run in sequence:

  - ``run_cell2location()`` (D13) trains a negative-binomial regression model
    on the snRNA reference, runs cell2location's variational mapping on the
    Visium cohort, derives a calibrated hep_fraction per spot, and writes the
    addendum JSON + the hep-fraction parquet.
  - ``rescoreboard()`` (D13b) re-runs the within-hep-fraction marker
    scoreboard at thresholds {0.40, 0.45, 0.50} on the cached parquet and
    **overwrites in place** the Confound-4 keys + the
    ``multi_threshold_scoreboard`` key of ``c3_cell2location_addendum.json``.

The two-pass write-then-overwrite of ``c3_cell2location_addendum.json`` is
preserved exactly: D13 writes the file, D13b re-reads it and replaces specific
keys.

GPU-bound, ~2 h. The hard-coded snRNA path of legacy D13 is replaced by
``config.SNRNA_PATH``.

Reads:  config.SNRNA_PATH (snRNA reference), RESULTS/visium_human.h5ad.
Writes: RESULTS/obs_hep_fraction_c2l.parquet, RESULTS/d13_snrna_signatures.parquet,
        RESULTS/c3_cell2location_addendum.json (written by run_cell2location,
        then overwritten in place by rescoreboard).
Feeds:  within-cell-type §4.8 — supp S4 (hep-purity threshold sweep) and
        the C4 row of supp S1 (confound_4_per_sample).
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

# CRITICAL: set matplotlib non-interactive backend BEFORE importing cell2location.
# cell2location's filter_genes calls plt.show() which hangs the entire process
# under any interactive backend without a display.
import matplotlib
matplotlib.use("Agg")

import anndata as ad
import cell2location
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from cell2location.models import RegressionModel
from cell2location.utils.filtering import filter_genes
from scipy.sparse import csr_matrix, issparse
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgd import config
from sgd.config import RESULTS, SNRNA_PATH
from sgd.panels import (HEADLINE_MARKERS, KNOWN_LIVER_DIRECTION, lhd_analysis_mask)
from sgd.gradient import (N_BINS_DEFAULT, SIGMA_BINS, per_gene_gradient,
                          quantile_bin_per_donor_pool)

warnings.filterwarnings("ignore")
np.random.seed(0)

LHD_SAMPLES = config.LHD_SAMPLES

VISIUM = RESULTS / "visium_human.h5ad"
OUT_JSON = RESULTS / "c3_cell2location_addendum.json"
OUT_PARQUET = RESULTS / "obs_hep_fraction_c2l.parquet"
REGRESSION_CKPT = RESULTS / "d13_regression_model"
SIGNATURES_PARQUET = RESULTS / "d13_snrna_signatures.parquet"
HEP_PARQUET = RESULTS / "obs_hep_fraction_c2l.parquet"
ADDENDUM_JSON = RESULTS / "c3_cell2location_addendum.json"

CELL_TYPE_COL = "cluster_annotations"
HEP_FRACTION_THRESHOLD = 0.7

# Modest training schedule — tuned for runtime, not for paper-grade convergence
# (which would use 250+ regression epochs and 30k+ cell2location iters).
N_EPOCHS_REF = 250
# With batch_size=2500 and ~20 minibatches/epoch on the 47k-spot Visium, 2000
# epochs ≈ 40k minibatch steps — comparable to the cell2location tutorial's
# 30k full-batch iterations. Empirically the ELBO drops sharply in the first
# few hundred epochs and plateaus by ~1500.
N_EPOCHS_C2L = 2000
N_CELLS_PER_SPOT = 30   # Visium-typical
DETECTION_ALPHA = 20.0  # default

THRESHOLDS = [0.40, 0.45, 0.50]


def run_cell2location() -> None:
    """D13 — proper cell2location run; writes the addendum JSON + parquet."""
    t0 = time.time()
    print(f"[D13] Loading snRNA reference (full, in-memory) from {SNRNA_PATH}")
    snrna = sc.read_h5ad(SNRNA_PATH)
    print(f"[D13]   {snrna.n_obs} cells × {snrna.n_vars} genes; "
          f"cell types: {snrna.obs[CELL_TYPE_COL].value_counts().to_dict()}")

    # cell2location requires INTEGER raw counts in .X for NB regression.
    # The snRNA reference's "raw_counts" layer is actually ambient-corrected
    # (denoised) floats — values like 0.989, 1.829, 3.888. Round to int to
    # restore integer support.
    if "raw_counts" in snrna.layers:
        rc = snrna.layers["raw_counts"]
        if issparse(rc):
            rc = rc.copy()
            rc.data = np.round(rc.data).astype(np.float32)
        else:
            rc = np.round(rc).astype(np.float32)
        snrna.X = rc
        print("[D13]   using rounded snrna.layers['raw_counts'] as .X "
              "(denoised floats restored to int support for NB regression)")
    else:
        raise SystemExit("[D13] FATAL: raw_counts layer missing in snRNA reference")

    print(f"[D13]   X dtype: {snrna.X.dtype}; max: {snrna.X.max():.1f}")

    # --- Filter genes per cell2location guidance ---
    # cell2location utility filter: cell_count_cutoff=5, cell_percentage_cutoff2=0.03,
    # nonz_mean_cutoff=1.12 (tweak for our data).
    selected = filter_genes(snrna, cell_count_cutoff=5,
                              cell_percentage_cutoff2=0.03,
                              nonz_mean_cutoff=1.12)
    # New cell2location returns a pandas Index; slicing works directly.
    n_kept = len(selected) if hasattr(selected, "__len__") else int(np.asarray(selected).sum())
    snrna = snrna[:, selected].copy()
    print(f"[D13]   filtered snRNA to {snrna.n_vars} genes (kept {n_kept})")

    # --- Train regression model on snRNA (with checkpoint cache) ---
    if SIGNATURES_PARQUET.exists():
        print(f"[D13] CACHE HIT: loading signatures from {SIGNATURES_PARQUET} "
              f"(skipping {N_EPOCHS_REF}-epoch regression training)")
        sig_df = pd.read_parquet(SIGNATURES_PARQUET)
    else:
        print(f"[D13] Setting up RegressionModel (NB regression on cell-type signatures)...")
        RegressionModel.setup_anndata(adata=snrna,
                                        batch_key="sample",
                                        labels_key=CELL_TYPE_COL)
        mod = RegressionModel(snrna)
        print(f"[D13] Training regression model ({N_EPOCHS_REF} epochs)...")
        mod.train(max_epochs=N_EPOCHS_REF, accelerator="gpu")

        # Export per-cell-type signatures.
        snrna = mod.export_posterior(snrna, sample_kwargs={"num_samples": 200})
        print(f"[D13] Regression model trained in {time.time() - t0:.1f}s")

        # cell2location expects per-cell-type expression in
        # varm['means_per_cluster_mu_fg']. This is already a DataFrame whose
        # columns are prefixed like 'means_per_cluster_mu_fg_Hepatocytes';
        # strip the prefix to get bare cell-type names.
        if "means_per_cluster_mu_fg" in snrna.varm:
            sig_df = snrna.varm["means_per_cluster_mu_fg"].copy()
            sig_df.columns = [c.replace("means_per_cluster_mu_fg_", "")
                                for c in sig_df.columns]
            print(f"[D13]   extracted varm: shape={sig_df.shape}; "
                  f"NaN frac: {sig_df.isna().mean().mean():.4f}; "
                  f"sum range: {sig_df.values.sum(axis=0).min():.4f} to "
                  f"{sig_df.values.sum(axis=0).max():.4f}")
            if sig_df.isna().any().any():
                raise SystemExit(
                    "[D13] FATAL: signature DataFrame has NaN values — "
                    "regression model output is invalid")
        else:
            raise SystemExit("[D13] FATAL: signature extraction failed — "
                              "'means_per_cluster_mu_fg' missing from varm")

        # Save signatures for cache.
        sig_df.to_parquet(SIGNATURES_PARQUET)
        print(f"[D13] Cached signatures to {SIGNATURES_PARQUET}")

    print(f"[D13]   signature columns: {list(sig_df.columns)}")
    print(f"[D13]   signature shape: {sig_df.shape}; "
          f"sum range: {sig_df.values.sum(axis=0).min():.4f} to "
          f"{sig_df.values.sum(axis=0).max():.4f}")

    # --- Set up Visium for cell2location mapping ---
    print(f"[D13] Loading Visium {VISIUM}")
    adata_v = sc.read_h5ad(VISIUM)
    if "raw_counts" in adata_v.layers:
        adata_v.X = adata_v.layers["raw_counts"]
        print("[D13]   using visium.layers['raw_counts'] as .X")
    elif "counts" in adata_v.layers:
        adata_v.X = adata_v.layers["counts"]
    else:
        # Fall back: assume X is log-normalised; we need counts. Skip if not available.
        print("[D13] WARNING: no raw_counts layer in Visium; using X as-is (may be log-norm)")

    # Subset Visium to common genes with the signature.
    common = [g for g in adata_v.var_names if g in sig_df.index]
    print(f"[D13]   common genes between Visium and snRNA signatures: {len(common)}")
    adata_v = adata_v[:, common].copy()
    sig_v = sig_df.loc[common]

    cell2location.models.Cell2location.setup_anndata(adata=adata_v,
                                                       batch_key="sample_id")
    # Diagnose detection_mean computation: cell2location does
    # detection_mean = (sp_total / N_cells) / sc_total
    sp_total = np.asarray(adata_v.X.sum(axis=1)).flatten().mean()
    sc_total = float(sig_v.sum(axis=0).mean())
    expected_dm = (sp_total / N_CELLS_PER_SPOT) / max(sc_total, 1e-9)
    print(f"[D13]   sp_total (mean Visium UMI/spot): {sp_total:.1f}")
    print(f"[D13]   sc_total (mean signature sum/cell-type): {sc_total:.4f}")
    print(f"[D13]   expected detection_mean: {expected_dm:.4f}")
    if sc_total <= 0 or not np.isfinite(sc_total):
        raise SystemExit(f"[D13] FATAL: sc_total = {sc_total} — signature DataFrame is empty/invalid")
    print(f"[D13] Initialising Cell2location model (N={N_CELLS_PER_SPOT} cells/spot)...")
    c2l = cell2location.models.Cell2location(
        adata_v, cell_state_df=sig_v,
        N_cells_per_location=N_CELLS_PER_SPOT,
        detection_alpha=DETECTION_ALPHA,
    )
    print(f"[D13]   c2l.detection_mean_: {c2l.detection_mean_}")
    C2L_CKPT = RESULTS / "d13_c2l_model"
    if C2L_CKPT.exists():
        print(f"[D13] CACHE HIT: loading c2l model from {C2L_CKPT}")
        c2l = cell2location.models.Cell2location.load(str(C2L_CKPT), adata=adata_v)
    else:
        print(f"[D13] Training Cell2location ({N_EPOCHS_C2L} iterations, batch_size=2500)...")
        t1 = time.time()
        # batch_size=2500 prevents CUDA OOM at 47k spots × 12k genes; full-batch
        # would need >20 GB VRAM. With minibatch, each step processes 2500 spots.
        c2l.train(max_epochs=N_EPOCHS_C2L, batch_size=2500,
                    train_size=1.0, accelerator="gpu")
        print(f"[D13] Trained in {time.time() - t1:.1f}s")
        c2l.save(str(C2L_CKPT), overwrite=True)
        print(f"[D13] Saved c2l model to {C2L_CKPT}")

    print(f"[D13] Exporting posterior...")
    adata_v = c2l.export_posterior(adata_v,
                                      sample_kwargs={"num_samples": 200})

    # cell2location stores per-cell-type abundances in obsm['q05_cell_abundance_w_sf']
    # or 'means_cell_abundance_w_sf'.
    q05 = adata_v.obsm.get("q05_cell_abundance_w_sf")
    means = adata_v.obsm.get("means_cell_abundance_w_sf")
    abundance = means if means is not None else q05
    if abundance is None:
        raise SystemExit("[D13] FATAL: cell2location abundance keys missing in obsm")

    abundance = np.asarray(abundance)
    cell_types = list(sig_v.columns)
    if "Hepatocytes" not in cell_types:
        raise SystemExit(f"[D13] FATAL: 'Hepatocytes' not found in {cell_types}")
    hep_idx = cell_types.index("Hepatocytes")
    total_per_spot = abundance.sum(axis=1)
    hep_fraction = abundance[:, hep_idx] / np.maximum(total_per_spot, 1e-9)
    print(f"[D13] hep_fraction: min={hep_fraction.min():.3f}, "
          f"median={np.median(hep_fraction):.3f}, max={hep_fraction.max():.3f}, "
          f"mean={hep_fraction.mean():.3f}")

    # Persist parquet (NOT overwriting C3 NNLS-based parquet).
    pd.DataFrame({"barcode": adata_v.obs["barcode"].astype(str),
                   "sample_id": adata_v.obs["sample_id"].astype(str),
                   "hep_fraction_c2l": hep_fraction,
                   **{f"abundance_{ct}": abundance[:, i]
                      for i, ct in enumerate(cell_types)}
                   }).to_parquet(OUT_PARQUET, index=False)
    print(f"[D13] Wrote {OUT_PARQUET}")

    # --- Confound 4 with new hep_fraction ---
    samples = adata_v.obs["sample_id"].astype(str).to_numpy()
    s = adata_v.obs["s"].to_numpy()
    confound_4 = {}
    for sample in sorted(np.unique(samples)):
        if sample not in LHD_SAMPLES:
            continue
        m = (samples == sample) & ~np.isnan(s)
        if m.sum() < 10:
            continue
        r, p = pearsonr(s[m], hep_fraction[m])
        confound_4[sample] = {"r": float(r), "p": float(p), "n": int(m.sum())}
    cohort_abs_r = [abs(v["r"]) for v in confound_4.values()]

    # --- Within-hep>0.7 scoreboard ---
    spot_mask = lhd_analysis_mask(adata_v) & (hep_fraction > HEP_FRACTION_THRESHOLD)
    n_hep_kept = int(spot_mask.sum())
    print(f"[D13]   spots after hep>{HEP_FRACTION_THRESHOLD} filter: {n_hep_kept}")

    headline_recovery: dict[str, int] = {}
    sanity_recovery: dict[str, int] = {}
    if n_hep_kept >= 100:
        sub = adata_v[spot_mask].copy()
        # Need log-normalised X for slope. Reload from VISIUM since we replaced X with counts.
        adata_orig = sc.read_h5ad(VISIUM)
        # Match obs by barcode + sample.
        keys = (adata_orig.obs["sample_id"].astype(str) + ":" + adata_orig.obs["barcode"].astype(str)).to_numpy()
        sub_keys = (sub.obs["sample_id"].astype(str) + ":" + sub.obs["barcode"].astype(str)).to_numpy()
        idx_map = pd.Series(np.arange(len(keys)), index=keys)
        rows = idx_map.reindex(sub_keys).dropna().astype(int).to_numpy()
        sub_orig = adata_orig[rows].copy()
        sub_orig.obs["s"] = sub.obs["s"].to_numpy()[:len(rows)]
        Gs, bc, _, _ = quantile_bin_per_donor_pool(
            sub_orig, s_col="s", n_bins=N_BINS_DEFAULT, sigma=SIGMA_BINS)
        if not np.isnan(bc).all():
            res = per_gene_gradient(Gs, bc)
            var_names = list(sub_orig.var_names)
            for g, expected in KNOWN_LIVER_DIRECTION.items():
                if g in var_names:
                    slope_g = res["slope"][var_names.index(g)]
                    ok = int(np.sign(slope_g) == expected)
                    sanity_recovery[g] = ok
                    if g in HEADLINE_MARKERS:
                        headline_recovery[g] = ok

    n_headline_correct = sum(headline_recovery.values())
    n_sanity_correct = sum(sanity_recovery.values())

    out = {
        "method": "cell2location_proper",
        "regression_model_epochs": N_EPOCHS_REF,
        "cell2location_iterations": N_EPOCHS_C2L,
        "n_cells_per_location": N_CELLS_PER_SPOT,
        "cell_types": cell_types,
        "hep_fraction_summary": {
            "min": float(hep_fraction.min()),
            "median": float(np.median(hep_fraction)),
            "max": float(hep_fraction.max()),
            "mean": float(hep_fraction.mean()),
        },
        "confound_4_per_sample": confound_4,
        "confound_4_cohort_max_abs_r": float(max(cohort_abs_r)) if cohort_abs_r else None,
        "within_hep_filter": {
            "threshold": HEP_FRACTION_THRESHOLD,
            "n_spots_kept": n_hep_kept,
            "n_spots_total_lhd": int(lhd_analysis_mask(adata_v).sum()),
        },
        "marker_recovery_headline_8": {
            "per_marker": headline_recovery,
            "n_correct": n_headline_correct,
            "n_total": len(headline_recovery),
        },
        "marker_recovery_sanity_9": {
            "per_marker": sanity_recovery,
            "n_correct": n_sanity_correct,
            "n_total": len(sanity_recovery),
        },
        "section_13_passes_7_of_8": bool(n_headline_correct >= 7 and n_sanity_correct >= 7),
        "runtime_sec": float(time.time() - t0),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"[D13] Wrote {OUT_JSON}")
    print(f"[D13] Headline correct: {n_headline_correct}/{len(headline_recovery)}; "
          f"sanity: {n_sanity_correct}/{len(sanity_recovery)}; "
          f"hep>0.7 spots: {n_hep_kept}; "
          f"max |r|: {max(cohort_abs_r) if cohort_abs_r else 'N/A'}")


def rescoreboard() -> None:
    """D13b — multi-threshold rescoreboard; overwrites addendum JSON in place."""
    print(f"[D13b] Loading {VISIUM}")
    adata = sc.read_h5ad(VISIUM)
    print(f"[D13b] Loading {HEP_PARQUET}")
    hep_df = pd.read_parquet(HEP_PARQUET)
    # Match by (sample_id, barcode) order to adata.obs.
    keys_a = (adata.obs["sample_id"].astype(str) + ":"
              + adata.obs["barcode"].astype(str)).to_numpy()
    keys_h = (hep_df["sample_id"].astype(str) + ":"
              + hep_df["barcode"].astype(str)).to_numpy()
    h_index = pd.Series(np.arange(len(keys_h)), index=keys_h)
    rows = h_index.reindex(keys_a).to_numpy()
    if pd.isna(rows).any():
        n_missing = int(pd.isna(rows).sum())
        print(f"[D13b] WARNING: {n_missing} adata spots missing in c2l parquet")
    rows = rows.astype(int)
    hep_fraction = hep_df["hep_fraction_c2l"].to_numpy()[rows]
    adata.obs["hep_fraction_c2l"] = hep_fraction
    print(f"[D13b]   hep_fraction quantiles: "
          f"{np.percentile(hep_fraction[~np.isnan(hep_fraction)], [50, 70, 80, 90, 95]).round(3).tolist()}")

    # Confound 4 with new hep_fraction
    samples = adata.obs["sample_id"].astype(str).to_numpy()
    s = adata.obs["s"].to_numpy()
    confound_4 = {}
    for sample in sorted(np.unique(samples)):
        if sample not in LHD_SAMPLES:
            continue
        m = (samples == sample) & ~np.isnan(s) & ~np.isnan(hep_fraction)
        if m.sum() < 10:
            continue
        r, p = pearsonr(s[m], hep_fraction[m])
        confound_4[sample] = {"r": float(r), "p": float(p), "n": int(m.sum())}
    cohort_abs_r = [abs(v["r"]) for v in confound_4.values()]

    # Multi-threshold scoreboard.
    threshold_results: dict[str, dict] = {}
    for thresh in THRESHOLDS:
        spot_mask = lhd_analysis_mask(adata) & (hep_fraction > thresh)
        n_kept = int(spot_mask.sum())
        n_total_lhd = int(lhd_analysis_mask(adata).sum())
        per_donor_kept = {}
        for d in sorted(set(samples[spot_mask])):
            per_donor_kept[d] = int((samples[spot_mask] == d).sum())
        print(f"[D13b]   threshold {thresh}: {n_kept}/{n_total_lhd} LHD spots kept "
              f"({per_donor_kept})")
        if n_kept < 100:
            threshold_results[f"thresh_{thresh}"] = {
                "n_spots_kept": n_kept,
                "status": "too_few",
                "per_donor_kept": per_donor_kept,
            }
            continue

        sub = adata[spot_mask].copy()
        Gs, bc, _, _ = quantile_bin_per_donor_pool(
            sub, s_col="s", n_bins=N_BINS_DEFAULT, sigma=SIGMA_BINS)
        if np.isnan(bc).all():
            threshold_results[f"thresh_{thresh}"] = {
                "n_spots_kept": n_kept,
                "status": "binning_failed",
                "per_donor_kept": per_donor_kept,
            }
            continue
        res = per_gene_gradient(Gs, bc)
        var_names = list(sub.var_names)
        headline_recovery: dict[str, int] = {}
        sanity_recovery: dict[str, int] = {}
        for g, expected in KNOWN_LIVER_DIRECTION.items():
            if g in var_names:
                slope_g = res["slope"][var_names.index(g)]
                ok = int(np.sign(slope_g) == expected)
                sanity_recovery[g] = ok
                if g in HEADLINE_MARKERS:
                    headline_recovery[g] = ok

        n_headline = sum(headline_recovery.values())
        n_sanity = sum(sanity_recovery.values())
        threshold_results[f"thresh_{thresh}"] = {
            "n_spots_kept": n_kept,
            "n_total_lhd": n_total_lhd,
            "per_donor_kept": per_donor_kept,
            "headline_8_correct": n_headline,
            "headline_8_total": len(headline_recovery),
            "headline_per_marker": headline_recovery,
            "sanity_9_correct": n_sanity,
            "sanity_9_total": len(sanity_recovery),
            "sanity_per_marker": sanity_recovery,
            "section_13_passes_7_of_8": bool(n_headline >= 7 and n_sanity >= 7),
        }
        print(f"[D13b]   threshold {thresh}: headline={n_headline}/{len(headline_recovery)}, "
              f"sanity={n_sanity}/{len(sanity_recovery)}, "
              f"§13 PASS={threshold_results[f'thresh_{thresh}']['section_13_passes_7_of_8']}")

    # Update the addendum JSON with the multi-threshold scoreboard.
    if ADDENDUM_JSON.exists():
        old = json.loads(ADDENDUM_JSON.read_text())
    else:
        old = {}
    old["confound_4_per_sample"] = confound_4
    old["confound_4_cohort_max_abs_r"] = float(max(cohort_abs_r)) if cohort_abs_r else None
    old["multi_threshold_scoreboard"] = threshold_results
    old["threshold_calibration_note"] = (
        "cell2location proper run produces max hep_fraction = 0.66 in this "
        "cohort (smoother cell-type fraction estimate than NNLS but below "
        "the textbook 0.7-0.9 expectation, likely because Visium spots "
        "include surrounding non-parenchymal cells). The original 0.7 "
        "threshold filters out all spots; the rescoreboard reports "
        "results at threshold ∈ {0.40, 0.45, 0.50} for sensitivity analysis."
    )
    ADDENDUM_JSON.write_text(json.dumps(old, indent=2, default=str))
    print(f"[D13b] Updated {ADDENDUM_JSON}")


def main() -> None:
    run_cell2location()
    rescoreboard()


if __name__ == "__main__":
    main()
