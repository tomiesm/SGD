"""
12 — Per-gene spatial-pattern (per-tertile g(s) curves) for steatosis genes.

REBUILT ORPHAN. Re-authored from legacy ``stage_D/D11_spatial_patterns.py``.
The single re-authoring change versus legacy D11 is **parametrising the gene
set**: legacy D11 hard-coded ``ROBUST_GENES = ("GLUL", "LYZ", "ORM1")`` at
module scope and iterated it directly; here that tuple becomes the function
parameter ``genes``. Every other line — ``N_BINS = 20``, ``N_BOOT = 200``,
the lipid tertile cuts at the 33rd / 67th percentile, the ``build_p6_axis``
recipe, the per-bin bootstrap, the ``np.random.default_rng(0)`` seed and the
fixed iteration order — is copied verbatim from D11. The figure-PNG
side-effects of legacy D11 are dropped (figures are rendered separately by
``scripts/figures/``); only the JSON outputs are kept.

One script run produces both gene-set JSONs:
  - ``d11_spatial_patterns.json``        — the 3 robust genes (GLUL/LYZ/ORM1).
  - ``d11_spatial_patterns_called8.json`` — the 8-gene "called" set
    (robust_1234 + donor_sensitive_12 + weak_1, read from
    ``steatosis_call_criteria.parquet``). This variant additionally carries a
    ``_classification`` block of per-tertile diagnostics per gene.
Plus the MT1H leave-M1 follow-up ``d11_mt1h_leave_m1.json`` via ``--leave-out``.

Reads:  RESULTS/visium_human.h5ad (needs obs.lipid_pct from script 11),
        RESULTS/steatosis_call_criteria.parquet.
Writes: RESULTS/d11_spatial_patterns.json, RESULTS/d11_spatial_patterns_called8.json,
        RESULTS/d11_mt1h_leave_m1.json.
Feeds:  §4.9 / results-weak1 — R4.6 (tertile cuts), Fig 4B, R5.4 (SAA1 per-
        tertile Δ), supp S6 (reads _called8.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgd import config
from sgd.config import RESULTS, STEATOTIC_DONORS
from sgd.axis import build_p6_axis
from sgd.steatosis import per_bin_mean_with_bootstrap, steatotic_analysis_mask

warnings.filterwarnings("ignore")
np.random.seed(0)

VISIUM = RESULTS / "visium_human.h5ad"
CALL = RESULTS / "steatosis_call_criteria.parquet"
OUT_JSON = RESULTS / "d11_spatial_patterns.json"
OUT_JSON_CALLED8 = RESULTS / "d11_spatial_patterns_called8.json"
OUT_JSON_MT1H = RESULTS / "d11_mt1h_leave_m1.json"

# Verbatim from legacy D11 — the 3-gene robust set is now the default
# argument of the re-authored compute function, not a module constant.
ROBUST_GENES = ("GLUL", "LYZ", "ORM1")
N_BINS = 20
N_BOOT = 200


def _classification(tertiles: dict) -> dict:
    """Per-tertile diagnostics for the 8-gene called set — the documented
    weak_1 extension (results_verification_weak1_extended.md §A): Δ(high−low)
    at the portal half (lower 10 bins), the central half (upper 10 bins), and
    overall; plus the per-tertile range and the high/low range ratio."""
    lo = np.asarray(tertiles["low"]["bin_means"], dtype=float)
    hi = np.asarray(tertiles["high"]["bin_means"], dtype=float)
    delta = hi - lo
    half = N_BINS // 2
    range_lo = float(np.nanmax(lo) - np.nanmin(lo))
    range_hi = float(np.nanmax(hi) - np.nanmin(hi))
    return {
        "delta_high_minus_low_mean_low_s": float(np.nanmean(delta[:half])),
        "delta_high_minus_low_mean_high_s": float(np.nanmean(delta[half:])),
        "delta_high_minus_low_overall": float(np.nanmean(delta)),
        "range_lo_tertile": range_lo,
        "range_hi_tertile": range_hi,
        "range_ratio_hi_over_lo": range_hi / range_lo,
    }


def compute_spatial_patterns(adata: sc.AnnData, genes: tuple,
                              with_classification: bool = False) -> dict:
    """Per-gene per-tertile g(s) curves. Body verbatim from legacy D11's
    ``main()`` (cohort 3-panel block) with the gene loop iterating the
    ``genes`` parameter instead of the module-scope ``ROBUST_GENES``."""
    adata.obs["s"] = build_p6_axis(adata)
    spot_mask = steatotic_analysis_mask(adata)
    s_all = adata.obs["s"].to_numpy()
    spot_mask = spot_mask & np.isfinite(s_all)
    sub = adata[spot_mask].copy()
    print(f"[D11]   cohort spots: {sub.n_obs}")
    print(f"[D11]   per-donor: "
          f"{sub.obs['sample_id'].value_counts().to_dict()}")

    # Lipid tertiles (per the full pooled cohort).
    lipid = sub.obs["lipid_pct"].to_numpy().astype(float)
    s = sub.obs["s"].to_numpy().astype(float)
    donor = sub.obs["sample_id"].astype(str).to_numpy()
    q33, q67 = np.percentile(lipid, [33, 67])
    print(f"[D11]   lipid tertile cuts: {q33:.3f}, {q67:.3f}")

    tertile = np.where(lipid <= q33, "low",
                np.where(lipid <= q67, "mid", "high"))

    # Get gene matrix (already log-normalised in Stage A).
    var = list(sub.var_names)
    results: dict = {"lipid_tertile_cuts": [float(q33), float(q67)],
                      "n_bins": N_BINS, "per_gene": {}}

    rng = np.random.default_rng(0)
    X = sub.X
    if issparse(X):
        X = X.toarray()

    for gene in genes:
        if gene not in var:
            print(f"[D11] {gene}: not in var")
            continue
        gi = var.index(gene)
        x = X[:, gi]
        gene_results = {}
        for label in ("low", "mid", "high"):
            m = tertile == label
            n = int(m.sum())
            if n < 50:
                gene_results[label] = {"n_spots": n, "status": "too_few"}
                continue
            centers, means, ci_lo, ci_hi = per_bin_mean_with_bootstrap(
                s[m], x[m], N_BINS, rng=rng)
            gene_results[label] = {
                "n_spots": n,
                "bin_centers": centers.tolist(),
                "bin_means": means.tolist(),
                "bin_ci_lo": ci_lo.tolist(),
                "bin_ci_hi": ci_hi.tolist(),
            }
        if with_classification:
            gene_results["_classification"] = _classification(gene_results)
        results["per_gene"][gene] = gene_results

    return results


def compute_leave_out(adata: sc.AnnData, gene: str) -> dict:
    """MT1H leave-M1 follow-up — same per-tertile pipeline run on three
    cohort subsets (full / leave-M1 / M1-only). Each subset recomputes its
    own lipid tertiles. Verbatim per-tertile recipe from D11."""
    adata.obs["s"] = build_p6_axis(adata)
    spot_mask = steatotic_analysis_mask(adata)
    s_all = adata.obs["s"].to_numpy()
    spot_mask = spot_mask & np.isfinite(s_all)
    sub_full = adata[spot_mask].copy()

    cohort_defs = {
        "full_M1M2M3P6": list(STEATOTIC_DONORS),
        "leave_M1_M2M3P6": [d for d in STEATOTIC_DONORS if d != "M1"],
        "M1_only": ["M1"],
    }
    out: dict = {"gene": gene, "n_bins": N_BINS, "cohorts": {}}

    for name, donors in cohort_defs.items():
        donor_arr = sub_full.obs["sample_id"].astype(str).to_numpy()
        cohort_mask = np.isin(donor_arr, donors)
        sub = sub_full[cohort_mask].copy()
        lipid = sub.obs["lipid_pct"].to_numpy().astype(float)
        s = sub.obs["s"].to_numpy().astype(float)
        q33, q67 = np.percentile(lipid, [33, 67])
        tertile = np.where(lipid <= q33, "low",
                    np.where(lipid <= q67, "mid", "high"))
        var = list(sub.var_names)
        rng = np.random.default_rng(0)
        X = sub.X
        if issparse(X):
            X = X.toarray()
        cohort_results: dict = {
            "tertile_cuts": [float(q33), float(q67)],
            "n_total_spots": int(sub.n_obs),
        }
        if gene not in var:
            print(f"[D11] {gene}: not in var")
            out["cohorts"][name] = cohort_results
            continue
        gi = var.index(gene)
        x = X[:, gi]
        for label in ("low", "mid", "high"):
            m = tertile == label
            n = int(m.sum())
            if n < 50:
                cohort_results[label] = {"n_spots": n, "status": "too_few"}
                continue
            centers, means, ci_lo, ci_hi = per_bin_mean_with_bootstrap(
                s[m], x[m], N_BINS, rng=rng)
            cohort_results[label] = {
                "n_spots": n,
                "bin_centers": centers.tolist(),
                "bin_means": means.tolist(),
                "bin_ci_lo": ci_lo.tolist(),
                "bin_ci_hi": ci_hi.tolist(),
            }
        lo = np.asarray(cohort_results["low"]["bin_means"], dtype=float)
        hi = np.asarray(cohort_results["high"]["bin_means"], dtype=float)
        delta = hi - lo
        half = N_BINS // 2
        range_lo = float(np.nanmax(lo) - np.nanmin(lo))
        range_hi = float(np.nanmax(hi) - np.nanmin(hi))
        cohort_results["_classification"] = {
            "portal_delta": float(np.nanmean(delta[:half])),
            "central_delta": float(np.nanmean(delta[half:])),
            "range_lo": range_lo,
            "range_hi": range_hi,
            "range_ratio_hi_over_lo": range_hi / range_lo,
        }
        out["cohorts"][name] = cohort_results
        print(f"[D11]   {name}: n={sub.n_obs}, cuts={[round(q33,4), round(q67,4)]}")

    return out


def called_genes_from_parquet() -> tuple:
    """The 8-gene 'called' set: criterion conjunction flags robust_1234 +
    donor_sensitive_12 + weak_1, in parquet-row order."""
    df = pd.read_parquet(CALL)
    genes: list[str] = []
    for flag in ("robust_1234", "donor_sensitive_12", "weak_1"):
        genes.extend(df[df["four_criteria_flag"] == flag]["gene"].tolist())
    return tuple(genes)


def main() -> None:
    parser = argparse.ArgumentParser(description="D11 rebuild — per-tertile g(s).")
    parser.add_argument("--gene-set", choices=["robust3", "called8", "both"],
                        default="both")
    parser.add_argument("--leave-out", action="store_true",
                        help="also run the MT1H leave-M1 follow-up.")
    args = parser.parse_args()

    print(f"[D11] Loading {VISIUM}")
    adata = sc.read_h5ad(VISIUM)

    if args.gene_set in ("robust3", "both"):
        res3 = compute_spatial_patterns(adata, ROBUST_GENES,
                                        with_classification=False)
        OUT_JSON.write_text(json.dumps(res3, indent=2, default=str))
        print(f"[D11] Wrote {OUT_JSON}")

    if args.gene_set in ("called8", "both"):
        called8 = called_genes_from_parquet()
        print(f"[D11]   8-gene called set: {called8}")
        res8 = compute_spatial_patterns(adata, called8,
                                        with_classification=True)
        OUT_JSON_CALLED8.write_text(json.dumps(res8, indent=2, default=str))
        print(f"[D11] Wrote {OUT_JSON_CALLED8}")

    if args.leave_out:
        res_mt1h = compute_leave_out(adata, "MT1H")
        OUT_JSON_MT1H.write_text(json.dumps(res_mt1h, indent=2, default=str))
        print(f"[D11] Wrote {OUT_JSON_MT1H}")


if __name__ == "__main__":
    main()
