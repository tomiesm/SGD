"""
Moor 2018 intestinal crypt-villus — Analysis 1: marker recovery on the
zone-based axis.

Second-tissue validation for the SGD manuscript. Loads the Moor et al. 2018
LCM zonation reconstruction (table_D), filters genes, and runs the paper's
per-gene gradient pipeline on the 7 anatomical zones (Crypt -> V1..V6 tip).
The zone-mean profile of each gene is log1p-transformed, Gaussian-smoothed
across the 7 zones with sigma=0.8 (the smoothing step used inside
sgd.gradient.quantile_bin_per_donor_pool), and the OLS slope of the smoothed
profile on the zone-centre coordinate s = linspace(0,1,7) is taken via
sgd.gradient.per_gene_gradient.

Output numbers only; no interpretation.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Make the src/sgd package importable without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sgd.config import MOOR_TABLE_D, RESULTS  # noqa: E402
from sgd.gradient import per_gene_gradient, SIGMA_BINS  # noqa: E402

warnings.filterwarnings("ignore")
np.random.seed(0)

TABLE_D = MOOR_TABLE_D
RESULTS_JSON = RESULTS / "moor_crypt_results.json"

ZONE_MEAN_COLS = [
    "Crypt_mean", "V1_mean", "V2_mean", "V3_mean",
    "V4_mean", "V5_mean", "V6_mean",
]

# Canonical crypt-villus markers + literature-expected slope direction on s
# (s runs Crypt=0 -> V6 tip=1).
MARKERS_NEGATIVE = ["Lgr5", "Axin2", "Myc", "Sox9", "Olfm4", "Mki67", "Cd44"]
MARKERS_POSITIVE = ["Apoa1", "Apoa4", "Fabp2", "Ada", "Slc5a1", "Vil1",
                    "Nt5e", "Krt20"]


def load_table_d() -> pd.DataFrame:
    """Load table_D indexed on Gene name."""
    df = pd.read_csv(TABLE_D, sep="\t")
    df = df.set_index("Gene name")
    return df


def filter_genes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Keep genes where (mean of the 7 zone-means) > 25th percentile of that
    quantity across all genes, AND qval < 0.05.
    """
    n_total = df.shape[0]
    zone_means = df[ZONE_MEAN_COLS].to_numpy(dtype=float)
    per_gene_mean = zone_means.mean(axis=1)
    p25 = np.percentile(per_gene_mean, 25)

    expr_mask = per_gene_mean > p25
    qval = df["qval"].to_numpy(dtype=float)
    qval_mask = qval < 0.05  # NaN qval -> False
    keep_mask = expr_mask & qval_mask

    kept = df.loc[keep_mask].copy()
    meta = {
        "n_total_genes": int(n_total),
        "expression_quantity": "mean of the 7 zone-mean columns per gene",
        "p25_of_expression_quantity": float(p25),
        "n_pass_expression_filter": int(expr_mask.sum()),
        "n_pass_qval_filter": int(qval_mask.sum()),
        "n_kept": int(keep_mask.sum()),
    }
    return kept, meta


def smoothed_profiles(df: pd.DataFrame) -> np.ndarray:
    """
    Return (7 zones, n_genes) matrix: per gene, the 7 zone-means -> log1p ->
    Gaussian-smoothed across zones with sigma=SIGMA_BINS (0.8). This matches
    the smoothing call inside quantile_bin_per_donor_pool:
    scipy.ndimage.gaussian_filter1d(col, sigma=sigma).
    """
    zone_means = df[ZONE_MEAN_COLS].to_numpy(dtype=float)  # (n_genes, 7)
    logged = np.log1p(zone_means)                          # (n_genes, 7)
    n_genes = logged.shape[0]
    Gs = np.zeros((7, n_genes))
    for g in range(n_genes):
        Gs[:, g] = gaussian_filter1d(logged[g, :], sigma=SIGMA_BINS)
    return Gs


def main():
    print(f"[A1] Loading {TABLE_D}")
    df = load_table_d()
    print(f"[A1]   table_D: {df.shape[0]} genes x {df.shape[1]} columns")

    kept, filt_meta = filter_genes(df)
    print(f"[A1]   gene filter: kept {filt_meta['n_kept']} genes "
          f"(p25={filt_meta['p25_of_expression_quantity']:.6g})")

    # s coordinate: zone centres, Crypt=0 .. V6 tip=1.
    s = np.linspace(0, 1, 7)

    # Slopes on the full filtered gene set (reported background population).
    Gs_kept = smoothed_profiles(kept)  # (7, n_kept)
    grad_kept = per_gene_gradient(Gs_kept, s)
    n_kept_with_negative_slope = int((grad_kept["slope"] < 0).sum())
    n_kept_with_positive_slope = int((grad_kept["slope"] >= 0).sum())

    # Marker recovery. The 15 markers are a fixed evaluation set: every marker
    # present in table_D has a 7-zone-mean profile, so its OLS slope is
    # computed directly from table_D regardless of whether it passed the
    # gene filter. "passed_gene_filter" is recorded as metadata. A marker is
    # only "not found" if genuinely absent from table_D.
    all_markers = MARKERS_NEGATIVE + MARKERS_POSITIVE
    markers_in_d = [m for m in all_markers if m in df.index]
    marker_df = df.loc[markers_in_d]
    Gs_markers = smoothed_profiles(marker_df)  # (7, n_markers_in_d)
    grad_markers = per_gene_gradient(Gs_markers, s)
    marker_slope = dict(zip(markers_in_d, grad_markers["slope"]))

    marker_rows = []
    n_found = 0
    n_correct = 0
    misses = []
    for gene in all_markers:
        expected = "negative" if gene in MARKERS_NEGATIVE else "positive"
        present_in_table_d = gene in df.index
        if not present_in_table_d:
            marker_rows.append({
                "gene": gene,
                "expected_direction": expected,
                "in_table_d": False,
                "passed_gene_filter": False,
                "slope": None,
                "observed_direction": None,
                "sign_match": "not_found",
            })
            continue
        n_found += 1
        slope = float(marker_slope[gene])
        observed = "negative" if slope < 0 else "positive"
        match = (observed == expected)
        if match:
            n_correct += 1
        else:
            misses.append(gene)
        marker_rows.append({
            "gene": gene,
            "expected_direction": expected,
            "in_table_d": True,
            "passed_gene_filter": bool(gene in kept.index),
            "slope": slope,
            "observed_direction": observed,
            "sign_match": "correct" if match else "incorrect",
        })

    print(f"[A1]   markers: {n_correct}/{len(all_markers)} correct  "
          f"(n_found in table_D = {n_found})")
    if misses:
        print(f"[A1]   misses: {', '.join(misses)}")

    analysis1 = {
        "data_source": str(TABLE_D),
        "n_zones": 7,
        "zone_order": ["Crypt", "V1", "V2", "V3", "V4", "V5", "V6"],
        "zone_mean_columns": ZONE_MEAN_COLS,
        "s_coordinate": [float(x) for x in s],
        "smoothing": {
            "function": "scipy.ndimage.gaussian_filter1d",
            "sigma": float(SIGMA_BINS),
            "transform_before_smoothing": "log1p of zone-mean",
        },
        "slope_statistic": "OLS slope of smoothed log1p profile on s "
                           "(sgd.gradient.per_gene_gradient['slope'])",
        "gene_filter": filt_meta,
        "filtered_population_slope_signs": {
            "n_negative_slope": n_kept_with_negative_slope,
            "n_positive_slope": n_kept_with_positive_slope,
        },
        "n_markers_total": len(MARKERS_NEGATIVE + MARKERS_POSITIVE),
        "n_markers_found": n_found,
        "n_markers_correct": n_correct,
        "marker_score_string": f"{n_correct}/{len(MARKERS_NEGATIVE + MARKERS_POSITIVE)}",
        "marker_score_string_found": f"{n_correct}/{n_found}",
        "marker_misses": misses,
        "markers": marker_rows,
        "marker_lists": {
            "expected_negative": MARKERS_NEGATIVE,
            "expected_positive": MARKERS_POSITIVE,
        },
    }

    # Merge into the shared results JSON (preserve other analyses if present).
    RESULTS.mkdir(parents=True, exist_ok=True)
    if RESULTS_JSON.exists():
        results = json.loads(RESULTS_JSON.read_text())
    else:
        results = {}
    results["analysis1_marker_recovery"] = analysis1
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"[A1] Wrote analysis1 block to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
