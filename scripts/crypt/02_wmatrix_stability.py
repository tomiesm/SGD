"""
Moor 2018 intestinal crypt-villus — Analysis 2: W non-identifiability on
crypt-villus zonation data.

From the Analysis-1 filtered gene set, selects the top 20 spatially variable
genes (variance of the log1p zone-mean profile across the 7 zones), builds the
7-zone x 20-gene matrix G (log1p + sigma=0.8 smoothing), runs PCA, fits a
ridge W with leave-one-bin-out CV for lambda, block-bootstraps W stability,
and reports the observed block-bootstrap variability. The former gene-label
permutation comparison was removed because permuting response-gene columns only
permutes W rows and cannot define a null for an aggregate entry fraction.

Uses the paper's pipeline functions: sgd.wmatrix.{select_ridge_lambda_loocv,
fit_W_ridge, effective_rank, block_bootstrap_W} and
sgd.gradient.per_gene_gradient. The W diagnostics mirror those of the liver
W-stability step, 09_wmatrix_stability.py.

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
from sgd.wmatrix import (  # noqa: E402
    block_bootstrap_W,
    effective_rank,
    fit_W_ridge,
    select_ridge_lambda_loocv,
)

warnings.filterwarnings("ignore")
np.random.seed(0)

TABLE_D = MOOR_TABLE_D
RESULTS_JSON = RESULTS / "moor_crypt_results.json"

ZONE_MEAN_COLS = [
    "Crypt_mean", "V1_mean", "V2_mean", "V3_mean",
    "V4_mean", "V5_mean", "V6_mean",
]
N_TOP_GENES = 20
N_BOOT = 200             # empirical block-bootstrap resamples
BLOCK_SIZE = 1           # 7 zones; block size 1
Z_STABLE_THRESHOLD = 1.96


def load_filtered() -> pd.DataFrame:
    """Re-apply the Analysis-1 gene filter and return the kept rows."""
    df = pd.read_csv(TABLE_D, sep="\t").set_index("Gene name")
    zone_means = df[ZONE_MEAN_COLS].to_numpy(dtype=float)
    per_gene_mean = zone_means.mean(axis=1)
    p25 = np.percentile(per_gene_mean, 25)
    expr_mask = per_gene_mean > p25
    qval_mask = df["qval"].to_numpy(dtype=float) < 0.05
    return df.loc[expr_mask & qval_mask].copy()


def log1p_profiles(df: pd.DataFrame) -> np.ndarray:
    """(n_genes, 7) log1p zone-mean profiles (no smoothing)."""
    return np.log1p(df[ZONE_MEAN_COLS].to_numpy(dtype=float))


def smooth_zone_axis(profiles_genes_by_zone: np.ndarray) -> np.ndarray:
    """
    Gaussian-smooth across the 7 zones, sigma=SIGMA_BINS. Input (n_genes, 7);
    returns (7, n_genes) (zones along axis 0, ready for the W primitives).
    """
    n_genes = profiles_genes_by_zone.shape[0]
    Gs = np.zeros((7, n_genes))
    for g in range(n_genes):
        Gs[:, g] = gaussian_filter1d(profiles_genes_by_zone[g, :],
                                     sigma=SIGMA_BINS)
    return Gs


def stable_fraction(W_mean: np.ndarray, W_std: np.ndarray) -> float:
    """
    Fraction of W entries with |z| > 1.96, z = |W_mean| / (W_std + 1e-12).
    Matches the |z| formula used in 09_wmatrix_stability.py (the paper's
    phenomenological-stability diagnostic).
    """
    z = np.abs(W_mean) / (W_std + 1e-12)
    return float((z > Z_STABLE_THRESHOLD).sum() / max(1, z.size))


def main():
    print(f"[A2] Loading {TABLE_D}")
    kept = load_filtered()
    print(f"[A2]   filtered gene set: {kept.shape[0]} genes")

    # --- Top 20 spatially variable genes -----------------------------------
    log_profiles = log1p_profiles(kept)                # (n_kept, 7)
    profile_var = log_profiles.var(axis=1)             # variance across zones
    order = np.argsort(profile_var)[::-1]
    top_idx = order[:N_TOP_GENES]
    top_genes = kept.index.to_numpy()[top_idx].astype(str).tolist()
    top_var = profile_var[top_idx]
    print(f"[A2]   top {N_TOP_GENES} variable genes selected")

    # --- Build G = 7 zones x 20 genes (smoothed log1p) ---------------------
    G = smooth_zone_axis(log_profiles[top_idx, :])     # (7, 20)
    print(f"[A2]   G shape: {G.shape}  (zones x genes)")

    s = np.linspace(0, 1, 7)

    # --- PCA of G (centre per gene) ----------------------------------------
    G_centered = G - G.mean(axis=0, keepdims=True)
    U, sv, Vt = np.linalg.svd(G_centered, full_matrices=False)
    var = sv ** 2
    evr = var / var.sum()
    cev = np.cumsum(evr)

    def n_components_for(thr: float):
        idx = int(np.argmax(cev >= thr))
        if cev[idx] < thr:
            return None
        return idx + 1

    n_90 = n_components_for(0.90)
    n_95 = n_components_for(0.95)
    n_99 = n_components_for(0.99)
    print(f"[A2]   PCA: singular values = {np.round(sv, 4)}")
    print(f"[A2]   N_eff: 90%={n_90}, 95%={n_95}, 99%={n_99}")

    # --- Ridge W with leave-one-bin-out CV ---------------------------------
    # The W primitives regress per-gene dg/ds (central-difference gradient)
    # on G. per_gene_gradient supplies dgds_per_bin (7 zones x 20 genes).
    grad = per_gene_gradient(G, s)
    dgds = grad["dgds_per_bin"]                         # (7, 20)
    print(f"[A2]   dgds (regression target) shape: {dgds.shape}")

    chosen_lambda, mse_grid = select_ridge_lambda_loocv(G, dgds)
    print(f"[A2]   chosen ridge lambda (leave-one-bin-out CV) = {chosen_lambda}")

    eff_rank = effective_rank(G)
    print(f"[A2]   effective rank of design matrix G = {eff_rank}")

    W, b = fit_W_ridge(G, dgds, chosen_lambda)
    print(f"[A2]   W shape: {W.shape}")

    # --- Block bootstrap, block size 1 -------------------------------------
    W_mean, W_std = block_bootstrap_W(
        G, dgds, chosen_lambda, n_boot=N_BOOT, block_size=BLOCK_SIZE,
        rng_seed=0,
    )
    empirical_stable = stable_fraction(W_mean, W_std)
    print(f"[A2]   block-bootstrap empirical stable fraction "
          f"(|z|>{Z_STABLE_THRESHOLD}) = {empirical_stable:.6g}")

    analysis2 = {
        "data_source": str(TABLE_D),
        "n_zones": 7,
        "n_filtered_genes": int(kept.shape[0]),
        "n_top_genes": N_TOP_GENES,
        "top_gene_selection": "variance of log1p zone-mean profile across 7 "
                              "zones, descending",
        "top_genes": top_genes,
        "top_gene_profile_variance": [float(x) for x in top_var],
        "G_shape": [int(G.shape[0]), int(G.shape[1])],
        "smoothing": {
            "function": "scipy.ndimage.gaussian_filter1d",
            "sigma": float(SIGMA_BINS),
            "transform_before_smoothing": "log1p of zone-mean",
        },
        "pca": {
            "centering": "per-gene (column) zero-mean before SVD",
            "singular_values": [float(x) for x in sv],
            "explained_variance_ratio": [float(x) for x in evr],
            "cumulative_explained_variance": [float(x) for x in cev],
            "N_eff_90pct": n_90,
            "N_eff_95pct": n_95,
            "N_eff_99pct": n_99,
        },
        "ridge_W": {
            "regression_target": "per-gene central-difference dg/ds "
                                  "(per_gene_gradient['dgds_per_bin'])",
            "cv": "leave-one-bin-out (LeaveOneOut over the 7 zones)",
            "lambda_grid": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
            "mse_per_lambda": [float(x) for x in mse_grid],
            "chosen_lambda": float(chosen_lambda),
            "effective_rank_of_G": int(eff_rank),
            "effective_rank_tol": 1e-3,
            "W_shape": [int(W.shape[0]), int(W.shape[1])],
        },
        "block_bootstrap": {
            "n_boot": N_BOOT,
            "block_size": BLOCK_SIZE,
            "rng_seed": 0,
            "z_threshold": Z_STABLE_THRESHOLD,
            "z_formula": "|W_mean| / (W_std + 1e-12), as in "
                         "09_wmatrix_stability.py",
            "empirical_stable_fraction": empirical_stable,
        },
        "n7_adaptations": {
            "block_size": "set to 1 (one zone per block); the 7 zones are "
                          "treated as 7 independent blocks. block_bootstrap_W "
                          "default estimates block size from autocorrelation; "
                          "here it is passed explicitly.",
            "cv_folds": "LeaveOneOut over the 7 zones = 7 folds, trains on 6 "
                        "zones per fold. select_ridge_lambda_loocv runs "
                        "unmodified at N=7; no fold-count change needed.",
            "functions_ran_unmodified": True,
            "no_internal_function_edits": "fit_W_ridge, "
                                          "select_ridge_lambda_loocv, "
                                          "block_bootstrap_W and "
                                          "effective_rank all ran at N=7 "
                                          "zones without modification; only "
                                          "call-site arguments (block_size=1) "
                                          "were set.",
        },
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    if RESULTS_JSON.exists():
        results = json.loads(RESULTS_JSON.read_text())
    else:
        results = {}
    results["analysis2_W_nonidentifiability"] = analysis2
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"[A2] Wrote analysis2 block to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
