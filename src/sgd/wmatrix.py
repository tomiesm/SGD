"""W-matrix ridge fit, effective rank, block bootstrap, donor-split stability, HD helpers."""

from __future__ import annotations

from typing import Iterable  # noqa: F401

import anndata as ad
import numpy as np
from scipy.sparse import issparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

from sgd.config import LHD_SAMPLES
from sgd.confounds import donor_aware_residualize
from sgd.gradient import (SIGMA_BINS, per_gene_gradient,
                          quantile_bin_per_donor_pool)

# Bin-count grid for §8 phenomenological stability.
N_BINS_STANDARD = (30, 50, 100, 150)
N_BINS_HD = (200, 500)

# Ridge regularisation grid.
RIDGE_LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)

# Default per-stage bin count (re-exported from gradient knobs for parity
# with the legacy Stage C utils surface).
N_BINS_DEFAULT = 50


# ---------------------------------------------------------------------------
# Ridge LOOCV W fitting
# ---------------------------------------------------------------------------

def select_ridge_lambda_loocv(G: np.ndarray, dgds: np.ndarray,
                              lambdas: Iterable[float] = RIDGE_LAMBDA_GRID
                              ) -> tuple[float, np.ndarray]:
    """
    Select λ minimising mean MSE across genes via leave-one-bin-out CV.

    Returns (best_lambda, mean_mse_per_lambda).
    """
    n_bins = G.shape[0]
    n_genes = dgds.shape[1]
    lambdas = list(lambdas)
    mse = np.zeros(len(lambdas))
    loo = LeaveOneOut()
    for li, lam in enumerate(lambdas):
        per_gene_mse = np.zeros(n_genes)
        for gi in range(n_genes):
            preds = np.zeros(n_bins)
            for train_idx, test_idx in loo.split(G):
                m = Ridge(alpha=lam, fit_intercept=True)
                m.fit(G[train_idx], dgds[train_idx, gi])
                preds[test_idx[0]] = m.predict(G[test_idx])[0]
            per_gene_mse[gi] = np.mean((preds - dgds[:, gi]) ** 2)
        mse[li] = per_gene_mse.mean()
    best = lambdas[int(np.argmin(mse))]
    return float(best), mse


def fit_W_ridge(G: np.ndarray, dgds: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene Ridge fit. Returns (W [k×k], b [k])."""
    n_genes = dgds.shape[1]
    W = np.zeros((n_genes, G.shape[1]))
    b = np.zeros(n_genes)
    for gi in range(n_genes):
        m = Ridge(alpha=lam, fit_intercept=True)
        m.fit(G, dgds[:, gi])
        W[gi] = m.coef_
        b[gi] = m.intercept_
    return W, b


# ---------------------------------------------------------------------------
# Effective rank, nullspace
# ---------------------------------------------------------------------------

def effective_rank(G: np.ndarray, tol: float = 1e-3) -> int:
    """
    Effective rank of (n_bins × p) design matrix at Frobenius-norm-ratio tolerance.

    Counts singular values above `tol × sigma_max`.
    """
    if G.size == 0:
        return 0
    sigmas = np.linalg.svd(G, compute_uv=False)
    if sigmas.max() <= 0:
        return 0
    return int((sigmas > tol * sigmas.max()).sum())


# ---------------------------------------------------------------------------
# Block bootstrap of W
# ---------------------------------------------------------------------------

def autocorr_length(x: np.ndarray, max_lag: int = 20) -> int:
    """Crude autocorrelation length: smallest lag where r drops below 0.2."""
    n = len(x)
    if n < 4:
        return 1
    x_c = x - x.mean()
    var = (x_c ** 2).sum()
    if var == 0:
        return 1
    for lag in range(1, min(max_lag, n - 1)):
        ac = (x_c[:-lag] * x_c[lag:]).sum() / var
        if ac < 0.2:
            return lag
    return min(max_lag, n - 1)


def block_bootstrap_W(G: np.ndarray, dgds: np.ndarray, lam: float,
                       n_boot: int = 200, block_size: int | None = None,
                       rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Block-bootstrap of W, holding λ fixed at the full-data CV pick.

    Returns (W_mean [k×k], W_std [k×k]) over n_boot resamples.
    """
    n_bins = G.shape[0]
    n_genes = dgds.shape[1]
    if block_size is None:
        # Estimate per-gene autocorr length, take median × 1.5.
        lags = [autocorr_length(dgds[:, gi]) for gi in range(min(n_genes, 50))]
        block_size = max(1, int(np.median(lags) * 1.5))
    rng = np.random.RandomState(rng_seed)
    W_boots = np.zeros((n_boot, n_genes, G.shape[1]))
    n_blocks = (n_bins + block_size - 1) // block_size
    for b in range(n_boot):
        starts = rng.randint(0, max(1, n_bins - block_size + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_size, n_bins)) for s in starts])
        idx = idx[:n_bins]
        W_b, _ = fit_W_ridge(G[idx], dgds[idx], lam)
        W_boots[b] = W_b
    return W_boots.mean(axis=0), W_boots.std(axis=0)


# ---------------------------------------------------------------------------
# Donor-split stability
# ---------------------------------------------------------------------------

def donor_split_W_distance(adata: ad.AnnData, donor_col: str, s_col: str,
                            panel_mask: np.ndarray, n_bins: int,
                            n_splits: int = 1000, rng_seed: int = 0,
                            ridge_lambda: float | None = None
                            ) -> dict:
    """
    Random 4-vs-4 LHD donor splits. For each split, fit W on one half and
    on the other; report Frobenius distance ‖W₁−W₂‖/‖W₁‖.
    """
    donors = sorted([d for d in adata.obs[donor_col].astype(str).unique()
                     if d in LHD_SAMPLES])
    if len(donors) < 8:
        return {"status": "fewer_than_8_donors", "n_donors": len(donors)}
    rng = np.random.RandomState(rng_seed)
    distances = []
    chosen_lambda: float | None = ridge_lambda
    for _ in range(n_splits):
        perm = rng.permutation(donors)
        half_a = list(perm[:len(perm) // 2])
        half_b = list(perm[len(perm) // 2:])
        Wa, Wb = None, None
        for half, slot in ((half_a, "a"), (half_b, "b")):
            sub = adata[adata.obs[donor_col].astype(str).isin(half)].copy()
            sub_panel = sub[:, panel_mask].copy()
            Gs, bc, _, _ = quantile_bin_per_donor_pool(
                sub_panel, s_col=s_col, n_bins=n_bins, sigma=SIGMA_BINS)
            if np.isnan(bc).all() or len(bc) < 3:
                continue
            grad = per_gene_gradient(Gs, bc)
            if chosen_lambda is None:
                chosen_lambda, _ = select_ridge_lambda_loocv(Gs, grad["dgds_per_bin"])
            assert chosen_lambda is not None
            W, _ = fit_W_ridge(Gs, grad["dgds_per_bin"], chosen_lambda)
            if slot == "a":
                Wa = W
            else:
                Wb = W
        if Wa is None or Wb is None:
            continue
        dist = float(np.linalg.norm(Wa - Wb) / max(1e-12, np.linalg.norm(Wa)))
        distances.append(dist)
    distances = np.array(distances)
    return {
        "n_splits_evaluated": int(len(distances)),
        "mean_frobenius_ratio": float(distances.mean()) if len(distances) else float("nan"),
        "median_frobenius_ratio": float(np.median(distances)) if len(distances) else float("nan"),
        "p25": float(np.percentile(distances, 25)) if len(distances) else float("nan"),
        "p75": float(np.percentile(distances, 75)) if len(distances) else float("nan"),
        "ridge_lambda_used": chosen_lambda,
    }


# ---------------------------------------------------------------------------
# HD-side helpers
# ---------------------------------------------------------------------------

def hd_path_a_donor_subset(adata_hd: ad.AnnData, sample: str,
                           sample_col: str = "sample_id") -> ad.AnnData:
    """Return path-A pre-segmented subset for one HD donor (M2 or M6)."""
    sub = adata_hd[(adata_hd.obs.get("platform_path", "").astype(str) == "hd_segmented")
                    & (adata_hd.obs[sample_col].astype(str) == sample)].copy()
    return sub


def per_cell_log_count_residuals(X: np.ndarray, total_counts: np.ndarray
                                  ) -> np.ndarray:
    """
    Per-cell log(total_counts) residualisation analogue of B7 model A,
    suitable for HD path A (single-cell, no donor effect within sample).

    For each gene, fit y_i = α + β·log(total_counts)_i + ε_i; return residuals.
    """
    if issparse(X):
        X = X.toarray()
    log_count = np.log1p(total_counts).astype(float)
    log_count = log_count - log_count.mean()
    denom = (log_count ** 2).sum()
    beta = (log_count[:, None] * X).sum(axis=0) / max(denom, 1e-12)
    fitted = log_count[:, None] * beta[None, :]
    return X - fitted


def apply_donor_aware_model_A(adata_strict: ad.AnnData,
                               donor_col: str = "donor_id"
                               ) -> ad.AnnData:
    """
    Apply Stage B model-A residualisation to a strict-panel AnnData copy.
    `g ~ donor + log(total_umi_raw) + ε`; returns a new AnnData with the
    residuals as X. Required for parity between standard-Visium W fits
    and HD per-cell log-count residualised fits.
    """
    if "total_umi_raw" not in adata_strict.obs.columns:
        raise RuntimeError(
            "obs.total_umi_raw missing — Stage A A2 didn't snapshot raw UMI before gene filter")
    log_umi = np.log1p(adata_strict.obs["total_umi_raw"].to_numpy().astype(float))
    donor_ids = adata_strict.obs[donor_col].astype(str).to_numpy()
    resid, _ = donor_aware_residualize(adata_strict.X, donor_ids, log_umi, model="A")
    out = adata_strict.copy()
    out.X = resid
    return out
