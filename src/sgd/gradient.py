"""Per-donor quantile binning and the per-gene spatial-gradient primitive."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import issparse

# Default Stage B knobs.
N_BINS_DEFAULT = 50
N_BINS_SWEEP = (30, 50, 100, 150)
SIGMA_BINS = 0.8
DENSITY_FLOOR_FRAC = 0.01

# Quantile-layer count for the cross-pipeline approach (a) aggregation.
N_LAYERS = 8


# ---------------------------------------------------------------------------
# Binning + per-gene gradient
# ---------------------------------------------------------------------------

def quantile_bin_per_donor_pool(
    adata: ad.AnnData,
    s_col: str,
    sample_col: str = "sample_id",
    n_bins: int = N_BINS_DEFAULT,
    sigma: float = SIGMA_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-donor quantile bin then pool to cohort-level binned matrix.

    Each donor's spots get assigned to N quantile bins of `s` (donor-
    specific edges). The per-bin mean expression is computed per donor,
    then averaged across donors that contribute spots to that bin index.
    Returns the smoothed per-bin matrix and the cohort bin centres.

    Returns
    -------
    Gs : (n_bins, n_genes) smoothed per-bin mean log-expression
    bc : (n_bins,) cohort bin centres (mean of donor-bin centres)
    cpb : (n_bins,) cell counts per bin (cohort)
    donor_bin_centres : (n_donors, n_bins) per-donor bin centres
    """
    samples = adata.obs[sample_col].astype(str).to_numpy()
    s = adata.obs[s_col].to_numpy()
    valid_mask = ~np.isnan(s)
    donors = sorted(np.unique(samples[valid_mask]))
    n_donors = len(donors)

    G_per_donor = np.full((n_donors, n_bins, adata.n_vars), np.nan)
    bc_per_donor = np.full((n_donors, n_bins), np.nan)
    cpb_per_donor = np.zeros((n_donors, n_bins), dtype=int)

    for di, d in enumerate(donors):
        m = (samples == d) & valid_mask
        s_d = s[m]
        if len(s_d) < n_bins:
            continue
        be = np.percentile(s_d, np.linspace(0, 100, n_bins + 1))
        be = np.unique(be)
        nb = len(be) - 1
        bc = 0.5 * (be[:-1] + be[1:])
        ba = np.clip(np.digitize(s_d, be) - 1, 0, nb - 1)
        X_d = adata[m].X
        if issparse(X_d):
            X_d = X_d.toarray()
        for b in range(nb):
            mask_b = ba == b
            cpb_per_donor[di, b] = mask_b.sum()
            if cpb_per_donor[di, b] > 0:
                G_per_donor[di, b] = X_d[mask_b].mean(axis=0)
                bc_per_donor[di, b] = bc[b]

    # Cohort pool: mean across donors at each bin index, ignoring NaN.
    G_pool = np.nanmean(G_per_donor, axis=0)            # (n_bins, n_genes)
    bc_pool = np.nanmean(bc_per_donor, axis=0)          # (n_bins,)
    cpb_pool = cpb_per_donor.sum(axis=0)                # (n_bins,)

    # Smooth per gene along the bin axis.
    Gs = np.zeros_like(G_pool)
    for g in range(G_pool.shape[1]):
        col = G_pool[:, g].copy()
        # Replace NaN with column mean before smoothing to avoid filter blow-up.
        if np.any(np.isnan(col)):
            col = np.where(np.isnan(col), np.nanmean(col), col)
        Gs[:, g] = gaussian_filter1d(col, sigma=sigma)
    return Gs, bc_pool, cpb_pool, bc_per_donor


def per_gene_gradient(
    Gs: np.ndarray,
    bc: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Two parallel summary statistics for each gene's spatial profile:

    * **bin_mean_dgds** — bin-wise mean of central-difference dg/ds
    * **slope** — OLS slope β from linear regression of g(s) on s

    Plus a **monotonicity** percentage per gene: fraction of bins
    agreeing on the dominant sign of dg/ds.
    """
    n_bins = Gs.shape[0]
    # Central differences.
    dgds = np.zeros_like(Gs)
    dgds[1:-1] = (Gs[2:] - Gs[:-2]) / (bc[2:] - bc[:-2])[:, None]
    dgds[0] = (Gs[1] - Gs[0]) / (bc[1] - bc[0])
    dgds[-1] = (Gs[-1] - Gs[-2]) / (bc[-1] - bc[-2])
    bin_mean = dgds.mean(axis=0)

    # Linear slope via numpy lstsq, vectorised across genes.
    bc_c = bc - bc.mean()
    denom = (bc_c ** 2).sum()
    Gs_c = Gs - Gs.mean(axis=0, keepdims=True)
    slope = (bc_c[:, None] * Gs_c).sum(axis=0) / max(denom, 1e-12)

    # Monotonicity: fraction of bins agreeing with the dominant sign.
    pos = (dgds > 0).sum(axis=0)
    neg = (dgds < 0).sum(axis=0)
    dominant = np.maximum(pos, neg) / n_bins
    return {
        "bin_mean_dgds": bin_mean,
        "slope": slope,
        "monotonicity": dominant,
        "dgds_per_bin": dgds,
    }


# ---------------------------------------------------------------------------
# Per-gene quantile-layer means (cross-pipeline approach (a))
# ---------------------------------------------------------------------------

def per_gene_layer_means(adata: 'AnnData', s_col: str = "s",
                         n_layers: int = N_LAYERS) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Per-donor 8-quantile binning along obs.s, then per-layer mean expression
    pooled across donors (mean of per-donor means at each layer).
    Returns (layer_means_df indexed by gene with cols Layer_1..Layer_8,
             cells_per_layer).
    """
    samples = adata.obs["sample_id"].astype(str).to_numpy()
    s = adata.obs[s_col].to_numpy()
    valid = ~np.isnan(s)
    donors = sorted(np.unique(samples[valid]))
    n_donors = len(donors)
    means_per_donor = np.full((n_donors, n_layers, adata.n_vars), np.nan)
    counts = np.zeros(n_layers, dtype=int)

    for di, d in enumerate(donors):
        m = (samples == d) & valid
        s_d = s[m]
        if len(s_d) < n_layers:
            continue
        be = np.percentile(s_d, np.linspace(0, 100, n_layers + 1))
        be = np.unique(be)
        nb = len(be) - 1
        ba = np.clip(np.digitize(s_d, be) - 1, 0, nb - 1)
        X_d = adata[m].X
        if issparse(X_d):
            X_d = X_d.toarray()
        for b in range(nb):
            mask_b = ba == b
            counts[b] += int(mask_b.sum())
            if mask_b.sum() > 0:
                means_per_donor[di, b] = X_d[mask_b].mean(axis=0)

    pooled = np.nanmean(means_per_donor, axis=0)  # (n_layers, n_genes)
    df = pd.DataFrame(pooled.T, index=adata.var_names,
                      columns=[f"Layer_{i+1}" for i in range(n_layers)])
    return df, counts
