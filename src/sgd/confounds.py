"""Confound-diagnostic primitives: edge detection, Moran's I, donor-aware residualisation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from scipy.sparse import spmatrix


# ---------------------------------------------------------------------------
# C5 — slide-edge directional confound score
# ---------------------------------------------------------------------------

def gradient_direction(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """OLS slope vector of ``values`` on (x, y); returns unit vector."""
    X = np.column_stack([np.ones(len(coords)), coords])
    beta, *_ = np.linalg.lstsq(X, values, rcond=None)
    g = beta[1:]
    n = np.linalg.norm(g)
    return g / n if n > 1e-12 else np.zeros_like(g)


def morans_i_knn(values: np.ndarray, coords: np.ndarray, k: int = 6) -> float:
    """
    Moran's I of ``values`` on a k-NN spatial graph. Binary weights,
    self excluded. Suited to the Visium hex lattice (k=6 default).
    """
    n = len(values)
    if n < k + 2:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]  # drop self
    rows = np.repeat(np.arange(n), k)
    cols = idx.flatten()
    W = csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(n, n))
    Wsum = W.sum()
    z = values - values.mean()
    num = (z * np.asarray(W.dot(z)).flatten()).sum()
    denom = (z * z).sum()
    if denom < 1e-12 or Wsum < 1e-12:
        return float("nan")
    return float((n / Wsum) * (num / denom))


def c5_score(
    adata_sample: ad.AnnData,
    s_col: str = "s_preliminary",
    k: int = 6,
    n_perm: int = 200,
    rng_seed: int = 0,
) -> dict:
    """
    C5 directional-edge confound score for one sample.

    score = |cos(angle(grad(log_UMI), grad(s)))| * |Moran_I(log_UMI)|

    Returns dict with score, components, and a permutation p-value
    (shuffling spot positions). Trigger: score > 0.30 AND p < 0.05.
    """
    coords = np.asarray(adata_sample.obsm["spatial"])
    if "counts" in adata_sample.layers:
        counts = adata_sample.layers["counts"]
    else:
        counts = adata_sample.X
    if issparse(counts):
        totals = np.asarray(counts.sum(axis=1)).flatten()
    else:
        totals = counts.sum(axis=1)
    log_umi = np.log1p(totals)
    s = adata_sample.obs[s_col].to_numpy()
    keep = ~np.isnan(s) & ~np.isnan(log_umi)
    if keep.sum() < k + 2:
        return {"score": float("nan"), "p": float("nan"),
                "r_dir": float("nan"), "I_local": float("nan"),
                "n": int(keep.sum())}
    coords_k = coords[keep]
    log_umi_k = log_umi[keep]
    s_k = s[keep]

    g_umi = gradient_direction(log_umi_k, coords_k)
    g_s = gradient_direction(s_k, coords_k)
    r_dir = float(abs(np.dot(g_umi, g_s)))
    I_local = morans_i_knn(log_umi_k, coords_k, k=k)
    score = r_dir * abs(I_local) if not np.isnan(I_local) else float("nan")

    # Permutation null: shuffle coords; recompute score.
    rng = np.random.RandomState(rng_seed)
    null = np.empty(n_perm)
    for j in range(n_perm):
        perm = rng.permutation(coords_k)
        gu = gradient_direction(log_umi_k, perm)
        gs = gradient_direction(s_k, perm)
        r_d = abs(np.dot(gu, gs))
        Il = morans_i_knn(log_umi_k, perm, k=k)
        null[j] = r_d * abs(Il) if not np.isnan(Il) else 0.0
    p = float((null >= score).mean()) if not np.isnan(score) else float("nan")
    return {
        "score": float(score) if not np.isnan(score) else float("nan"),
        "p": p,
        "r_dir": r_dir,
        "I_local": float(I_local) if not np.isnan(I_local) else float("nan"),
        "n": int(keep.sum()),
    }


# ---------------------------------------------------------------------------
# B7 — donor-aware residualisation
# ---------------------------------------------------------------------------

def donor_aware_residualize(
    X: np.ndarray | spmatrix,
    donor_ids: np.ndarray,
    log_umi: np.ndarray,
    model: str = "A",
) -> tuple[np.ndarray, dict]:
    """
    Per-gene residualisation against donor + log(UMI).

    Model A (default):  g = α_donor + β · log_umi + ε
    Model B:            g = α_donor + β_donor · log_umi + ε

    Returns the residual matrix (cells × genes) and a per-gene fit-quality
    dict (R² per gene under the chosen model). Both fits use a single
    design matrix per model; per-gene OLS is one numpy lstsq call broadcast
    over the gene axis.
    """
    if issparse(X):
        X = X.toarray()
    if model not in ("A", "B"):
        raise ValueError(model)
    donors = pd.Categorical(donor_ids)
    donor_oh = pd.get_dummies(donors, drop_first=False).to_numpy().astype(float)

    if model == "A":
        # Design: [donor_dummies, log_umi]  — drop_first=True to avoid
        # collinearity with intercept (subsume intercept in the dummies).
        X_design = np.column_stack([donor_oh, log_umi])
    else:
        # Design: [donor_dummies, donor_oh * log_umi]
        donor_log_umi = donor_oh * log_umi[:, None]
        X_design = np.column_stack([donor_oh, donor_log_umi])

    # Vectorised OLS: (XᵀX)⁻¹ Xᵀ Y for each gene column simultaneously.
    XtX = X_design.T @ X_design
    XtY = X_design.T @ X
    try:
        coef = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        coef, *_ = np.linalg.lstsq(X_design, X, rcond=None)
    fitted = X_design @ coef
    resid = X - fitted

    # Per-gene R² for model selection.
    ss_res = (resid ** 2).sum(axis=0)
    ss_tot = ((X - X.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return resid.astype(np.float32), {"r2": r2.astype(np.float32), "model": model}


def select_donor_aware_model(
    X: np.ndarray | spmatrix,
    donor_ids: np.ndarray,
    log_umi: np.ndarray,
    headline_idx: np.ndarray,
    delta_r2_threshold: float = 0.02,
) -> str:
    """
    Pick model A or B (B7 spec). Upgrade to B if the average ΔR² over the
    headline gene set exceeds `delta_r2_threshold`.
    """
    _, info_A = donor_aware_residualize(X, donor_ids, log_umi, model="A")
    _, info_B = donor_aware_residualize(X, donor_ids, log_umi, model="B")
    delta = float((info_B["r2"][headline_idx] - info_A["r2"][headline_idx]).mean())
    return "B" if delta > delta_r2_threshold else "A"


# ---------------------------------------------------------------------------
# B7 — tissue-boundary edge detection (k-NN angle-gap method)
# ---------------------------------------------------------------------------

def detect_boundary_spots(coords: np.ndarray, k: int = 6,
                           gap_deg_threshold: float = 120.0) -> np.ndarray:
    """
    Boundary mask via k-NN angle-gap detection. A spot is on the boundary
    if the largest angular gap between its k-NN directions exceeds
    `gap_deg_threshold` (default 120°). Tolerates concave tissue shapes
    where convex-hull / alpha-shape methods are brittle.
    """
    n = len(coords)
    if n < k + 1:
        return np.ones(n, dtype=bool)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]  # drop self
    boundary = np.zeros(n, dtype=bool)
    gap_thr = np.deg2rad(gap_deg_threshold)
    for i in range(n):
        nbr_vecs = coords[idx[i]] - coords[i]
        angles = np.arctan2(nbr_vecs[:, 1], nbr_vecs[:, 0])
        angles_sorted = np.sort(angles)
        # Wrap-around gap.
        gaps = np.diff(np.concatenate([angles_sorted, angles_sorted[:1] + 2 * np.pi]))
        if gaps.max() > gap_thr:
            boundary[i] = True
    return boundary


def edge_spots_mask(coords: np.ndarray, edge_frac: float = 0.05) -> np.ndarray:
    """
    Spots whose distance to the nearest boundary spot is ≤
    `edge_frac × sample_diagonal`. `sample_diagonal` is the diagonal of
    the spatial bounding box of the sample.
    """
    boundary = detect_boundary_spots(coords)
    if boundary.sum() == 0:
        return np.zeros(len(coords), dtype=bool)
    bbox = coords.max(axis=0) - coords.min(axis=0)
    diag = float(np.linalg.norm(bbox))
    nn = NearestNeighbors(n_neighbors=1).fit(coords[boundary])
    dists, _ = nn.kneighbors(coords)
    return (dists[:, 0] <= edge_frac * diag)


# ---------------------------------------------------------------------------
# B7 — per-gene edge-concentration score
# ---------------------------------------------------------------------------

def per_gene_edge_concentration(
    adata_lhd: ad.AnnData,
    gene_idx: np.ndarray,
    sample_col: str = "sample_id",
    min_total_count: int = 100,
    min_detected_frac: float = 0.05,
    edge_frac: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """
    Per-gene edge-concentration score across LHD samples.

    Uses raw counts from `layers['counts']`. For each (gene, sample):
    edge_fraction = sum(counts in edge_spots) / sum(counts in all spots),
    computed only if the gene has ≥ min_total_count total counts in that
    sample AND is detected (≥1 count) in ≥ min_detected_frac of spots.
    Otherwise NaN.

    The per-gene aggregate is mean over evaluable samples of
    (edge_fraction − baseline) / baseline, where baseline = |edge|/|spots|
    per sample. NaN if fewer than 4 of the 8 LHD samples are evaluable.
    """
    if "counts" not in adata_lhd.layers:
        raise ValueError("adata_lhd.layers['counts'] required (raw counts)")
    samples = adata_lhd.obs[sample_col].astype(str).to_numpy()
    donors = sorted(np.unique(samples))
    G = len(gene_idx)
    edge_fracs = np.full((len(donors), G), np.nan)
    baselines = np.full(len(donors), np.nan)
    per_sample_meta: dict = {}

    for di, d in enumerate(donors):
        m = samples == d
        coords = np.asarray(adata_lhd[m].obsm["spatial"])
        edge = edge_spots_mask(coords, edge_frac=edge_frac)
        baselines[di] = float(edge.mean()) if len(edge) else np.nan
        per_sample_meta[d] = {
            "n_spots": int(m.sum()),
            "n_edge_spots": int(edge.sum()),
            "edge_baseline": baselines[di],
        }
        X = adata_lhd[m].layers["counts"]
        if issparse(X):
            X = X.toarray()
        X = X[:, gene_idx]
        # Per-gene total and detection.
        total = X.sum(axis=0)
        detected_frac = (X > 0).mean(axis=0)
        # Edge sum.
        edge_sum = X[edge].sum(axis=0)
        ok = (total >= min_total_count) & (detected_frac >= min_detected_frac)
        with np.errstate(invalid="ignore", divide="ignore"):
            ef = np.where(ok, edge_sum / np.maximum(total, 1), np.nan)
        edge_fracs[di] = ef

    # Per-gene aggregate, ignoring NaN.
    base_b = baselines[:, None]
    rel = (edge_fracs - base_b) / np.maximum(base_b, 1e-9)
    n_evaluable = (~np.isnan(rel)).sum(axis=0)
    score = np.where(n_evaluable >= 4, np.nanmean(rel, axis=0), np.nan)
    return score.astype(np.float32), {
        "per_sample": per_sample_meta,
        "n_evaluable_per_gene": n_evaluable,
    }


# ---------------------------------------------------------------------------
# C4 — convex-hull edge mask
# ---------------------------------------------------------------------------

def edge_mask_visium(adata_sample: ad.AnnData) -> np.ndarray:
    """
    C5 edge mask via convex hull of the spatial point cloud — replaces
    the k-NN angle-gap detector which was either too permissive (120°
    default in Stage B flagged 75-97% of spots) or too aggressive (60°
    threshold flagged 100%). Convex hull is unambiguous: spots within
    `0.05 × bounding-box-diagonal` of any hull vertex are flagged.
    """
    from scipy.spatial import ConvexHull
    coords = np.asarray(adata_sample.obsm["spatial"])
    if len(coords) < 4:
        return np.zeros(adata_sample.n_obs, dtype=bool)
    try:
        hull = ConvexHull(coords)
    except Exception:
        return np.zeros(adata_sample.n_obs, dtype=bool)
    hull_pts = coords[hull.vertices]
    bbox = coords.max(axis=0) - coords.min(axis=0)
    diag = float(np.linalg.norm(bbox))
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(hull_pts)
    dists, _ = nn.kneighbors(coords)
    return (dists[:, 0] <= 0.05 * diag)
