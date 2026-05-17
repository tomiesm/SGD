"""Gene-panel masks, liver-zonation marker constants, and cohort spot masks."""

from __future__ import annotations

from typing import Iterable

import anndata as ad
import numpy as np
from scipy.sparse import issparse

from sgd.config import LHD_SAMPLES

# Markers (§5.1).
PORTAL_MARKERS = ("SDS", "ASS1", "CYP2A6", "HAL", "CPS1")
CENTRAL_MARKERS = ("CYP2E1", "CYP1A2", "GLUL", "CYP3A4")
KNOWN_LIVER_DIRECTION = {  # +1 central, -1 portal — all 9 used downstream
    "SDS": -1, "ASS1": -1, "CYP2A6": -1, "HAL": -1, "CPS1": -1,
    "CYP2E1": +1, "CYP1A2": +1, "GLUL": +1, "CYP3A4": +1,
}

# Markers used for fibrotic-spot detection (§4.5 derivation path).
COLLAGEN_MARKERS = ("COL1A1", "COL3A1", "ACTA2")
HEP_MARKERS = ("ALB", "HNF4A", "TTR")

# CPS1 excluded from headline per Stage B addendum forward-flag.
HEADLINE_MARKERS = ("SDS", "ASS1", "CYP2A6", "HAL",
                    "CYP2E1", "CYP1A2", "GLUL", "CYP3A4")


# ---------------------------------------------------------------------------
# Per-cell gene score
# ---------------------------------------------------------------------------

def _gene_score(adata: ad.AnnData, genes: Iterable[str]) -> np.ndarray:
    """Per-cell mean log-expression of ``genes`` present in ``adata``."""
    present = [g for g in genes if g in adata.var_names]
    if not present:
        return np.full(adata.n_obs, np.nan)
    X = adata[:, present].X
    if issparse(X):
        X = X.toarray()
    return np.asarray(X.mean(axis=1)).flatten()


# ---------------------------------------------------------------------------
# Analysis gene panel
# ---------------------------------------------------------------------------

def analysis_gene_panel(
    adata: ad.AnnData, mode: str = "strict",
    sample_col: str = "sample_id",
    relaxed_min_donors: int = 6,
    relaxed_min_count: int = 1,
) -> np.ndarray:
    """
    Boolean per-gene mask for the analysis panel.

    ``strict``  → ``var.panel_intersection`` from Stage A (top-2000 HVGs
                  intersected across all 8 LHD donors). The default for
                  the §13 gate.
    ``relaxed`` → genes detected (≥1 raw count) in ≥ ``relaxed_min_donors``
                  of the LHD donors. Used as a secondary consistency
                  check when 66 genes is too small.
    """
    if mode == "strict":
        if "panel_intersection" not in adata.var.columns:
            raise RuntimeError("var.panel_intersection missing — re-run Stage A A2.")
        return adata.var["panel_intersection"].astype(bool).to_numpy()
    if mode != "relaxed":
        raise ValueError(f"unknown mode: {mode}")

    counts = adata.layers.get("counts", adata.X)
    samples = adata.obs[sample_col].astype(str).to_numpy()
    detection = np.zeros(adata.n_vars, dtype=int)
    for d in LHD_SAMPLES:
        m = samples == d
        if m.sum() == 0:
            continue
        sub = counts[m]
        if issparse(sub):
            present = (sub >= relaxed_min_count).getnnz(axis=0) > 0
        else:
            present = (sub >= relaxed_min_count).any(axis=0)
        detection += np.asarray(present).flatten().astype(int)
    return detection >= relaxed_min_donors


# ---------------------------------------------------------------------------
# Cohort spot masks
# ---------------------------------------------------------------------------

def lhd_mask(adata: ad.AnnData, sample_col: str = "sample_id") -> np.ndarray:
    return adata.obs[sample_col].astype(str).isin(LHD_SAMPLES).to_numpy()


def cohort_mask(adata: ad.AnnData, samples: Iterable[str],
                 sample_col: str = "sample_id") -> np.ndarray:
    return adata.obs[sample_col].astype(str).isin(list(samples)).to_numpy()


def lhd_analysis_mask(adata: ad.AnnData, sample_col: str = "sample_id",
                       exclude_fibrotic: bool = True) -> np.ndarray:
    """
    LHD spot mask AND-ed with the §4.5 fibrotic-spot exclusion. Used by
    every gradient-bearing Stage B script. The fibrotic mask is computed
    in Stage A's A6.5 and stored in ``obs.fibrotic_spot``; if absent,
    no fibrotic exclusion is applied (with a runtime warning).
    """
    m = lhd_mask(adata, sample_col)
    if exclude_fibrotic:
        if "fibrotic_spot" in adata.obs.columns:
            fib = adata.obs["fibrotic_spot"].astype(bool).to_numpy()
            m = m & ~fib
        else:
            print("[utils] WARNING: obs.fibrotic_spot missing — Stage A A6.5 not run? "
                  "Proceeding without fibrotic exclusion.")
    return m
