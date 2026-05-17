"""Spatial portal-central axis construction (§5.1-5.4) and the fibrotic-spot mask."""

from __future__ import annotations

from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

from sgd.config import SUPP_T2
from sgd.panels import (CENTRAL_MARKERS, COLLAGEN_MARKERS, HEP_MARKERS,
                        KNOWN_LIVER_DIRECTION, PORTAL_MARKERS, _gene_score)

# Anatomical-landmark anchors for Variant C.
BILE_DUCT_MARKERS = ("KRT7", "KRT19", "SOX9")
CENTRAL_VEIN_MARKERS = ("RSPO3", "AXIN2")

# Paper convention (verified empirically against the 9 known markers):
# Mean_Zone_1 ≈ central, Mean_Zone_8 ≈ portal. Confirmed by GLUL, CYP3A4,
# CYP1A2, CYP2E1 all peaking at Z1 (the central markers) and SDS, ASS1, HAL,
# CYP2A6 peaking at Z6-Z7 (the portal markers).
CENTRAL_LAYERS = ("Mean_Zone_1", "Mean_Zone_2")
PORTAL_LAYERS = ("Mean_Zone_7", "Mean_Zone_8")


# ---------------------------------------------------------------------------
# Preliminary axis (§5.1)
# ---------------------------------------------------------------------------

def compute_preliminary_axis(
    adata: ad.AnnData,
    sample_col: str = "sample_id",
    portal_markers: Iterable[str] = PORTAL_MARKERS,
    central_markers: Iterable[str] = CENTRAL_MARKERS,
    eps: float = 1e-9,
) -> dict:
    """
    Compute §5.1 primary + §5.2 ratio axes per sample, with §5.3
    orientation flip. Writes ``obs.s_preliminary`` and
    ``obs.s_alt_preliminary``; returns a per-sample diagnostic dict
    (orientation flipped, marker-direction agreement count).
    """
    s_prim = np.full(adata.n_obs, np.nan)
    s_alt = np.full(adata.n_obs, np.nan)
    diagnostics: dict[str, dict] = {}

    for sample in adata.obs[sample_col].unique():
        mask = (adata.obs[sample_col] == sample).to_numpy()
        sub = adata[mask]
        P = _gene_score(sub, portal_markers)
        C = _gene_score(sub, central_markers)
        if np.isnan(P).all() or np.isnan(C).all():
            diagnostics[str(sample)] = {"status": "no_markers"}
            continue

        Pz = (P - np.nanmean(P)) / (np.nanstd(P) + eps)
        Cz = (C - np.nanmean(C)) / (np.nanstd(C) + eps)
        delta = Cz - Pz
        ranks = pd.Series(delta).rank(method="average").to_numpy() - 1
        s_i = ranks / max(1, len(ranks) - 1)

        # §5.3 orientation: pericentral markers should be higher at high-s.
        cy_score = _gene_score(sub, ("CYP2E1", "CYP1A2"))
        if not np.isnan(cy_score).all():
            hi = s_i > 0.5
            lo = s_i <= 0.5
            if cy_score[hi].mean() < cy_score[lo].mean():
                s_i = 1.0 - s_i
                flipped = True
            else:
                flipped = False
        else:
            flipped = False

        # §5.2 ratio variant.
        s_alt_i = (C - P) / (np.abs(C) + np.abs(P) + eps)
        if flipped:
            s_alt_i = -s_alt_i

        s_prim[mask] = s_i
        s_alt[mask] = s_alt_i

        # Per-marker direction sanity (5/9 portal lower at hi-s; 4/9 central higher at hi-s).
        marker_check = {}
        for g, expected in KNOWN_LIVER_DIRECTION.items():
            if g not in adata.var_names:
                continue
            gx = _gene_score(sub, (g,))
            if np.isnan(gx).all():
                continue
            sign = np.sign(gx[s_i > 0.5].mean() - gx[s_i <= 0.5].mean())
            marker_check[g] = int(sign == expected)
        diagnostics[str(sample)] = {
            "status": "ok",
            "n_spots": int(mask.sum()),
            "orientation_flipped": flipped,
            "marker_agreement": marker_check,
            "marker_correct": sum(marker_check.values()),
            "marker_total": len(marker_check),
        }

    adata.obs["s_preliminary"] = s_prim
    adata.obs["s_alt_preliminary"] = s_alt
    return diagnostics


# ---------------------------------------------------------------------------
# Fibrotic-spot mask (§4.5 derivation path)
# ---------------------------------------------------------------------------

def derive_fibrotic_mask(
    adata: ad.AnnData,
    sample_col: str = "sample_id",
    z_coll_thr: float = 1.5,
    z_hep_thr: float = -0.5,
    collagen_markers: Iterable[str] = COLLAGEN_MARKERS,
    hep_markers: Iterable[str] = HEP_MARKERS,
) -> dict:
    """
    Per-sample collagen-up + hepatocyte-down mask.

    Returns per-sample fraction flagged. Writes ``obs.fibrotic_spot``.
    Spots are NOT removed; Stage B applies the mask as a filter.
    """
    flag = np.zeros(adata.n_obs, dtype=bool)
    fractions: dict[str, float] = {}
    for sample in adata.obs[sample_col].unique():
        mask = (adata.obs[sample_col] == sample).to_numpy()
        sub = adata[mask]
        coll = _gene_score(sub, collagen_markers)
        hep = _gene_score(sub, hep_markers)
        if np.isnan(coll).all() or np.isnan(hep).all():
            fractions[str(sample)] = float("nan")
            continue
        zc = (coll - np.nanmean(coll)) / (np.nanstd(coll) + 1e-9)
        zh = (hep - np.nanmean(hep)) / (np.nanstd(hep) + 1e-9)
        flagged = (zc > z_coll_thr) & (zh < z_hep_thr)
        flag[mask] = flagged
        fractions[str(sample)] = float(flagged.mean())

    adata.obs["fibrotic_spot"] = flag
    return fractions


# ---------------------------------------------------------------------------
# Score-difference axis (§5.1 / §5.2) — LHD cohort
# ---------------------------------------------------------------------------

def build_score_axis(adata_lhd: ad.AnnData, portal_genes, central_genes,
                      sample_col: str = "sample_id") -> tuple[np.ndarray, np.ndarray, dict]:
    """
    §5.1 rank-normalized score-difference axis on LHD only, per sample.
    §5.2 ratio variant computed in parallel.
    Returns (s_primary, s_alt, per-sample diagnostics).
    """
    n = adata_lhd.n_obs
    s_prim = np.full(n, np.nan)
    s_alt = np.full(n, np.nan)
    diagnostics: dict[str, dict] = {}
    samples_arr = adata_lhd.obs[sample_col].astype(str).to_numpy()

    for sample in np.unique(samples_arr):
        m = samples_arr == sample
        sub = adata_lhd[m]
        P = _gene_score(sub, portal_genes)
        C = _gene_score(sub, central_genes)
        if np.isnan(P).all() or np.isnan(C).all():
            diagnostics[sample] = {"status": "no_markers"}
            continue
        Pz = (P - np.nanmean(P)) / (np.nanstd(P) + 1e-9)
        Cz = (C - np.nanmean(C)) / (np.nanstd(C) + 1e-9)
        delta = Cz - Pz
        ranks = pd.Series(delta).rank(method="average").to_numpy() - 1
        s_i = ranks / max(1, len(ranks) - 1)

        # §5.3 orientation flip if pericentral markers are not high at high-s.
        cy = _gene_score(sub, ("CYP2E1", "CYP1A2"))
        flipped = False
        if not np.isnan(cy).all():
            hi = s_i > 0.5
            if cy[hi].mean() < cy[~hi].mean():
                s_i = 1.0 - s_i
                flipped = True

        s_alt_i = (C - P) / (np.abs(C) + np.abs(P) + 1e-9)
        if flipped:
            s_alt_i = -s_alt_i
        s_prim[m] = s_i
        s_alt[m] = s_alt_i
        diagnostics[sample] = {
            "status": "ok",
            "n_spots": int(m.sum()),
            "orientation_flipped": bool(flipped),
            "n_portal_genes_present": int(sum(1 for g in portal_genes if g in adata_lhd.var_names)),
            "n_central_genes_present": int(sum(1 for g in central_genes if g in adata_lhd.var_names)),
        }
    return s_prim, s_alt, diagnostics


# ---------------------------------------------------------------------------
# Anatomical-landmark axis (§5.4 Variant C)
# ---------------------------------------------------------------------------

def build_landmark_axis(adata_lhd: ad.AnnData,
                         sample_col: str = "sample_id",
                         k: int = 25,
                         high_quantile: float = 0.9) -> tuple[np.ndarray, dict]:
    """
    Variant C: KNN-distance to bile-duct anchors vs central-vein anchors,
    per sample. Anchors are spots in the top quantile of the corresponding
    marker score. Exploratory.
    """
    s_v = np.full(adata_lhd.n_obs, np.nan)
    diagnostics: dict[str, dict] = {}
    samples_arr = adata_lhd.obs[sample_col].astype(str).to_numpy()

    for sample in np.unique(samples_arr):
        m = samples_arr == sample
        sub = adata_lhd[m]
        coords = np.asarray(sub.obsm["spatial"])
        bd = _gene_score(sub, BILE_DUCT_MARKERS)
        cv = _gene_score(sub, CENTRAL_VEIN_MARKERS)
        if np.isnan(bd).all() or np.isnan(cv).all() or len(coords) < k + 1:
            diagnostics[sample] = {"status": "insufficient_anchors"}
            continue
        bd_anchors = coords[bd >= np.nanquantile(bd, high_quantile)]
        cv_anchors = coords[cv >= np.nanquantile(cv, high_quantile)]
        if len(bd_anchors) < k or len(cv_anchors) < k:
            diagnostics[sample] = {"status": "few_anchors",
                                    "n_bd": int(len(bd_anchors)),
                                    "n_cv": int(len(cv_anchors))}
            continue
        nn_bd = NearestNeighbors(n_neighbors=k).fit(bd_anchors)
        nn_cv = NearestNeighbors(n_neighbors=k).fit(cv_anchors)
        d_bd = nn_bd.kneighbors(coords)[0].mean(axis=1)
        d_cv = nn_cv.kneighbors(coords)[0].mean(axis=1)
        s_i = d_bd / (d_bd + d_cv + 1e-9)  # near bile duct → s≈0 (portal); near CV → s≈1
        s_v[m] = s_i
        diagnostics[sample] = {
            "status": "ok",
            "n_bd_anchors": int(len(bd_anchors)),
            "n_cv_anchors": int(len(cv_anchors)),
        }
    return s_v, diagnostics


# ---------------------------------------------------------------------------
# Published-panel axis (§5.4 Variant B) — Supp Table 2 reader + directions
# ---------------------------------------------------------------------------

def load_supp_table_2() -> pd.DataFrame:
    """
    Read the non-steatotic mean-zonation sheet. Header is on row 1
    (row 0 is the description prose). Returns a DataFrame indexed by
    gene_name with Mean_Zone_1..8 columns.
    """
    df = pd.read_excel(SUPP_T2, sheet_name="mean zonation of non steatotic",
                        header=1)
    df.columns = [str(c).strip() for c in df.columns]
    if "gene_name" not in df.columns:
        # Some sheets use Gene_Name; normalise.
        for c in df.columns:
            if c.lower().replace("_", "") == "genename":
                df = df.rename(columns={c: "gene_name"})
                break
    df = df.dropna(subset=["gene_name"]).copy()
    df["gene_name"] = df["gene_name"].astype(str)
    df = df.set_index("gene_name")
    # Coerce numeric columns.
    zone_cols = [c for c in df.columns if c.startswith("Mean_Zone_") or c.startswith("mean_zone_")]
    df[zone_cols] = df[zone_cols].apply(pd.to_numeric, errors="coerce")
    # Normalise zone column casing.
    df.columns = [c.replace("mean_zone_", "Mean_Zone_") for c in df.columns]
    return df


def derive_directions(df: pd.DataFrame, min_max_zone: float = 1e-6) -> dict[str, int]:
    """
    For each gene, compute portal_score = mean(Mean_Zone_1, Mean_Zone_2) and
    central_score = mean(Mean_Zone_7, Mean_Zone_8). Direction:
      +1 if central > portal (central-enriched)
      -1 if portal > central (portal-enriched)
       0 if equal or both zero
    Genes with maximum zone expression below min_max_zone are excluded
    (matches the paper's threshold for the panel).
    """
    portal_cols = [c for c in PORTAL_LAYERS if c in df.columns]
    central_cols = [c for c in CENTRAL_LAYERS if c in df.columns]
    if not portal_cols or not central_cols:
        raise RuntimeError(
            f"missing zone columns: portal {portal_cols}, central {central_cols}"
        )
    p = df[portal_cols].mean(axis=1)
    c = df[central_cols].mean(axis=1)
    zone_cols = [col for col in df.columns if col.startswith("Mean_Zone_")]
    max_zone = df[zone_cols].max(axis=1)
    direction = pd.Series(0, index=df.index, dtype=int)
    direction[(c > p) & (max_zone > min_max_zone)] = +1
    direction[(p > c) & (max_zone > min_max_zone)] = -1
    return direction.to_dict()


def build_axis_from_panel(adata_lhd: ad.AnnData,
                          portal_genes: list[str],
                          central_genes: list[str],
                          sample_col: str = "sample_id") -> tuple[np.ndarray, dict]:
    """
    §5.1 axis using the published portal/central gene sets. Per sample:
    z-score-difference rank-normalize, then orient by the 9 markers.
    """
    s = np.full(adata_lhd.n_obs, np.nan)
    diag: dict[str, dict] = {}
    samples_arr = adata_lhd.obs[sample_col].astype(str).to_numpy()
    portal_in_data = [g for g in portal_genes if g in adata_lhd.var_names]
    central_in_data = [g for g in central_genes if g in adata_lhd.var_names]
    for sample in np.unique(samples_arr):
        m = samples_arr == sample
        sub = adata_lhd[m]
        P = _gene_score(sub, portal_in_data)
        C = _gene_score(sub, central_in_data)
        if np.isnan(P).all() or np.isnan(C).all():
            diag[sample] = {"status": "no_genes"}
            continue
        Pz = (P - np.nanmean(P)) / (np.nanstd(P) + 1e-9)
        Cz = (C - np.nanmean(C)) / (np.nanstd(C) + 1e-9)
        delta = Cz - Pz
        ranks = pd.Series(delta).rank(method="average").to_numpy() - 1
        s_i = ranks / max(1, len(ranks) - 1)
        # Orient using the literature markers: pericentral markers higher at high-s.
        cy = _gene_score(sub, ("CYP2E1", "CYP1A2"))
        flipped = False
        if not np.isnan(cy).all():
            hi = s_i > 0.5
            if cy[hi].mean() < cy[~hi].mean():
                s_i = 1.0 - s_i
                flipped = True
        s[m] = s_i
        diag[sample] = {
            "status": "ok",
            "n_spots": int(m.sum()),
            "orientation_flipped": bool(flipped),
            "n_portal_used": len(portal_in_data),
            "n_central_used": len(central_in_data),
        }
    return s, diag


# ---------------------------------------------------------------------------
# Single-donor axis builder — the §5.1 recipe for P-cohort donors
# ---------------------------------------------------------------------------

def build_axis_for_p_donor(adata: sc.AnnData, donor: str) -> np.ndarray:
    """§5.1 score axis on a single donor (used to add obs.s for P6, which
    Stage B did not include). Returns array length n_obs with values only
    at donor's spots, NaN elsewhere."""
    samples_arr = adata.obs["sample_id"].astype(str).to_numpy()
    m = samples_arr == donor
    s_donor = np.full(adata.n_obs, np.nan)
    if m.sum() < 10:
        return s_donor
    sub = adata[m]
    P = _gene_score(sub, PORTAL_MARKERS)
    C = _gene_score(sub, CENTRAL_MARKERS)
    if np.isnan(P).all() or np.isnan(C).all():
        return s_donor
    Pz = (P - np.nanmean(P)) / (np.nanstd(P) + 1e-9)
    Cz = (C - np.nanmean(C)) / (np.nanstd(C) + 1e-9)
    delta = Cz - Pz
    ranks = pd.Series(delta).rank(method="average").to_numpy() - 1
    s_d = ranks / max(1, len(ranks) - 1)
    cy = _gene_score(sub, ("CYP2E1", "CYP1A2"))
    if not np.isnan(cy).all() and cy[s_d > 0.5].mean() < cy[s_d <= 0.5].mean():
        s_d = 1.0 - s_d
    s_donor[m] = s_d
    return s_donor


def build_p6_axis(adata: sc.AnnData) -> np.ndarray:
    """§5.1 axis on P6 (Stage B excluded P6 from axis construction).
    Same recipe as D2 / D6."""
    samples_arr = adata.obs["sample_id"].astype(str).to_numpy()
    s_full = adata.obs["s"].to_numpy().copy() if "s" in adata.obs.columns else np.full(adata.n_obs, np.nan)
    m = samples_arr == "P6"
    if m.sum() == 0 or np.isfinite(s_full[m]).any():
        return s_full
    sub = adata[m]
    P = _gene_score(sub, PORTAL_MARKERS)
    C = _gene_score(sub, CENTRAL_MARKERS)
    if np.isnan(P).all() or np.isnan(C).all():
        return s_full
    Pz = (P - np.nanmean(P)) / (np.nanstd(P) + 1e-9)
    Cz = (C - np.nanmean(C)) / (np.nanstd(C) + 1e-9)
    delta = Cz - Pz
    ranks = pd.Series(delta).rank(method="average").to_numpy() - 1
    s_p = ranks / max(1, len(ranks) - 1)
    cy = _gene_score(sub, ("CYP2E1", "CYP1A2"))
    if not np.isnan(cy).all() and cy[s_p > 0.5].mean() < cy[s_p <= 0.5].mean():
        s_p = 1.0 - s_p
    s_full[m] = s_p
    return s_full
