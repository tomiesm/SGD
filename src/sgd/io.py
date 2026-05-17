"""Visium flat-archive loaders, total-count normalisation, and small I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse


# ---------------------------------------------------------------------------
# Visium I/O — flat-archive aware
# ---------------------------------------------------------------------------

def read_tissue_positions_robust(path: Path) -> pd.DataFrame:
    """
    Read Space Ranger ``tissue_positions[_list].csv`` tolerantly.

    Three formats encountered in this project:
      * Standard Visium (no header, 6 cols).
      * Visium HD M6 (header, 6 cols).
      * Visium HD M1/M2 (header + leading pandas-style row-index col → 7 cols).

    Header is detected by token sniffing — any canonical token in
    the first row → has_header=True. The leading "Unnamed: 0" /
    blank column from a pandas dump is dropped if present.
    """
    cols = ["barcode", "in_tissue", "array_row", "array_col",
            "pxl_row_in_fullres", "pxl_col_in_fullres"]
    tokens = ("barcode", "in_tissue", "pxl_row", "pxl_col", "array_row")
    head = pd.read_csv(path, header=None, nrows=1, dtype=str)
    has_header = any(t in str(v).lower() for v in head.iloc[0].tolist()
                     for t in tokens)
    df = pd.read_csv(path, header=0 if has_header else None)
    if has_header:
        df.columns = [str(c).strip() for c in df.columns]
        # Strip pandas-output leading index column if present.
        if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
            df = df.iloc[:, 1:].copy()
        # If the column names don't already match canonical, force-align.
        if list(df.columns[:6]) != cols:
            df = df.iloc[:, :6].copy()
            df.columns = cols
    else:
        df.columns = cols
    df["in_tissue"] = df["in_tissue"].astype(int)
    return df


def load_visium_sample(
    h5_path: Path,
    tissue_positions_path: Path,
    scalefactors_path: Path | None,
    sample_id: str,
    in_tissue_only: bool = True,
) -> ad.AnnData:
    """
    Load a single Visium sample from flat archive layout into AnnData.

    Avoids ``sc.read_visium`` because the bundled archives don't carry
    the standard Space Ranger directory tree (no ``spatial/`` subdir,
    no lowres image). Reads the matrix, joins tissue positions onto
    ``obs``, sets ``obsm['spatial']`` to pixel coords, and stores
    scalefactors in ``uns`` if provided.
    """
    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()

    pos = read_tissue_positions_robust(Path(tissue_positions_path))
    pos = pos.set_index("barcode")
    common = adata.obs_names.intersection(pos.index)
    if len(common) == 0:
        raise RuntimeError(f"[{sample_id}] no barcode overlap between matrix and tissue positions")
    adata = adata[common].copy()
    pos = pos.loc[adata.obs_names]
    for col in ("in_tissue", "array_row", "array_col",
                "pxl_row_in_fullres", "pxl_col_in_fullres"):
        adata.obs[col] = pos[col].to_numpy()
    adata.obsm["spatial"] = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy().astype(float)

    if in_tissue_only:
        adata = adata[adata.obs["in_tissue"] == 1].copy()

    if scalefactors_path is not None and Path(scalefactors_path).exists():
        with open(scalefactors_path) as f:
            adata.uns["spatial_scalefactors"] = json.load(f)

    adata.obs["sample_id"] = sample_id
    adata.obs["sample_id"] = adata.obs["sample_id"].astype("category")
    return adata


# ---------------------------------------------------------------------------
# Total-count normalisation (§4.1)
# ---------------------------------------------------------------------------

def normalize_total_log1p(adata: ad.AnnData, target_sum: float = 1e4) -> None:
    """
    Total-count normalise each spot to ``target_sum``, then log1p, in-place.

    Each spot is rescaled so its counts sum to ``target_sum`` (default 1e4);
    the scale factor is ``target_sum / spot_total`` and depends only on the
    spot itself. Raw counts are preserved in ``layers['counts']`` for the
    log(UMI) covariate used by the downstream confound diagnostics.
    """
    adata.layers["counts"] = adata.X.copy()  # preserve raw counts
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def total_umi(adata: ad.AnnData) -> np.ndarray:
    """
    Per-spot total UMI. Prefers ``obs['total_umi_raw']`` (snapshot taken
    in A2 before gene filtering), falls back to ``layers['counts']``,
    then ``X``. The raw snapshot is the right input for the §4.6
    confound diagnostics — gene-filtered totals miss the mt-tail and
    other low-detected gene contributions that are part of the per-spot
    capture-efficiency signal.
    """
    if "total_umi_raw" in adata.obs.columns:
        return adata.obs["total_umi_raw"].to_numpy().astype(float)
    src = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if issparse(src):
        return np.asarray(src.sum(axis=1)).flatten()
    return src.sum(axis=1)


def normalise_supp_t10_barcode(spot_name: str) -> str:
    """Supp T10 uses '_1' suffix; our barcodes use '-1'."""
    if spot_name.endswith("_1"):
        return spot_name[:-2] + "-1"
    return spot_name


def file_size_mb(path: Path) -> float:
    return Path(path).stat().st_size / 1e6
