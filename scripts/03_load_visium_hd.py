"""Load Visium HD (integrated M2/M6 segmented + raw M1 8 µm bins).

Purpose: load the Visium HD data along two paths — the pre-segmented integrated
h5ad for M2 and M6, and the raw 8 µm bin matrix for M1 — and concatenate them
into one AnnData tagged with ``obs.platform_path``.

Reads:
  - integrated Visium HD h5ad (``visiumHD_data_M2_M6.h5ad``)
  - raw Visium HD M1 8 µm bin matrix + tissue positions

Writes:
  - ``results/visium_hd.h5ad``

Feeds: cross-platform (Fig 2A, R2.1-2.3); W platform-split (Fig 3D, R3.5);
Methods §4.1 HD-load. HD-M1 is loaded but its steatosis sub-check (legacy D5)
is dropped.

Absorbs legacy ``stage_A/A3_visium_hd.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/sgd package importable without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from sgd.config import DATA, RESULTS
from sgd.io import load_visium_sample, file_size_mb

warnings.filterwarnings("ignore")
np.random.seed(0)

INVENTORY = RESULTS / "data_inventory.json"
OUT = RESULTS / "visium_hd.h5ad"
# Legacy A3 read raw HD from RAW/VisiumHD; the fresh repo downloads it into data/.
HD_DIR = DATA / "VisiumHD"
INTEGRATED_H5AD = HD_DIR / "visiumHD_data_M2_M6.h5ad"


def load_path_A() -> ad.AnnData:
    """Pre-segmented integrated h5ad for M2 and M6."""
    if not INTEGRATED_H5AD.exists():
        raise FileNotFoundError(INTEGRATED_H5AD)
    print(f"[A3] Loading integrated HD h5ad ({file_size_mb(INTEGRATED_H5AD):.0f} MB)...")
    adata = sc.read_h5ad(str(INTEGRATED_H5AD))
    adata.obs["platform_path"] = "hd_segmented"
    if "sample_id" not in adata.obs.columns:
        for cand in ("sample", "Sample", "donor", "donor_id"):
            if cand in adata.obs.columns:
                adata.obs["sample_id"] = adata.obs[cand].astype(str)
                break
    print(f"[A3]   path A: {adata.n_obs} units × {adata.n_vars} genes; "
          f"obs cols: {list(adata.obs.columns)[:8]}...")
    return adata


def load_path_B_M1() -> ad.AnnData:
    """Raw 8 µm bin matrix for M1 (no segmentation)."""
    d = HD_DIR / "M1"
    h5 = d / "filtered_feature_bc_matrix_8um.h5"
    pos = d / "tissue_positions_orig.csv"
    sf = d / "scalefactors_json.json"
    if not h5.exists():
        raise FileNotFoundError(h5)
    print(f"[A3] Loading M1 raw 8 µm bins from {h5} ({file_size_mb(h5):.0f} MB)...")
    adata = load_visium_sample(
        h5_path=h5,
        tissue_positions_path=pos,
        scalefactors_path=sf if sf.exists() else None,
        sample_id="M1",
        in_tissue_only=True,
    )
    # HD bins are sparse — drop empties only.
    sc.pp.filter_cells(adata, min_counts=1)
    adata.obs["platform_path"] = "hd_raw_8um"
    adata.obs["bin_size"] = "8um"
    print(f"[A3]   path B (M1): {adata.n_obs} bins × {adata.n_vars} genes")
    return adata


def main() -> None:
    a = load_path_A()
    b = load_path_B_M1()

    # Concat with outer join: M1's gene panel may differ from the integrated h5ad.
    # We do not re-normalise here; A and B may already be on different scales.
    # Stage C will re-normalise for cross-platform comparison.
    adata = ad.concat([a, b], axis=0, join="outer", merge="same",
                      uns_merge="same", label="hd_path", keys=["A", "B"],
                      index_unique="-")
    print(f"[A3] Concatenated HD: {adata.n_obs} units × {adata.n_vars} genes")

    adata.write_h5ad(OUT, compression="gzip")
    print(f"[A3] Wrote {OUT} ({file_size_mb(OUT):.1f} MB)")

    if INVENTORY.exists():
        inv = json.loads(INVENTORY.read_text())
        inv["visium_hd"]["loaded"] = {
            "n_units_total": int(adata.n_obs),
            "n_path_A": int(a.n_obs),
            "n_path_B": int(b.n_obs),
            "out_path": str(OUT),
            "out_size_mb": file_size_mb(OUT),
        }
        INVENTORY.write_text(json.dumps(inv, indent=2, default=str))


if __name__ == "__main__":
    main()
