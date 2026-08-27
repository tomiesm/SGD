"""Load the 16 human Visium samples into a single QC'd, normalised AnnData.

Purpose: load all 16 human Visium samples in parallel, apply the §4.1 spot/gene
filters, total-count normalise + log1p, select per-sample HVGs, and compute the
strict LHD-cohort HVG-intersection panel.

Reads:
  - ``results/data_inventory.json`` (per-sample file paths from step 01)
  - raw Visium ``filtered_feature_bc_matrix`` h5 + ``tissue_positions`` CSVs

Writes:
  - ``results/visium_human.h5ad`` (16 samples, QC, total-count norm+log1p, raw
    counts in ``layers['counts']``, strict-66 ``var.panel_intersection``)

Feeds: R1.1 (16 samples / 47,828 spots); strict-66 + relaxed panels;
Methods §4.1. Upstream of every analysis.

Absorbs legacy ``stage_A/A2_visium.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/sgd package importable without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from scipy.sparse import issparse

from sgd.config import (RESULTS, LHD_SAMPLES, P_SAMPLES, STEATOTIC_SAMPLES)
from sgd.io import (load_visium_sample, normalize_total_log1p,
                    file_size_mb)

warnings.filterwarnings("ignore")
np.random.seed(0)

INVENTORY = RESULTS / "data_inventory.json"
OUT = RESULTS / "visium_human.h5ad"

N_JOBS = 16  # one per Visium sample
MIN_GENES_PER_SPOT = 200
MIN_UMI_PER_SPOT = 500
MIN_SPOTS_PER_GENE = 3
N_TOP_SVG = 2000
ALLOW_PARTIAL = os.environ.get("SGD_ALLOW_PARTIAL", "0") == "1"


def _detect_mt_pattern(var_names: pd.Index) -> str | None:
    """Return the matching mitochondrial gene prefix in the panel, or None."""
    for prefix in ("MT-", "mt-", "Mt-"):
        if any(g.startswith(prefix) for g in var_names):
            return prefix
    # Ensembl IDs would not have a prefix; fall back to a known list lookup
    # is out of scope here (Stage A flag only).
    return None


def load_one(sample_id: str, paths: dict, condition: str,
             steatotic: bool, lipid_pct: float | None) -> ad.AnnData:
    """Load + per-sample QC + per-sample SVG selection."""
    adata = load_visium_sample(
        h5_path=Path(paths["h5_matrix"]),
        tissue_positions_path=Path(paths["tissue_positions"]),
        scalefactors_path=Path(paths["scalefactors"]) if paths.get("scalefactors") else None,
        sample_id=sample_id,
    )

    # §4.1 spot/gene filters.
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    keep_spot = (adata.obs["n_genes_by_counts"] >= MIN_GENES_PER_SPOT) & \
                (adata.obs["total_counts"] >= MIN_UMI_PER_SPOT)
    adata = adata[keep_spot].copy()
    # Snapshot the original spot UMI total **before** gene filtering so the
    # confound diagnostics in A7 use the biologically meaningful total, not
    # the post-filter remainder. Also save the raw barcode for cross-stage
    # joins (e.g. lipid supplements keyed by barcode).
    adata.obs["total_umi_raw"] = adata.obs["total_counts"].astype(float).to_numpy()
    adata.obs["barcode"] = adata.obs_names.to_numpy()
    sc.pp.filter_genes(adata, min_cells=MIN_SPOTS_PER_GENE)

    # Mt-fraction (compute, do not filter).
    prefix = _detect_mt_pattern(adata.var_names)
    if prefix:
        mt_mask = adata.var_names.str.startswith(prefix)
        if issparse(adata.X):
            mt_counts = np.asarray(adata.X[:, mt_mask].sum(axis=1)).flatten()
        else:
            mt_counts = adata.X[:, mt_mask].sum(axis=1)
        totals = adata.obs["total_counts"].to_numpy()
        adata.obs["pct_counts_mt"] = 100.0 * mt_counts / np.maximum(totals, 1)
    else:
        adata.obs["pct_counts_mt"] = np.nan
    adata.uns["mt_prefix"] = prefix or "none_found"

    # Tag obs.
    adata.obs["condition"] = condition
    adata.obs["steatotic"] = steatotic
    adata.obs["lipid_pct"] = lipid_pct if lipid_pct is not None else np.nan
    adata.obs["donor_id"] = sample_id

    # Per-sample SVG: standard scanpy seurat_v3 on raw counts, top-N.
    # Equivalent to the plan's "spatially variable" selection at this stage;
    # axis-aware SVG (Moran's I) is a Stage B refinement if needed.
    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", n_top_genes=N_TOP_SVG, layer=None, inplace=True,
    )
    adata.var[f"hvg_top{N_TOP_SVG}_per_sample"] = adata.var["highly_variable"].copy()
    return adata


def main() -> None:
    inv = json.loads(INVENTORY.read_text())
    by_sample_meta = inv["metadata"].get("by_sample", {}) if inv.get("metadata") else {}
    sample_paths = inv["visium_human"]["samples"]

    # Build the work list. Exact manuscript reproduction requires all 16
    # samples; partial runs must be requested explicitly for development.
    plan = []
    incomplete = []
    for sid in LHD_SAMPLES + P_SAMPLES:
        paths = sample_paths.get(sid, {})
        if not paths.get("complete"):
            incomplete.append(sid)
            continue
        cond = "LHD" if sid in LHD_SAMPLES else "P"
        steat = sid in STEATOTIC_SAMPLES
        # Lipid_pct is taken from metadata if a column resembling lipid is found
        # (A1 marks this; here we keep NaN unless explicitly populated).
        lipid_pct = None
        if sid in by_sample_meta:
            for k, v in by_sample_meta[sid].items():
                kl = str(k).lower()
                if any(t in kl for t in ("lipid", "steatos", "fat", "tg")):
                    try:
                        lipid_pct = float(v)
                        break
                    except (TypeError, ValueError):
                        pass
        plan.append((sid, paths, cond, steat, lipid_pct))

    if incomplete and not ALLOW_PARTIAL:
        raise RuntimeError(
            "[A2] exact reproduction requires all 16 Visium samples; "
            f"incomplete inventory for {incomplete}. Re-run step 01 after "
            "installing the inputs, or set SGD_ALLOW_PARTIAL=1 for an "
            "explicit non-reproduction development run."
        )
    if incomplete:
        print(f"[A2] PARTIAL RUN requested; skipping incomplete samples: {incomplete}")

    n_jobs = min(N_JOBS, len(plan), os.cpu_count() or 1)
    print(f"[A2] Loading {len(plan)} samples in parallel (n_jobs={n_jobs})...")
    adatas = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(load_one)(sid, paths, cond, steat, lipid)
        for (sid, paths, cond, steat, lipid) in plan
    )

    # Concatenate. Use outer join on var so per-sample HVG flags survive in var; we'll
    # reduce to the LHD HVG intersection below. Visium barcodes repeat across
    # samples, so we force unique obs_names of the form "{sample_id}:{barcode}";
    # the raw barcode survives in obs["barcode"] for cross-stage joins.
    print("[A2] Concatenating...")
    adata = ad.concat(adatas, axis=0, join="outer", merge="same",
                      uns_merge="same", label="batch")
    adata.obs["sample_id"] = adata.obs["donor_id"]
    adata.obs_names = (adata.obs["sample_id"].astype(str) + ":"
                       + adata.obs["barcode"].astype(str)).to_numpy()
    if not adata.obs_names.is_unique:
        raise RuntimeError("[A2] non-unique obs_names after sample-prefixing — investigate")
    print(f"[A2] Concatenated: {adata.n_obs} spots × {adata.n_vars} genes")

    # Total-count normalize each spot to 1e4, then log1p (raw kept in layers['counts']).
    print("[A2] Total-count normalize + log1p...")
    normalize_total_log1p(adata, target_sum=1e4)

    # LHD intersection HVG panel: a gene is in the panel if it was top-N HVG
    # in every LHD sample where it was present. We rebuild this from the per-
    # sample adatas because var-level flags are lost on concat with outer join.
    print("[A2] Computing LHD HVG intersection panel...")
    panel = None
    for sub in adatas:
        if sub.obs["donor_id"].iloc[0] not in LHD_SAMPLES:
            continue
        hvg_set = set(sub.var_names[sub.var["highly_variable"]])
        panel = hvg_set if panel is None else (panel & hvg_set)
    panel = panel or set()
    adata.var["panel_intersection"] = adata.var_names.isin(panel)
    print(f"[A2] LHD HVG intersection: {int(adata.var['panel_intersection'].sum())} genes")

    # Save.
    adata.write_h5ad(OUT, compression="gzip")
    print(f"[A2] Wrote {OUT} ({file_size_mb(OUT):.1f} MB)")

    # Append summary to inventory for downstream stages.
    inv["visium_human"]["loaded"] = {
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "panel_intersection_size": int(adata.var["panel_intersection"].sum()),
        "samples_loaded": [sid for sid, *_ in plan],
        "out_path": str(OUT),
        "out_size_mb": file_size_mb(OUT),
    }
    INVENTORY.write_text(json.dumps(inv, indent=2, default=str))


if __name__ == "__main__":
    main()
