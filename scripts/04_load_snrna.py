"""Verify the integrated snRNA-seq reference and record a pointer to it.

Purpose: load the integrated ``snRNAseq.h5ad`` in backed mode, verify the obs
columns the cell2location reference needs (cell-type annotation, sample id),
record the verification result, and create a pointer to the file. Does NOT
re-process and does NOT copy the multi-GB file.

Reads:
  - ``config.SNRNA_PATH`` — integrated ``snRNAseq.h5ad`` (Zenodo download)

Writes:
  - ``results/snrna_ref.h5ad`` symlink (or ``results/snrna_ref.path`` pointer)
  - snRNA verification block appended to ``results/data_inventory.json``

Feeds: cell2location reference for step 10. No manuscript number of its own.

Absorbs legacy ``stage_A/A4_snrna.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/sgd package importable without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import warnings

import scanpy as sc

from sgd.config import RESULTS, SNRNA_PATH
from sgd.io import file_size_mb

warnings.filterwarnings("ignore")

INVENTORY = RESULTS / "data_inventory.json"
SYMLINK = RESULTS / "snrna_ref.h5ad"

# Candidate column names we will look for; first match wins.
CELLTYPE_CANDIDATES = ("cell_type", "celltype", "cluster", "leiden",
                       "cell_type_final", "annotation", "harmonized_celltype")
SAMPLE_CANDIDATES = ("sample_id", "sample", "donor", "donor_id", "Sample")


def find_first(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def main() -> None:
    inv = json.loads(INVENTORY.read_text())
    # Legacy A4 sourced the snRNA path from the inventory; the fresh repo
    # resolves it from config.SNRNA_PATH (the single portability fix).
    src = SNRNA_PATH
    if not src.exists():
        print("[A4] integrated snRNAseq h5ad not found at config.SNRNA_PATH; skipping")
        return
    src = Path(src)

    print(f"[A4] Loading {src} ({file_size_mb(src):.0f} MB) in backed mode...")
    adata = sc.read_h5ad(str(src), backed="r")  # verify obs/var/var_names only — no matrix access

    cols = list(adata.obs.columns)
    var_n = list(adata.var.columns)
    ct_col = find_first(cols, CELLTYPE_CANDIDATES)
    s_col = find_first(cols, SAMPLE_CANDIDATES)

    summary = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_cols": cols,
        "var_cols": var_n,
        "celltype_col": ct_col,
        "sample_col": s_col,
        "obsm_keys": list(adata.obsm.keys()) if hasattr(adata, "obsm") else [],
        "layers": list(adata.layers.keys()) if hasattr(adata, "layers") else [],
        "var_names_head": list(adata.var_names[:5]),
        "var_names_appear_HGNC": all(not v.startswith("ENS") for v in adata.var_names[:50]),
    }
    if ct_col:
        summary["n_celltypes"] = int(adata.obs[ct_col].nunique())
    if s_col:
        summary["samples"] = sorted(adata.obs[s_col].astype(str).unique().tolist())

    # Try symlink; if cross-filesystem permissions disallow, write a path-pointer
    # text file instead. Either way the downstream stages should resolve via
    # data_inventory.json (the canonical path lookup).
    pointer = SYMLINK.with_suffix(".path")
    if SYMLINK.exists() or SYMLINK.is_symlink():
        SYMLINK.unlink()
    if pointer.exists():
        pointer.unlink()
    try:
        SYMLINK.symlink_to(src)
        print(f"[A4] Symlinked {SYMLINK} → {src}")
        link_path = str(SYMLINK)
    except (OSError, PermissionError) as e:
        pointer.write_text(str(src))
        print(f"[A4] Symlink not permitted ({e.__class__.__name__}); "
              f"wrote pointer {pointer} → {src}")
        link_path = str(pointer)

    inv["snrna"]["loaded"] = {**summary, "ref_path": link_path}
    INVENTORY.write_text(json.dumps(inv, indent=2, default=str))
    for k, v in summary.items():
        if isinstance(v, list) and len(v) > 12:
            v = f"<{len(v)} items>"
        print(f"[A4]   {k}: {v}")


if __name__ == "__main__":
    main()
