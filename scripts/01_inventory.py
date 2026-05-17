"""Inventory the raw input tree and fetch the Yakubovsky published reference.

Purpose: extract the human Visium archives, scan the dataset tree, ingest the
per-sample metadata, and clone/inventory the Yakubovsky GitHub reference repo —
the prerequisite step that produces the input manifests every later step keys
off. This script is h5ad-free; the lipid-supplement encoding (which needs the
loaded h5ad) is folded into ``05_axis_and_confounds.py`` instead.

Reads:
  - raw Zenodo download tree under ``data/`` (Visium zips, Visium HD, snRNA-seq,
    metadata xlsx)
  - Yakubovsky GitHub ``OranYak/Human-liver`` (cloned into ``data/``)

Writes:
  - ``results/data_inventory.json``
  - ``results/yakubovsky_reference.json``

Feeds: prerequisite for every later step; no standalone manuscript number.

Absorbs legacy ``stage_A/A1_inventory.py`` and ``stage_B/B1_yakubovsky_fetch.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/sgd package importable without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hashlib
import json
import re
import shutil
import subprocess
import zipfile

import numpy as np
import pandas as pd

from sgd.config import (DATA, RESULTS, LHD_SAMPLES, P_SAMPLES,
                        STEATOTIC_SAMPLES)
from sgd.io import file_size_mb

np.random.seed(0)

# --- Paths ---------------------------------------------------------------
# Legacy A1 read raw inputs from a RAW root and wrote into DATA. The fresh
# repo downloads the raw tree into data/ (see data/README.md) and writes
# regenerated artifacts into results/.
RAW = DATA
VISIUM_DIR = RAW / "Visium"
VISIUM_HD_DIR = RAW / "VisiumHD"
MERFISH_DIR = RAW / "MERFISH"
SNRNA_DIR = RAW / "snRNAseq"
META_XLSX = RAW / "human_samples_metadata.xlsx"
META_NONHUMAN_CSV = RAW / "non_Human_samples_metadata.csv"

STAGE_VISIUM = RESULTS / "visium_extracted"
INVENTORY_OUT = RESULTS / "data_inventory.json"

VISIUM_ZIPS_TO_EXTRACT = [
    ("Human_h5_files.zip", "h5"),
    ("Human_Spatial_transcriptomics_data.zip", "spatial"),
]

# --- B1 paths ------------------------------------------------------------
REPO_URL = "https://github.com/OranYak/Human-liver.git"
REPO_DIR = DATA / "yakubovsky_repo"
YAKUBOVSKY_OUT = RESULTS / "yakubovsky_reference.json"


# ===========================================================================
# A1 — file extraction, staging-directory inventory, metadata ingestion
# ===========================================================================

def md5_short(p: Path, n_bytes: int = 1_000_000) -> str:
    """First-1MB MD5 — quick fingerprint, not a full integrity check."""
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:12]


def extract_zip(zip_path: Path, dest: Path) -> list[str]:
    """
    Extract a zip into ``dest`` only if not already extracted.

    Uses the system ``unzip`` binary because the bundled archives use
    Deflate64 (Windows zip default), which Python's stdlib ``zipfile``
    cannot decompress. Falls back to ``zipfile`` for plain Deflate.
    Listing always uses ``zipfile`` since the central directory itself
    is plain.
    """
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / (zip_path.stem + ".extracted")
    if marker.exists():
        with zipfile.ZipFile(zip_path) as zf:
            return zf.namelist()
    if shutil.which("unzip"):
        # -o overwrite, -q quiet, -d destination.
        subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(dest)], check=True)
    else:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    marker.touch()
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


def list_zip_contents(zip_path: Path) -> list[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        return [{"name": zi.filename, "size": zi.file_size}
                for zi in zf.infolist() if not zi.is_dir()]


def _sample_prefix_match(name: str, sample: str) -> bool:
    """
    Match a filename against a sample id by exact prefix + a delimiter.
    Prevents ``P2`` from accidentally matching ``P21_*`` files.
    Accepts ``{sample}_…``, ``{sample}.…``, ``{sample}-…``, or ``{sample}`` alone.
    """
    return re.match(rf"^{re.escape(sample)}([._-]|$)", name) is not None


def find_sample_files(extracted_root: Path, sample: str) -> dict:
    """
    Locate the four files we need for one Visium sample inside the
    flat extraction tree. Tolerant to underscore/dot/dash separators
    after the sample id; rejects same-prefix collisions like P2 vs P21.
    """
    found: dict[str, str | None] = {"h5": None, "tissue_positions": None,
                                     "scalefactors": None, "image_hires": None}
    for p in extracted_root.rglob("*"):
        if not p.is_file() or not _sample_prefix_match(p.name, sample):
            continue
        n = p.name.lower()
        if found["h5"] is None and ("filtered_feature" in n) and n.endswith(".h5"):
            found["h5"] = str(p)
        elif found["tissue_positions"] is None and "tissue_positions" in n and n.endswith(".csv"):
            found["tissue_positions"] = str(p)
        elif found["scalefactors"] is None and "scalefactors" in n and n.endswith(".json"):
            found["scalefactors"] = str(p)
        elif found["image_hires"] is None and "tissue_hires" in n and n.endswith((".png", ".jpg", ".tif", ".tiff")):
            found["image_hires"] = str(p)
    return found


def parse_metadata(xlsx: Path) -> dict:
    """
    Read ``human_samples_metadata.xlsx``. Returns a dict
    {sample_id -> {column -> value}} plus a list of all column names.
    Best-effort: if the sample ID column has a different name, look
    for the first column that matches all-known LHD+P sample tokens.
    """
    if not xlsx.exists():
        return {"available": False, "reason": f"{xlsx} not found"}
    df = pd.read_excel(xlsx, sheet_name=0)
    df.columns = [c.strip() for c in df.columns]
    expected = set(LHD_SAMPLES + P_SAMPLES)
    sample_col = None
    for c in df.columns:
        vals = set(df[c].dropna().astype(str).str.strip().tolist())
        if expected & vals:
            sample_col = c
            break
    if sample_col is None:
        return {"available": True, "sample_col_detected": None,
                "columns": list(df.columns), "n_rows": int(len(df)),
                "rows": df.to_dict(orient="records")}
    df[sample_col] = df[sample_col].astype(str).str.strip()
    rec: dict[str, dict] = {}
    for _, r in df.iterrows():
        sid = str(r[sample_col]).strip()
        if sid in expected:
            rec[sid] = {k: (None if pd.isna(v) else v) for k, v in r.items()}
    return {"available": True, "sample_col_detected": sample_col,
            "columns": list(df.columns), "n_rows": int(len(df)), "by_sample": rec}


def lipid_search_local(meta: dict) -> dict:
    """
    Inspect the metadata columns for any lipid-related field.
    Per the user, none is present in the Excel; this confirms it
    programmatically and records which keywords were searched.
    """
    keywords = ("lipid", "steatos", "fat", "triglyc", "lipo", "macrov", "microv", "tg_pct")
    cols = [c.lower() for c in meta.get("columns", [])]
    matches = {}
    for c in cols:
        for kw in keywords:
            if kw in c:
                matches.setdefault(c, []).append(kw)
    return {"keywords": list(keywords), "matches": matches,
            "found_in_excel": bool(matches)}


def run_inventory() -> None:
    """A1 — extract Visium archives, resolve per-sample files, inventory all
    modalities, ingest metadata, write data_inventory.json."""
    RESULTS.mkdir(parents=True, exist_ok=True)

    # --- 1. Extract human Visium archives -------------------------------
    extraction_log = {}
    for fname, tag in VISIUM_ZIPS_TO_EXTRACT:
        zip_path = VISIUM_DIR / fname
        dest = STAGE_VISIUM / tag
        if not zip_path.exists():
            extraction_log[fname] = {"status": "missing"}
            continue
        names = extract_zip(zip_path, dest)
        extraction_log[fname] = {"status": "ok", "n_entries": len(names),
                                  "dest": str(dest)}

    # --- 2. Per-sample file resolution ----------------------------------
    sample_paths = {}
    for sample in LHD_SAMPLES + P_SAMPLES:
        h5 = find_sample_files(STAGE_VISIUM / "h5", sample)
        sp = find_sample_files(STAGE_VISIUM / "spatial", sample)
        merged = {
            "h5_matrix": h5["h5"],
            "tissue_positions": sp["tissue_positions"] or h5["tissue_positions"],
            "scalefactors": sp["scalefactors"] or h5["scalefactors"],
            "image_hires": sp["image_hires"] or h5["image_hires"],
        }
        merged["complete"] = bool(merged["h5_matrix"] and merged["tissue_positions"])
        sample_paths[sample] = merged

    # --- 3. Visium HD inventory -----------------------------------------
    hd_inventory = {}
    for sub in ("M1", "M2", "M6"):
        d = VISIUM_HD_DIR / sub
        files = {
            "filtered_h5": next((str(p) for p in d.glob("filtered_feature_bc_matrix_8um.h5")), None),
            "raw_h5": next((str(p) for p in d.glob("raw_feature_bc_matrix_8um.h5")), None),
            "tissue_positions": next(
                (str(p) for p in d.glob("tissue_positions*.csv")), None),
            "scalefactors": next((str(p) for p in d.glob("scalefactors_json.json")), None),
        }
        hd_inventory[sub] = files
    hd_inventory["integrated_h5ad"] = (
        str(VISIUM_HD_DIR / "visiumHD_data_M2_M6.h5ad")
        if (VISIUM_HD_DIR / "visiumHD_data_M2_M6.h5ad").exists() else None
    )

    # --- 4. MERFISH inventory -------------------------------------------
    merfish_inventory = {}
    for sub in ("MERFISH_M5", "MERFISH_M8"):
        d = MERFISH_DIR / sub
        merfish_inventory[sub] = {
            "cell_by_gene": str(d / "cell_by_gene.csv") if (d / "cell_by_gene.csv").exists() else None,
            "cell_metadata": str(d / "cell_metadata.csv") if (d / "cell_metadata.csv").exists() else None,
            "detected_transcripts": str(d / "detected_transcripts.csv") if (d / "detected_transcripts.csv").exists() else None,
        }

    # --- 5. snRNA-seq inventory -----------------------------------------
    snrna_inventory = {
        "integrated_h5ad": str(SNRNA_DIR / "snRNAseq.h5ad")
        if (SNRNA_DIR / "snRNAseq.h5ad").exists() else None,
        "raw_dirs": [str(SNRNA_DIR / s) for s in ("M5", "M6", "M7", "M8")
                     if (SNRNA_DIR / s).exists()],
    }

    # --- 6. Metadata + lipid search -------------------------------------
    metadata = parse_metadata(META_XLSX)
    lipid_local = lipid_search_local(metadata)
    lipid_status = "found_in_excel" if lipid_local["found_in_excel"] else "not_found_locally"

    # --- 7. Non-human inventory only ------------------------------------
    nonhuman = {
        "h5_zip": {"path": str(VISIUM_DIR / "Non_human_h5_files.zip"),
                   "size_mb": file_size_mb(VISIUM_DIR / "Non_human_h5_files.zip")
                   if (VISIUM_DIR / "Non_human_h5_files.zip").exists() else None,
                   "contents": list_zip_contents(VISIUM_DIR / "Non_human_h5_files.zip")
                   if (VISIUM_DIR / "Non_human_h5_files.zip").exists() else []},
        "spatial_zip": {"path": str(VISIUM_DIR / "Non_Human_Spatial_transcriptomics_data.zip"),
                        "size_mb": file_size_mb(VISIUM_DIR / "Non_Human_Spatial_transcriptomics_data.zip")
                        if (VISIUM_DIR / "Non_Human_Spatial_transcriptomics_data.zip").exists() else None},
        "metadata_csv": str(META_NONHUMAN_CSV) if META_NONHUMAN_CSV.exists() else None,
        "extraction_status": "deferred_stage_B",
    }

    inventory = {
        "extraction_log": extraction_log,
        "visium_human": {
            "samples": sample_paths,
            "complete": [s for s, v in sample_paths.items() if v["complete"]],
            "incomplete": [s for s, v in sample_paths.items() if not v["complete"]],
        },
        "visium_hd": hd_inventory,
        "merfish": merfish_inventory,
        "snrna": snrna_inventory,
        "metadata": metadata,
        "lipid": {"local_search": lipid_local, "status": lipid_status,
                  "next_actions_if_missing": [
                      "search Yakubovsky GitHub https://github.com/OranYak/Human-liver",
                      "search Zenodo 10.5281/zenodo.17735506 ancillary files",
                      "search PhenoCycler companion 10.5281/zenodo.17730820",
                      "if found, write to data/lipid_supplement.json with keys "
                      "{sample_id: {spot_barcode: lipid_pct}} or {sample_id: lipid_pct}"]},
        "nonhuman": nonhuman,
        "sample_taxonomy": {
            "lhd": list(LHD_SAMPLES),
            "p": list(P_SAMPLES),
            "steatotic": list(STEATOTIC_SAMPLES),
        },
    }

    INVENTORY_OUT.write_text(json.dumps(inventory, indent=2, default=str))
    print(f"[A1] Wrote {INVENTORY_OUT} ({file_size_mb(INVENTORY_OUT):.2f} MB)")
    print(f"[A1] Visium human samples complete: "
          f"{len(inventory['visium_human']['complete'])}/{len(LHD_SAMPLES) + len(P_SAMPLES)}")
    if inventory['visium_human']['incomplete']:
        print(f"[A1] Incomplete: {inventory['visium_human']['incomplete']}")
    print(f"[A1] Lipid status: {lipid_status}")
    print(f"[A1] HD samples available: "
          f"M1={hd_inventory['M1']['filtered_h5'] is not None} "
          f"M2={hd_inventory['M2']['filtered_h5'] is not None} "
          f"M6={hd_inventory['M6']['filtered_h5'] is not None} "
          f"integrated={hd_inventory['integrated_h5ad'] is not None}")


# ===========================================================================
# B1 — fetch the Yakubovsky published reference panel
# ===========================================================================

def clone_or_update(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    if not shutil.which("git"):
        print(f"[B1] git not available; skipping clone of {url}")
        return False
    res = subprocess.run(["git", "clone", "--depth", "1", url, str(dest)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[B1] clone failed: {res.stderr.strip()[:200]}")
        return False
    return True


def find_panel_candidates(repo_dir: Path) -> list[dict]:
    """Files whose names look like a zonation panel or zone-index export."""
    keywords = ("zonation", "zone_index", "panel", "1711", "marker",
                "hepatocyte_zon", "central", "portal")
    out = []
    for p in repo_dir.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if any(k in name for k in keywords) and p.suffix.lower() in (
            ".csv", ".tsv", ".xlsx", ".xls", ".mat", ".rds", ".json"):
            out.append({"path": str(p.relative_to(repo_dir)),
                         "size_bytes": p.stat().st_size,
                         "suffix": p.suffix.lower()})
    return out


def run_yakubovsky_fetch() -> None:
    """B1 — clone the Yakubovsky GitHub repo, inventory it for the published
    reference panel, write yakubovsky_reference.json."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    cloned = clone_or_update(REPO_URL, REPO_DIR)
    payload: dict = {
        "repo_url": REPO_URL,
        "repo_cloned": cloned,
        "repo_dir": str(REPO_DIR) if cloned else None,
        "panel_1711_path": None,
        "zone_index_path": None,
        "candidates": [],
        "status": "not_evaluable",
        "notes": [],
    }

    if cloned:
        candidates = find_panel_candidates(REPO_DIR)
        payload["candidates"] = candidates
        # Heuristic: pick the largest CSV/TSV mentioning "zonation" or "1711".
        scored = sorted(
            (c for c in candidates if c["suffix"] in (".csv", ".tsv", ".xlsx")),
            key=lambda c: (("1711" in c["path"].lower())
                            + ("zonation" in c["path"].lower()),
                            c["size_bytes"]),
            reverse=True,
        )
        if scored:
            payload["panel_1711_path"] = scored[0]["path"]
            payload["status"] = "candidate_found"
            payload["notes"].append(
                f"Best candidate: {scored[0]['path']} ({scored[0]['size_bytes']} B). "
                f"Inspect manually before using as Variant B input.")
        else:
            payload["notes"].append(
                "No CSV/TSV/XLSX file with zonation-panel keywords found in "
                "Yakubovsky GitHub. The 1,711-gene panel and per-spot zone index "
                "are likely in the paper Supplementary Information rather than "
                "the public code repo.")
    else:
        payload["notes"].append(
            "Clone failed; B1 cannot search the GitHub repo. Stage B proceeds "
            "with s_v_B = not_evaluable per stage_B_plan.md.")

    YAKUBOVSKY_OUT.write_text(json.dumps(payload, indent=2))
    print(f"[B1] Wrote {YAKUBOVSKY_OUT}")
    print(f"[B1] Status: {payload['status']}")
    for note in payload["notes"]:
        print(f"[B1]   {note}")


def main() -> None:
    run_inventory()
    run_yakubovsky_fetch()


if __name__ == "__main__":
    main()
