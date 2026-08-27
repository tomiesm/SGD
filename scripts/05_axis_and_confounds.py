"""Build the spatial axis, fibrotic mask, lipid supplement, and confound diagnostics.

Purpose: derive the §5.1 preliminary portal-central axis and the §4.5 fibrotic-
spot mask, encode the Yakubovsky lipid annotations, run the C1/C2/C3/C5 confound
diagnostics, and build the validated primary axis plus its variants — everything
that turns the loaded Visium AnnData into an axis-bearing object.

Reads:
  - ``results/visium_human.h5ad``
  - Yakubovsky GitHub lipid CSVs under a documented ``Loupe_categories/``
    layout, or the directory selected by ``SGD_LOUPE_DIR``
  - ``results/yakubovsky_reference.json``

Writes:
  - ``results/visium_human.h5ad`` updated in place with ``obs.s`` (validated
    primary axis), ``obs.s_alt``, ``obs.s_v_C``, ``obs.s_preliminary``,
    ``obs.fibrotic_spot``
  - ``results/lipid_supplement.json``
  - ``results/axis_diagnostics.json``
  - ``results/confound_diagnostics.json``
  - ``results/axis_variants_summary.json``

Feeds: spatial-axis Methods §4.3; fibrotic exclusion §4.1; confound diagnostics
C1/C2/C3/C5 §4.2 → supp fig S1 (C4 row comes from step 10). ``obs.s`` is the
backbone of every gradient computation.

Note: ``obs.s_v_B`` is NOT written here — it depends on Supp Table 2
and step 07 is its sole writer. ``obs.s_v_B`` must be genuinely absent (not
NaN-filled) when step 06 runs.

Absorbs legacy ``stage_A/build_lipid_supplement.py``, ``stage_A/A6_axis_fibrotic.py``,
``stage_A/A7_diagnostics.py``, ``stage_B/B2_axis_variants.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/sgd package importable without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hashlib
import json
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors

from sgd.config import (DATA, RESULTS, LHD_SAMPLES, LOUPE_DIR_OVERRIDE,
                        STEATOTIC_SAMPLES)
from sgd.axis import (build_landmark_axis, build_score_axis,
                      compute_preliminary_axis, derive_fibrotic_mask)
from sgd.io import file_size_mb
from sgd.panels import (CENTRAL_MARKERS, PORTAL_MARKERS, _gene_score)
from sgd.confounds import c5_score
from sgd.io import total_umi

warnings.filterwarnings("ignore")
np.random.seed(0)

# --- Paths ---------------------------------------------------------------
VISIUM_H5AD = RESULTS / "visium_human.h5ad"
LIPID_SUPP = RESULTS / "lipid_supplement.json"
AXIS_DIAG = RESULTS / "axis_diagnostics.json"
CONFOUND_OUT = RESULTS / "confound_diagnostics.json"
AXIS_VARIANTS_OUT = RESULTS / "axis_variants_summary.json"
INVENTORY = RESULTS / "data_inventory.json"
YAKUB = RESULTS / "yakubovsky_reference.json"

def resolve_loupe_dir() -> Path:
    """Resolve the published Loupe annotations without silently inventing zeros."""
    candidates = []
    if LOUPE_DIR_OVERRIDE:
        candidates.append(Path(LOUPE_DIR_OVERRIDE).expanduser().resolve())
    candidates.extend([
        DATA / "Loupe_categories",
        DATA / "yakubovsky_repo" / "Loupe_categories",
        DATA / "yakubovsky_loupe",  # legacy local layout
    ])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    formatted = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Yakubovsky Loupe annotations were not found. Expected one of:\n  - "
        f"{formatted}\nSet SGD_LOUPE_DIR to an alternate Loupe_categories directory."
    )

# A7 knobs.
N_JOBS = 8  # one per LHD sample
N_PERM_C5 = 200

# B2 anatomical-landmark anchors for Variant C / test markers.
BILE_DUCT_MARKERS = ("KRT7", "KRT19", "SOX9")
CENTRAL_VEIN_MARKERS = ("RSPO3", "AXIN2")
TEST_MARKERS = tuple(PORTAL_MARKERS) + tuple(CENTRAL_MARKERS)


# ===========================================================================
# build_lipid_supplement — encode the Yakubovsky GitHub lipid CSVs
# ===========================================================================

def read_barcodes(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set(df.iloc[:, 0].astype(str).tolist())


def build_lipid_supplement() -> None:
    """Build lipid_supplement.json from the Yakubovsky GitHub lipid CSVs.

    The CSVs are categorical (per-spot in-lipid-zone indicator), not a
    continuous lipid percentage. We encode them as binary {0.0, 1.0} per
    spot and skip spots in lipid_discard.csv (NaN)."""
    loupe = resolve_loupe_dir()
    print(f"Loading {VISIUM_H5AD} (obs only); Loupe annotations: {loupe}")
    adata = sc.read_h5ad(VISIUM_H5AD, backed="r")
    obs = adata.obs[["sample_id", "barcode"]].copy()
    adata.file.close()

    supplement: dict[str, dict[str, float]] = {}
    summary: dict[str, dict] = {}

    for sample, lipid_csv, discard_csv in [
        ("M1", loupe / "M1_lipid_zones.csv", None),
        ("P6", loupe / "P6_lipid_zones.csv", loupe / "P6_lipid_discard.csv"),
    ]:
        if not lipid_csv.is_file():
            raise FileNotFoundError(
                f"Required lipid annotation is missing: {lipid_csv}. "
                "Refusing to encode every spot as non-lipid."
            )
        if discard_csv is not None and not discard_csv.is_file():
            raise FileNotFoundError(f"Required lipid discard annotation is missing: {discard_csv}")
        sample_barcodes = obs.loc[obs["sample_id"].astype(str) == sample, "barcode"].astype(str).tolist()
        zones = read_barcodes(lipid_csv)
        discards = read_barcodes(discard_csv) if discard_csv else set()
        rec: dict[str, float] = {}
        n_zone = n_zero = n_skip = 0
        for b in sample_barcodes:
            if b in discards:
                n_skip += 1
                continue  # skip → C2 ignores this spot
            if b in zones:
                rec[b] = 1.0
                n_zone += 1
            else:
                rec[b] = 0.0
                n_zero += 1
        supplement[sample] = rec
        summary[sample] = {
            "n_total_spots_in_anndata": len(sample_barcodes),
            "n_in_lipid_zone": n_zone,
            "n_outside_lipid_zone": n_zero,
            "n_skipped_lipid_discard": n_skip,
            "fraction_in_lipid_zone": n_zone / max(1, n_zone + n_zero),
        }
        print(f"  {sample}: {n_zone} lipid / {n_zero} non-lipid / {n_skip} discarded "
              f"(of {len(sample_barcodes)} spots)")

    # Mark M2 and M3 as not_available so the file documents the gap explicitly.
    payload = {
        "schema": "binary_in_lipid_zone",
        "encoding": "1.0 = spot in lipid_zones.csv, 0.0 = otherwise, absent = skipped (lipid_discard.csv)",
        "source": "https://github.com/OranYak/Human-liver/tree/main/Loupe_categories",
        "caveat": (
            "Binary per-spot annotation, not continuous lipid percentage. "
            "Pearson r is point-biserial; §14 #9 r<-0.5 threshold was set "
            "for continuous lipid_pct and is conservative for this encoding."
        ),
        "samples_with_data": ["M1", "P6"],
        "samples_not_available": ["M2", "M3"],
        "summary": summary,
        # The actual per-spot dict that A7 reads:
        **supplement,
    }
    LIPID_SUPP.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {LIPID_SUPP}")


# ===========================================================================
# A6 + A6.5 — preliminary axis (§5.1) and fibrotic-spot mask (§4.5)
# ===========================================================================

def run_axis_fibrotic() -> None:
    """A6 — compute the §5.1 preliminary axis on LHD and the §4.5 fibrotic
    mask on all samples; write both into visium_human.h5ad in place."""
    print(f"[A6] Loading {VISIUM_H5AD} ({file_size_mb(VISIUM_H5AD):.1f} MB)...")
    adata = sc.read_h5ad(VISIUM_H5AD)

    # --- Preliminary axis on LHD only ---
    lhd_mask = adata.obs["sample_id"].astype(str).isin(LHD_SAMPLES).to_numpy()
    print(f"[A6] Computing preliminary axis on {lhd_mask.sum()} LHD spots "
          f"({adata.obs.loc[lhd_mask, 'sample_id'].nunique()} samples)...")
    sub = adata[lhd_mask].copy()
    diag = compute_preliminary_axis(sub, sample_col="sample_id")

    # Write back to full adata; non-LHD spots stay NaN.
    s_full = np.full(adata.n_obs, np.nan)
    s_alt_full = np.full(adata.n_obs, np.nan)
    s_full[lhd_mask] = sub.obs["s_preliminary"].to_numpy()
    s_alt_full[lhd_mask] = sub.obs["s_alt_preliminary"].to_numpy()
    adata.obs["s_preliminary"] = s_full
    adata.obs["s_alt_preliminary"] = s_alt_full

    # Per-sample summary.
    print("[A6] Marker-direction agreement per LHD sample:")
    for sid, d in diag.items():
        if d.get("status") != "ok":
            print(f"[A6]   {sid}: {d.get('status')}")
            continue
        print(f"[A6]   {sid}: {d['marker_correct']}/{d['marker_total']} "
              f"flipped={d['orientation_flipped']} n={d['n_spots']}")

    # --- A6.5 fibrotic mask, on all samples ---
    print("[A6.5] Deriving fibrotic-spot mask (not applied)...")
    fractions = derive_fibrotic_mask(adata, sample_col="sample_id")
    for sid, f in sorted(fractions.items()):
        flag = " (>20% — inspect)" if (not np.isnan(f) and f > 0.20) else ""
        print(f"[A6.5]   {sid}: {100*f:.1f}% flagged{flag}")

    # Save back.
    adata.write_h5ad(VISIUM_H5AD, compression="gzip")
    print(f"[A6] Updated {VISIUM_H5AD} ({file_size_mb(VISIUM_H5AD):.1f} MB)")

    AXIS_DIAG.write_text(json.dumps({
        "axis": diag,
        "fibrotic_mask_fraction": fractions,
    }, indent=2, default=str))
    print(f"[A6] Wrote {AXIS_DIAG}")

    if INVENTORY.exists():
        inv = json.loads(INVENTORY.read_text())
        inv["axis"] = {
            "marker_summary": {sid: {"correct": d.get("marker_correct"),
                                      "total": d.get("marker_total"),
                                      "flipped": d.get("orientation_flipped")}
                                for sid, d in diag.items() if d.get("status") == "ok"},
            "fibrotic_fraction": fractions,
        }
        INVENTORY.write_text(json.dumps(inv, indent=2, default=str))


# ===========================================================================
# A7 — confound diagnostics on LHD Visium (4 of 5 evaluable)
# ===========================================================================

def stable_seed(name: str) -> int:
    """Process-stable seed for the C5 permutation null (replaces Python's hash())."""
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


def lipid_supplement() -> dict:
    """Optional per-spot or per-sample lipid override file."""
    if LIPID_SUPP.exists():
        return json.loads(LIPID_SUPP.read_text())
    return {}


def per_sample_diagnostics(sample_id: str, adata_sub: ad.AnnData,
                            lipid_lookup: dict | None,
                            run_axis_diagnostics: bool) -> dict:
    """
    Compute C2 always (when lipid data exists). C1, C3, C5 only when
    ``run_axis_diagnostics`` is True (i.e. the sample has a valid
    ``s_preliminary`` axis — LHD samples). For non-LHD samples those
    keys are recorded as ``not_evaluated`` rather than NaN-stuffed,
    so A8's tables can skip them cleanly.
    """
    out: dict = {"sample_id": sample_id,
                  "n_spots": int(adata_sub.n_obs),
                  "axis_evaluated": run_axis_diagnostics}

    log_umi = np.log1p(total_umi(adata_sub))

    if run_axis_diagnostics:
        s = adata_sub.obs["s_preliminary"].to_numpy()
        valid = ~np.isnan(s)

        # C1 — r(s, log UMI)
        if valid.sum() >= 3:
            r_c1, p_c1 = pearsonr(s[valid], log_umi[valid])
        else:
            r_c1, p_c1 = float("nan"), float("nan")
        out["C1"] = {"r": float(r_c1), "p": float(p_c1),
                      "trigger_03": bool(abs(r_c1) > 0.3) if not np.isnan(r_c1) else None,
                      "trigger_05": bool(abs(r_c1) > 0.5) if not np.isnan(r_c1) else None}

        # C3 — mt-fraction by s-quintile. Pericentral = highest-s quintile after orientation.
        mt = adata_sub.obs["pct_counts_mt"].to_numpy()
        quint = (pd.qcut(s[valid], q=5, labels=False, duplicates="drop")
                 if valid.sum() >= 10 else None)
        quintiles = []
        if quint is not None:
            mt_v = mt[valid]
            for q in range(int(quint.max()) + 1):
                mask_q = quint == q
                quintiles.append({
                    "quintile": int(q),
                    "n": int(mask_q.sum()),
                    "mt_median": float(np.nanmedian(mt_v[mask_q])),
                    "mt_mean": float(np.nanmean(mt_v[mask_q])),
                    "frac_above_25": float(np.nanmean(mt_v[mask_q] > 25)),
                    "frac_above_40": float(np.nanmean(mt_v[mask_q] > 40)),
                    "frac_above_50": float(np.nanmean(mt_v[mask_q] > 50)),
                })
        # Pericentral quintile = highest q present.
        peri_q = quintiles[-1] if quintiles else None
        max_above_25 = max((q["frac_above_25"] for q in quintiles), default=float("nan"))
        peri_above_25 = peri_q["frac_above_25"] if peri_q else float("nan")
        out["C3"] = {
            "quintiles": quintiles,
            "max_quintile_frac_above_25": float(max_above_25),
            "pericentral_frac_above_25": float(peri_above_25),
            "trigger_max": (bool(max_above_25 > 0.15)
                            if not np.isnan(max_above_25) else None),
            "trigger_pericentral": (bool(peri_above_25 > 0.15)
                                     if not np.isnan(peri_above_25) else None),
            "mt_prefix": adata_sub.uns.get("mt_prefix"),
        }

        # C5 — slide-edge directional score.
        if "spatial" in adata_sub.obsm:
            c5 = c5_score(adata_sub, s_col="s_preliminary", n_perm=N_PERM_C5,
                          rng_seed=stable_seed(sample_id))
            c5["trigger"] = bool(c5.get("score", float("nan")) > 0.30
                                  and c5.get("p", 1.0) < 0.05)
            out["C5"] = c5
        else:
            out["C5"] = {"score": None, "trigger": None, "note": "no spatial coords"}
    else:
        out["C1"] = {"evaluable": False, "reason": "no s_preliminary"}
        out["C3"] = {"evaluable": False, "reason": "no s_preliminary"}
        out["C5"] = {"evaluable": False, "reason": "no s_preliminary"}

    # C2 — r(lipid_pct, log UMI). Independent of s; runs on any sample with lipid data.
    c2_state: dict = {"r": None, "p": None, "n": int(adata_sub.n_obs),
                       "lipid_source": None, "evaluable": False}
    if lipid_lookup:
        rec = lipid_lookup.get(sample_id)
        if isinstance(rec, dict):
            # Per-spot lipid keyed by raw barcode (obs["barcode"]); fall back to obs_names.
            keys = (adata_sub.obs["barcode"].astype(str).to_numpy()
                    if "barcode" in adata_sub.obs.columns
                    else adata_sub.obs_names.to_numpy())
            lipid = np.array([rec.get(b, np.nan) for b in keys], dtype=float)
            ok = ~np.isnan(lipid) & ~np.isnan(log_umi)
            if ok.sum() >= 3:
                r, p = pearsonr(lipid[ok], log_umi[ok])
                c2_state.update({"r": float(r), "p": float(p), "n": int(ok.sum()),
                                  "lipid_source": "per_spot", "evaluable": True})
        elif isinstance(rec, (int, float)):
            c2_state.update({"lipid_value_per_sample": float(rec),
                              "lipid_source": "per_sample", "evaluable": False,
                              "note": "per-sample lipid; reframed to between-sample at A8"})
    out["C2"] = c2_state

    return out


def run_diagnostics() -> None:
    """A7 — run the C1/C2/C3/C5 confound diagnostics per LHD sample and write
    confound_diagnostics.json."""
    print(f"[A7] Loading {VISIUM_H5AD} ({file_size_mb(VISIUM_H5AD):.1f} MB)...")
    adata = sc.read_h5ad(VISIUM_H5AD)

    lhd_samples_present = sorted(set(adata.obs["sample_id"].astype(str).unique()) & set(LHD_SAMPLES))
    steat_present = sorted(set(adata.obs["sample_id"].astype(str).unique()) & set(STEATOTIC_SAMPLES))
    print(f"[A7] LHD samples: {lhd_samples_present}")
    print(f"[A7] Steatotic samples (for C2): {steat_present}")

    lipids = lipid_supplement()
    if lipids:
        print(f"[A7] lipid_supplement.json present: {list(lipids.keys())}")
    else:
        print("[A7] No lipid_supplement.json — C2 marked not_evaluable for non-steatotic samples")

    # Build per-sample views (must materialise before joblib because views over
    # a single shared AnnData and forking can be flaky with large objects).
    # axis_evaluated=True only for LHD samples; non-LHD steatotic samples (P6)
    # contribute only C2 — their C1/C3/C5 keys are recorded as not_evaluated.
    plan = []
    seen = set()
    for sid in lhd_samples_present + [s for s in steat_present if s not in lhd_samples_present]:
        if sid in seen:
            continue
        seen.add(sid)
        sub = adata[adata.obs["sample_id"].astype(str) == sid].copy()
        run_axis = sid in LHD_SAMPLES
        plan.append((sid, sub, run_axis))

    print(f"[A7] Running diagnostics on {len(plan)} samples (n_jobs={N_JOBS})...")
    results = Parallel(n_jobs=min(N_JOBS, len(plan)), backend="loky", verbose=5)(
        delayed(per_sample_diagnostics)(sid, sub, lipids, run_axis)
        for sid, sub, run_axis in plan
    )

    # Aggregate — C1 and C5 only over axis-evaluated samples.
    cohort_C1 = [r["C1"]["r"] for r in results
                  if r.get("axis_evaluated") and r["C1"].get("r") is not None
                  and not np.isnan(r["C1"]["r"])]
    summary = {
        "lhd_samples_evaluated": lhd_samples_present,
        "steatotic_samples_evaluated": steat_present,
        "C1_cohort": {
            "abs_r_mean": float(np.mean(np.abs(cohort_C1))) if cohort_C1 else None,
            "abs_r_max":  float(np.max(np.abs(cohort_C1))) if cohort_C1 else None,
            "any_above_03": bool(any(abs(r) > 0.3 for r in cohort_C1)),
            "any_above_05": bool(any(abs(r) > 0.5 for r in cohort_C1)),
        },
        "C4_status": "deferred_stage_D",
        "lipid_supplement_present": bool(lipids),
    }

    payload = {
        "schema_version": "stageA-v1",
        "n_perm_C5": N_PERM_C5,
        "summary": summary,
        "per_sample": {r["sample_id"]: r for r in results},
    }
    CONFOUND_OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[A7] Wrote {CONFOUND_OUT} ({file_size_mb(CONFOUND_OUT):.2f} MB)")
    print(f"[A7] Summary: {json.dumps(summary, indent=2, default=str)}")


# ===========================================================================
# B2 — axis variants on the LHD cohort
# ===========================================================================

def run_axis_variants() -> None:
    """B2 — re-validate the §5.1 primary axis, build the §5.2 ratio variant and
    the §5.4 Variant-C landmark axis, write them into visium_human.h5ad.

    Note: ``obs.s_v_B`` is NOT written here (step 07 is its sole writer); the
    earlier B2 all-NaN ``s_v_B`` placeholder write is intentionally omitted.
    """
    print(f"[B2] Loading {VISIUM_H5AD}")
    adata = sc.read_h5ad(VISIUM_H5AD)
    lhd = adata.obs["sample_id"].astype(str).isin(LHD_SAMPLES).to_numpy()
    print(f"[B2]   LHD spots: {lhd.sum()} (of {adata.n_obs})")

    # --- s and s_alt (§5.1, §5.2) — re-validation pass on LHD only.
    s = np.full(adata.n_obs, np.nan)
    s_alt = np.full(adata.n_obs, np.nan)
    sub = adata[lhd].copy()
    s_lhd, s_alt_lhd, diag_primary = build_score_axis(
        sub, PORTAL_MARKERS, CENTRAL_MARKERS)
    s[lhd] = s_lhd
    s_alt[lhd] = s_alt_lhd

    # --- s_v_B (§5.4 Variant B) — only if Yakubovsky panel is available.
    s_v_B = np.full(adata.n_obs, np.nan)
    diag_v_B: dict = {}
    yakub_status = "not_loaded"
    panel_path = None
    zone_index_path = None
    if YAKUB.exists():
        yk = json.loads(YAKUB.read_text())
        yakub_status = yk.get("status", "not_evaluable")
        panel_path = yk.get("panel_1711_path")
        zone_index_path = yk.get("zone_index_path")
        if panel_path:
            print(f"[B2] Yakubovsky panel candidate: {panel_path}")
            # Defer the actual panel load: Stage B does NOT use snRNA-derived
            # fallback. Only when we confirm the file is the published 1711-
            # gene panel (manual inspection step) do we wire this in.
            diag_v_B = {"status": "candidate_present_pending_manual_inspection",
                         "panel_path": panel_path}
        else:
            diag_v_B = {"status": "not_evaluable",
                         "reason": "no panel candidate located"}
    else:
        diag_v_B = {"status": "not_evaluable",
                     "reason": "yakubovsky_reference.json absent (run B1 first)"}

    # --- s_pub (§5.4 Variant A) — Yakubovsky published per-spot zone index.
    # Like Variant B, requires a manual inspection step before being wired in.
    s_pub = np.full(adata.n_obs, np.nan)
    diag_v_A: dict = ({"status": "not_evaluable",
                        "reason": "no per-spot zone index in B1 output"}
                       if zone_index_path is None else
                       {"status": "candidate_present_pending_manual_inspection",
                        "zone_index_path": zone_index_path})

    # --- s_v_C (§5.4 Variant C) — anatomical landmarks; exploratory.
    s_v_C = np.full(adata.n_obs, np.nan)
    sv_C_lhd, diag_v_C = build_landmark_axis(sub)
    s_v_C[lhd] = sv_C_lhd

    # Write back. obs.s_v_B is NOT written here — step 07 is its sole writer
    # and the column must be genuinely absent when step 06 runs.
    adata.obs["s"] = s
    adata.obs["s_alt"] = s_alt
    adata.obs["s_pub"] = s_pub
    adata.obs["s_v_C"] = s_v_C
    adata.write_h5ad(VISIUM_H5AD, compression="gzip")
    print(f"[B2] Updated {VISIUM_H5AD} with axis columns: s, s_alt, s_pub, s_v_C")

    # --- Cross-axis sensitivity table on LHD spots only (§5.2).
    cross: dict = {}
    pairs = [("s", "s_alt"), ("s", "s_v_C"), ("s_alt", "s_v_C")]
    for a, b in pairs:
        va, vb = adata.obs[a].to_numpy(), adata.obs[b].to_numpy()
        ok = ~np.isnan(va) & ~np.isnan(vb)
        if ok.sum() >= 10:
            r, _ = pearsonr(va[ok], vb[ok])
            cross[f"{a}_vs_{b}"] = float(r)

    summary = {
        "primary_diagnostics": diag_primary,
        "v_C_diagnostics": diag_v_C,
        "v_A_status": diag_v_A,
        "v_B_status": diag_v_B,
        "yakubovsky_panel_status": yakub_status,
        "cross_axis_pearson": cross,
        "n_lhd_spots": int(lhd.sum()),
    }
    AXIS_VARIANTS_OUT.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[B2] Wrote {AXIS_VARIANTS_OUT}")
    for ab, r in cross.items():
        print(f"[B2]   r({ab}) = {r:.3f}")
    print(f"[B2]   s_pub status: {diag_v_A.get('status')}")
    print(f"[B2]   s_v_B status: {diag_v_B.get('status')}")


def main() -> None:
    # Order: lipid supplement (reads obs of the loaded h5ad) → preliminary axis
    # + fibrotic mask → confound diagnostics (reads s_preliminary + the lipid
    # supplement) → axis variants (re-validates the primary axis).
    build_lipid_supplement()
    run_axis_fibrotic()
    run_diagnostics()
    run_axis_variants()


if __name__ == "__main__":
    main()
