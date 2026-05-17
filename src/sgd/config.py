"""Repo-root-relative paths and cohort-taxonomy constants.

This module is the single place that resolves filesystem locations. Every
path is derived from this file's own location, so the repository is
portable: there are no hard-coded absolute paths anywhere in ``src/sgd``
or ``scripts``. Legacy code hard-coded ``PROJ``/``DATA``/``RAW`` in four
per-stage ``utils.py`` files and an absolute snRNA path in D13; all of
those are replaced by the constants defined here.
"""

from __future__ import annotations

from pathlib import Path

# Repository root: src/sgd/config.py -> parents[0]=sgd, [1]=src, [2]=repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Top-level directories.
DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"

# Zenodo DOI for the atlas (raw Visium / Visium HD / snRNA-seq).
ZENODO_DOI = "10.5281/zenodo.17735506"

# snRNA-seq reference (cell2location reference for step 10). Legacy D13
# hard-coded /home/tmk-gpu/Desktop/Tomas_I/.../snRNAseq.h5ad — now under data/.
SNRNA_PATH = DATA / "snRNAseq.h5ad"

# Yakubovsky supplementary tables. Legacy code placed them under
# data/41586_2026_10377_MOESM1_ESM/2025-01-01424E-s1/; that layout is kept.
SUPP_DIR = DATA / "41586_2026_10377_MOESM1_ESM" / "2025-01-01424E-s1"
SUPP_T2 = SUPP_DIR / "supplementary_table_2.xlsx"
SUPP_T4 = SUPP_DIR / "supplementary_table_4.xlsx"
SUPP_T6 = SUPP_DIR / "supplementary_table_6.xlsx"
SUPP_T8 = SUPP_DIR / "supplementary_table_8.xlsx"
SUPP_T10 = SUPP_DIR / "supplementary_table_10.xlsx"
SUPP_T11 = SUPP_DIR / "supplementary_table_11.xlsx"

# ---------------------------------------------------------------------------
# Cohort taxonomy (§3)
# ---------------------------------------------------------------------------

# Living-healthy-donor cohort.
LHD_SAMPLES = ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")

# Patient cohort.
P_SAMPLES = ("P2", "P3", "P6", "P7", "P14", "P17", "P18", "P21")

# Steatotic samples (M1 + M2 + M3 are LHD; P6 is P-cohort).
STEATOTIC_SAMPLES = ("M1", "M2", "M3", "P6")

# Steatotic donors — alias of STEATOTIC_SAMPLES; legacy Stage D used this
# name (M1 + M2 + M3 are LHD; P6 is P-cohort).
STEATOTIC_DONORS = ("M1", "M2", "M3", "P6")
