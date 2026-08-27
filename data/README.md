# `data/` - inputs (not committed)

Everything in this directory is gitignored except this README. The
replication scripts read their raw inputs from here; download them as
described below before running `scripts/01_inventory.py`.

## 1. Atlas - Zenodo

Raw Visium (16 samples), Visium HD (M1 / M2 / M6) and the integrated
snRNA-seq reference are published on Zenodo:

> **DOI: [10.5281/zenodo.17735506](https://doi.org/10.5281/zenodo.17735506)**

Download the archive and unpack it into `data/` so the raw Visium h5
files, tissue-position CSVs, the Visium HD h5ad files, and the snRNA-seq
h5ad land at the paths the scripts expect (see the layout below).

The pre-computed cell2location output `obs_hep_fraction_c2l.parquet` is
also published as a Zenodo ancillary file. `scripts/10_cell2location.py`
is GPU-bound and takes ~2 h; a CPU-only replicator can drop the cached
parquet into `data/` and still run step 10's second pass and figure S4.

## 2. Yakubovsky supplementary tables - paper SI

Five supplementary tables from the Yakubovsky *et al.* human-liver paper
are read by the calibration / cross-pipeline steps. Download the
supplementary information bundle (`41586_2026_10377_MOESM1_ESM`) from the
paper and place the `.xlsx` files under
`data/41586_2026_10377_MOESM1_ESM/2025-01-01424E-s1/`:

| File | Used by |
|---|---|
| `supplementary_table_2.xlsx` | step 07 (Variant B), step 08 (cross-pipeline) |
| `supplementary_table_4.xlsx` | step 08 (cross-platform HD calibration) |
| `supplementary_table_6.xlsx` | step 13 (LHD-vs-P calibration) |
| `supplementary_table_10.xlsx` | step 11 (lipid join) |
| `supplementary_table_11.xlsx` | step 13 (T11 calibration) |

## 3. Yakubovsky GitHub lipid annotations

`scripts/01_inventory.py` also reads the `Loupe_categories/` lipid CSVs
from the Yakubovsky GitHub repository (`OranYak/Human-liver`). Clone or
download that directory into `data/Loupe_categories/`, or set
`SGD_LOUPE_DIR=/absolute/path/to/Loupe_categories`.

## 4. Moor 2018 intestinal crypt-villus (second-tissue validation)

`scripts/crypt/` validates the framework on a second tissue, the mouse
intestinal crypt-villus axis. It reads the laser-capture-microdissection
zonation reconstruction of Moor *et al.* 2018, published on Zenodo:

> **DOI: [10.5281/zenodo.3403670](https://doi.org/10.5281/zenodo.3403670)**

Download `table_D_zonation_reconstruction.tsv` (per-zone mean expression for
27,998 genes across seven crypt-villus zones, with a per-gene zone-dependence
*q*-value) and place it directly under `data/`. It is the only input the
`scripts/crypt/` steps need, and is independent of the liver atlas above.

## 5. Expected layout

```
data/
├── README.md                          (this file - the only committed item)
├── snRNAseq.h5ad                       integrated snRNA-seq reference (Zenodo)
├── obs_hep_fraction_c2l.parquet        cached cell2location output (Zenodo ancillary)
├── <raw Visium h5 + tissue_positions CSVs>     (Zenodo)
├── <Visium HD h5ad files: M1 / M2 / M6>        (Zenodo)
├── table_D_zonation_reconstruction.tsv  Moor 2018 crypt-villus zonation (Zenodo)
├── Loupe_categories/                   Yakubovsky lipid CSVs (GitHub)
└── 41586_2026_10377_MOESM1_ESM/
    └── 2025-01-01424E-s1/
        ├── supplementary_table_2.xlsx
        ├── supplementary_table_4.xlsx
        ├── supplementary_table_6.xlsx
        ├── supplementary_table_10.xlsx
        └── supplementary_table_11.xlsx
```

All of these paths are resolved by `src/sgd/config.py` relative to the
repository root. `SGD_DATA_DIR`, `SGD_RESULTS_DIR`, and `SGD_FIGURES_DIR` can
override the three top-level directories without editing code.
