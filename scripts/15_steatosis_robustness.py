"""Steatosis robustness analyses required for complete manuscript replication.

This step makes two analyses that were previously run by hard-coded ad-hoc
scripts part of the public pipeline:

1. Exhaustive Webb and Rademacher wild-cluster bootstrap inference for every
   strict-panel gene.
2. Refit of every gene after excluding GLUL from the axis-construction markers.

Run after ``11_steatosis_model.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgd.axis import build_axis_for_donor
from sgd.config import RESULTS, STEATOTIC_DONORS
from sgd.panels import CENTRAL_MARKERS, PORTAL_MARKERS, analysis_gene_panel
from sgd.steatosis import apply_calling_rule, per_gene_full, steatotic_analysis_mask
from sgd.wild_cluster import (
    RADEMACHER_WEIGHTS,
    WEBB_WEIGHTS,
    build_steatosis_design,
    exhaustive_weight_grid,
    interaction_column,
    wild_cluster_bootstrap,
)


VISIUM = RESULTS / "visium_human.h5ad"
CALL_CRITERIA = RESULTS / "steatosis_call_criteria.parquet"
WILD_TABLE = RESULTS / "steatosis_wild_bootstrap.parquet"
WILD_CSV = RESULTS / "supplementary_table_wild_bootstrap.csv"
WILD_SUMMARY = RESULTS / "steatosis_wild_bootstrap_summary.json"
AXIS_TABLE = RESULTS / "steatosis_axis_sensitivity.parquet"
AXIS_CSV = RESULTS / "supplementary_table_axis_sensitivity.csv"
AXIS_SUMMARY = RESULTS / "steatosis_axis_sensitivity_summary.json"

N_JOBS = min(int(os.environ.get("SGD_N_JOBS", "8")), os.cpu_count() or 1)
JOBLIB_BACKEND = os.environ.get("SGD_JOBLIB_BACKEND", "loky")
HEADLINE_GENES = ("GLUL", "LYZ", "ORM1", "FBLN1", "IGKC", "MT1H", "SAA1", "TPSB2")


def _analysis_arrays(adata, axis: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return common steatosis metadata, expression matrix, and gene names."""
    panel = analysis_gene_panel(adata, mode="strict")
    mask = steatotic_analysis_mask(adata)
    sub = adata[mask, panel].copy()
    axis_sub = axis[mask]
    meta = pd.DataFrame({
        "s": axis_sub,
        "lipid_pct": sub.obs["lipid_pct"].to_numpy(dtype=float),
        "log_total_umi_raw": np.log1p(sub.obs["total_umi_raw"].to_numpy(dtype=float)),
        "donor_id": sub.obs["donor_id"].astype(str).to_numpy(),
    })
    keep = (~meta.isna().any(axis=1)).to_numpy()
    meta = meta.loc[keep].reset_index(drop=True)
    X = sub.X.toarray() if issparse(sub.X) else np.asarray(sub.X)
    return meta, X[keep], sub.var_names.astype(str).tolist()


def _bh(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(dtype=float)
    q = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    if finite.any():
        _, q[finite], _, _ = multipletests(p[finite], method="fdr_bh")
    return q


def _baseline_axis(adata) -> np.ndarray:
    """Return the production axis, rebuilding any steatotic donor not persisted."""
    axis = adata.obs["s"].to_numpy(dtype=float).copy()
    samples = adata.obs["sample_id"].astype(str).to_numpy()
    for donor in STEATOTIC_DONORS:
        mask = samples == donor
        if mask.any() and not np.isfinite(axis[mask]).any():
            donor_axis = build_axis_for_donor(adata, donor)
            axis[mask] = donor_axis[mask]
    return axis


def run_wild_bootstrap(adata) -> None:
    """Exhaustive restricted-null wild-cluster bootstrap for all 66 genes."""
    meta, X, genes = _analysis_arrays(adata, _baseline_axis(adata))
    unrestricted = build_steatosis_design(meta, restricted=False)
    restricted = build_steatosis_design(meta, restricted=True)
    interaction_idx = interaction_column(unrestricted)
    X1 = unrestricted.to_numpy(dtype=float)
    X0 = restricted.to_numpy(dtype=float)
    clusters = meta["donor_id"].astype(str).to_numpy()
    cluster_order = tuple(STEATOTIC_DONORS)
    webb_grid = exhaustive_weight_grid(WEBB_WEIGHTS, len(cluster_order))
    rademacher_grid = exhaustive_weight_grid(RADEMACHER_WEIGHTS, len(cluster_order))

    print(f"[D15] Wild bootstrap: {len(genes)} genes, {len(meta)} spots, "
          f"{len(webb_grid)} Webb assignments, n_jobs={N_JOBS}")

    def one(gene: str, values: np.ndarray) -> dict:
        result = wild_cluster_bootstrap(
            values, X1, X0, interaction_idx, clusters, cluster_order,
            webb_grid=webb_grid, rademacher_grid=rademacher_grid,
        )
        result["gene"] = gene
        return result

    rows = Parallel(n_jobs=N_JOBS, backend=JOBLIB_BACKEND, verbose=5)(
        delayed(one)(gene, X[:, i]) for i, gene in enumerate(genes)
    )
    table = pd.DataFrame(rows)
    table["q_cluster_robust_asymptotic"] = _bh(table["p_cluster_robust_asymptotic"])
    table["q_webb"] = _bh(table["p_webb"])
    table["q_rademacher"] = _bh(table["p_rademacher"])
    table = table[[
        "gene", "beta_2", "se_2", "t_observed",
        "p_cluster_robust_asymptotic", "q_cluster_robust_asymptotic",
        "p_webb", "q_webb", "p_rademacher", "q_rademacher",
        "n_webb_valid", "n_rademacher_valid", "status",
    ]].sort_values("gene").reset_index(drop=True)
    table.to_parquet(WILD_TABLE, compression="zstd", index=False)
    table.to_csv(WILD_CSV, index=False)

    headline = table[table["gene"].isin(HEADLINE_GENES)].set_index("gene")
    summary = {
        "n_genes": len(table),
        "n_spots": len(meta),
        "n_clusters": len(cluster_order),
        "cluster_order": list(cluster_order),
        "design_unrestricted": "g ~ bs(s, df=4) + lipid_pct + s:lipid_pct + log_total_umi_raw + C(donor_id)",
        "design_restricted": "g ~ bs(s, df=4) + lipid_pct + log_total_umi_raw + C(donor_id)",
        "bootstrap_type": "restricted-null wild cluster bootstrap, CR1 studentized",
        "webb_weights": WEBB_WEIGHTS.tolist(),
        "webb_assignments": len(webb_grid),
        "rademacher_weights": RADEMACHER_WEIGHTS.tolist(),
        "rademacher_assignments": len(rademacher_grid),
        "pvalue_convention": "two-sided (1 + count(|t*| >= |t_obs|)) / (1 + valid assignments)",
        "tie_handling": "algebraic ties included using numpy.isclose(rtol=1e-10, atol=1e-12)",
        "multiple_testing": "Benjamini-Hochberg separately for each p-value family over the strict panel",
        "n_q_webb_lt_005": int((table["q_webb"] < 0.05).sum()),
        "headline_genes": headline.to_dict(orient="index"),
    }
    WILD_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[D15] Wrote {WILD_TABLE}, {WILD_CSV}, and {WILD_SUMMARY}")


def run_axis_sensitivity(adata) -> None:
    """Refit the strict panel after excluding GLUL from axis construction."""
    central_without_glul = tuple(g for g in CENTRAL_MARKERS if g != "GLUL")
    excluded_axis = np.full(adata.n_obs, np.nan)
    samples = adata.obs["sample_id"].astype(str).to_numpy()
    for donor in STEATOTIC_DONORS:
        donor_axis = build_axis_for_donor(
            adata, donor, portal_markers=PORTAL_MARKERS,
            central_markers=central_without_glul,
        )
        mask = samples == donor
        excluded_axis[mask] = donor_axis[mask]

    meta, X, genes = _analysis_arrays(adata, excluded_axis)
    print(f"[D15] GLUL-excluded axis: {len(genes)} genes, {len(meta)} spots, n_jobs={N_JOBS}")
    rows = Parallel(n_jobs=N_JOBS, backend=JOBLIB_BACKEND, verbose=5)(
        delayed(per_gene_full)(i, gene, X[:, i], meta)
        for i, gene in enumerate(genes)
    )
    excluded = apply_calling_rule(pd.DataFrame(rows))
    baseline = pd.read_parquet(CALL_CRITERIA)
    base_cols = [
        "gene", "beta_2", "q_2", "criterion_1", "criterion_2",
        "criterion_3", "criterion_4", "criterion_5", "criterion_6",
        "calling_status",
    ]
    compare = baseline[base_cols].merge(
        excluded[base_cols], on="gene", suffixes=("_baseline", "_glul_excluded")
    )
    compare.to_parquet(AXIS_TABLE, compression="zstd", index=False)
    compare.to_csv(AXIS_CSV, index=False)

    x = compare["beta_2_baseline"].to_numpy(dtype=float)
    y = compare["beta_2_glul_excluded"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    headline = compare[compare["gene"].isin(HEADLINE_GENES)].set_index("gene")
    summary = {
        "excluded_marker": "GLUL",
        "portal_markers": list(PORTAL_MARKERS),
        "central_markers_baseline": list(CENTRAL_MARKERS),
        "central_markers_glul_excluded": list(central_without_glul),
        "n_genes": len(compare),
        "n_spots": len(meta),
        "beta_2_pearson_r": float(pearsonr(x[finite], y[finite]).statistic),
        "beta_2_spearman_rho": float(spearmanr(x[finite], y[finite]).statistic),
        "robust_genes_glul_excluded": excluded.loc[
            excluded["four_criteria_flag"] == "robust_1234", "gene"
        ].astype(str).tolist(),
        "headline_genes": headline.to_dict(orient="index"),
    }
    AXIS_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[D15] Wrote {AXIS_TABLE}, {AXIS_CSV}, and {AXIS_SUMMARY}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis", choices=("all", "wild-bootstrap", "axis-sensitivity"),
        default="all",
    )
    args = parser.parse_args()
    adata = sc.read_h5ad(VISIUM)
    if "lipid_pct" not in adata.obs or not np.isfinite(adata.obs["lipid_pct"]).any():
        raise RuntimeError("Step 15 requires the lipid join from step 11")
    if args.analysis in ("all", "wild-bootstrap"):
        run_wild_bootstrap(adata)
    if args.analysis in ("all", "axis-sensitivity"):
        run_axis_sensitivity(adata)


if __name__ == "__main__":
    main()
