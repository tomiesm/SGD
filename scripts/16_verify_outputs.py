"""Fail-fast verification of the regenerated manuscript artifact set.

Run after analysis steps 01--15 and the figure-rendering scripts.  This is not
an analysis step: it checks cohort sizes, panel sizes, corrected stability
metadata, robustness-table coverage, and the complete figure inventory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import anndata as ad
import pandas as pd

from sgd.config import FIGURES, RESULTS
from sgd.gradient import N_BINS_DEFAULT


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(name: str) -> dict:
    path = RESULTS / name
    require(path.is_file(), f"missing result: {path}")
    return json.loads(path.read_text())


def main() -> None:
    visium_path = RESULTS / "visium_human.h5ad"
    require(visium_path.is_file(), f"missing result: {visium_path}")
    visium = ad.read_h5ad(visium_path, backed="r")
    require(visium.n_obs == 47_828, f"expected 47,828 Visium spots, found {visium.n_obs}")
    require("s" in visium.obs, "obs.s is absent from the Visium artifact")
    visium.file.close()

    cross = read_json("cross_donor_correlation.json")
    require(cross["gene_set_size"] == 66, "strict cross-donor panel is not 66 genes")
    require(
        cross["relaxed_panel"]["gene_set_size"] == 15_192,
        "relaxed cross-donor panel is not 15,192 genes",
    )

    cross_pipeline = read_json("stage_B_cross_pipeline_supp_table_2.json")
    cross_genes = pd.read_parquet(
        RESULTS / "stage_B_cross_pipeline_supp_table_2_genes.parquet"
    )
    require(cross_pipeline.get("n_bins") == N_BINS_DEFAULT,
            "cross-pipeline comparison does not use the default bin count")
    require(len(cross_genes) == 3_207,
            f"cross-pipeline primary set has {len(cross_genes)} rather than 3,207 genes")
    require(cross_genes["gene"].is_unique, "cross-pipeline gene table has duplicates")
    require(int(cross_genes["sign_agreement"].sum()) == 2_446,
            "cross-pipeline primary sign-agreement count is not 2,446")

    sign_baseline = read_json("sign_correlation_baseline.json")
    require(sign_baseline.get("n_bins") == N_BINS_DEFAULT,
            "sign-correlation baseline metadata does not use the default bin count")

    stability = read_json("phenom_stability.json")
    require(stability.get("schema") == "stage_C_phenom_stability_v2", "old W schema")
    require("permutation" not in json.dumps(stability).lower(), "invalid permutation null remains")
    require(stability["donor_split_stability"]["n_unique_partitions"] == 35,
            "donor splits are not the 35 unique 4-vs-4 partitions")
    for donor in ("M2", "M6"):
        require(stability["platform_split_M2_M6"][donor]["status"] == "ok",
                f"{donor} W platform split did not complete")

    wild = pd.read_parquet(RESULTS / "steatosis_wild_bootstrap.parquet")
    require(len(wild) == 66, f"wild-bootstrap table has {len(wild)} rather than 66 genes")
    require(int((wild["q_webb"] < 0.05).sum()) == 0,
            "unexpected Webb-bootstrap FDR discoveries")
    axis = pd.read_parquet(RESULTS / "steatosis_axis_sensitivity.parquet")
    require(len(axis) == 66, f"axis-sensitivity table has {len(axis)} rather than 66 genes")
    status_columns = ("calling_status_baseline", "calling_status_glul_excluded")
    require(all(column in axis.columns for column in status_columns),
            "axis-sensitivity table lacks the public calling-status columns")
    public_statuses = {"robust_all_six", "donor_sensitive", "criterion_1_only", "null"}
    observed_statuses = {
        status
        for column in status_columns
        for status in axis[column].dropna().unique()
    }
    require(observed_statuses <= public_statuses,
            f"axis-sensitivity table contains non-canonical statuses: "
            f"{sorted(observed_statuses - public_statuses)}")

    expected_figures = [
        "fig1_axis_validation.pdf", "fig2_multimodal.pdf",
        "fig3_phenom_stability.pdf", "fig4_steatosis.pdf",
        "figS1_confounds.pdf", "figS2_tier1.pdf",
        "figS3_cross_pipeline.pdf", "figS4_celltype.pdf",
        "figS5_steatosis_ranked.pdf", "figS6_saa1_portal_response.pdf",
        "figS7_lhd_vs_p.pdf", "figS8_sensitivity_sweep.pdf",
        "figS9_c1_adjustment.pdf", "figS10_crypt_validation.pdf",
    ]
    missing = [name for name in expected_figures if not (FIGURES / name).is_file()]
    require(not missing, f"missing figure PDFs: {missing}")

    print(
        "[verify] PASS: 47,828 spots; strict/relaxed panels 66/15,192; "
        "35 unique donor splits; M2+M6 platform splits; 66-gene robustness "
        "tables; 4 main and 10 supplementary figures."
    )


if __name__ == "__main__":
    main()
