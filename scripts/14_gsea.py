"""
14 — GSEA-style pathway enrichment on the ranked β₂ list.

Absorbs legacy ``stage_D/D12_gsea.py`` verbatim (only imports + paths
rewritten). Two analyses:

  - Prerank GSEA on the full ranked β₂ list across all 66 strict-panel genes,
    against MSigDB Hallmark, KEGG_2021_Human, and Reactome_2022.
  - Enrichr over-representation (ORA) on the called gene set (robust_1234 +
    donor_sensitive_12 + weak_1) against the 66-gene panel background.

Note on n=66: small for GSEA. Results are reported with caveats — pathway
coherence on a small panel is suggestive, not definitive.

Reads:  RESULTS/steatosis_call_criteria.parquet.
Writes: RESULTS/d12_gsea_summary.json.
Feeds:  pathway enrichment §4.10 — R4.8/R4.9 (Reactome prerank), R5.7
        (Enrichr ORA on the called genes); Fig 4D.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgd import config
from sgd.config import RESULTS

warnings.filterwarnings("ignore")
np.random.seed(0)

CALL = RESULTS / "steatosis_call_criteria.parquet"
OUT = RESULTS / "d12_gsea_summary.json"

LIBRARIES = ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "Reactome_2022"]
TOP_K_REPORT = 15


def main() -> None:
    print(f"[D12] Loading {CALL}")
    df = pd.read_parquet(CALL)
    df = df[df["cohort_status"] == "ok"].copy()
    print(f"[D12]   {len(df)} ok genes (strict-66 panel)")

    # Rank by β₂ (signed). gseapy prerank takes a 2-column DataFrame.
    rank = df[["gene", "beta_2"]].dropna().drop_duplicates(subset="gene")
    rank = rank.sort_values("beta_2", ascending=False).reset_index(drop=True)
    print(f"[D12]   ranked {len(rank)} genes; top: {rank.head(3).to_dict('records')}; "
          f"bottom: {rank.tail(3).to_dict('records')}")

    # --- 1) Prerank GSEA on each library ---
    library_results: dict = {}
    for lib in LIBRARIES:
        print(f"[D12] Prerank GSEA on {lib}...")
        try:
            pre_res = gp.prerank(
                rnk=rank,
                gene_sets=lib,
                threads=4,
                min_size=3,  # tiny panel → permit very small gene sets
                max_size=2000,
                permutation_num=1000,
                outdir=None,
                seed=0,
                verbose=False,
            )
            res_df = pre_res.res2d.copy()
            # gseapy uses 'NES','NOM p-val','FDR q-val','Term'
            cols = list(res_df.columns)
            n_total = int(len(res_df))
            sig_q = int((res_df.get("FDR q-val", pd.Series([1]*n_total)) < 0.25).sum())
            sig_p = int((res_df.get("NOM p-val", pd.Series([1]*n_total)) < 0.05).sum())
            top = res_df.head(TOP_K_REPORT).to_dict("records")
            library_results[lib] = {
                "n_terms": n_total,
                "n_sig_FDR_q_lt_025": sig_q,
                "n_sig_NOM_p_lt_005": sig_p,
                "top_terms": top,
                "columns": cols,
            }
            print(f"[D12]   {lib}: {n_total} terms; "
                  f"FDR<0.25 = {sig_q}; nominal p<0.05 = {sig_p}")
        except Exception as e:
            library_results[lib] = {"status": "failed",
                                      "error": f"{type(e).__name__}: {str(e)[:100]}"}
            print(f"[D12]   {lib}: FAILED — {e}")

    # --- 2) Over-representation on weak_1 + robust_1234 (called genes) ---
    called = df[df["four_criteria_flag"].isin(
        ["robust_1234", "donor_sensitive_12", "weak_1"]
    )]["gene"].tolist()
    background = df["gene"].tolist()
    print(f"[D12] Enrichr over-rep: called set size {len(called)}, "
          f"background {len(background)}")
    enrichr_results: dict = {}
    if len(called) >= 3:
        for lib in LIBRARIES:
            try:
                en = gp.enrichr(
                    gene_list=called,
                    gene_sets=lib,
                    background=background,
                    organism="Human",
                    outdir=None,
                    no_plot=True,
                )
                en_df = en.results.copy()
                top = en_df.sort_values("Adjusted P-value").head(TOP_K_REPORT)
                enrichr_results[lib] = {
                    "n_terms": int(len(en_df)),
                    "n_sig_q_lt_025": int((en_df["Adjusted P-value"] < 0.25).sum()),
                    "top": top.to_dict("records"),
                }
                print(f"[D12]   {lib}: {len(en_df)} terms; "
                      f"q<0.25 = {enrichr_results[lib]['n_sig_q_lt_025']}")
            except Exception as e:
                enrichr_results[lib] = {"status": "failed",
                                          "error": f"{type(e).__name__}: {str(e)[:100]}"}

    out = {
        "panel_size": int(len(rank)),
        "panel_caveat": (
            "Strict-66 panel is small for GSEA. Pathway-level coherence here "
            "is suggestive, not definitive. Min set size 3, max 2000. Results "
            "should be interpreted as 'do the called genes cluster on coherent "
            "pathways' rather than 'these pathways are reliably enriched'."
        ),
        "ranked_top_5": rank.head(5).to_dict("records"),
        "ranked_bottom_5": rank.tail(5).to_dict("records"),
        "called_genes": called,
        "prerank_per_library": library_results,
        "enrichr_per_library": enrichr_results,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"[D12] Wrote {OUT}")


if __name__ == "__main__":
    main()
