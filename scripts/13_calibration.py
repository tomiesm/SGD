"""
13 — Supp Table 11 steatosis calibration + Supp Table 6 LHD-vs-P calibration.

This script absorbs two "calibration vs a Yakubovsky supplementary table"
analyses:

  - ``supp_t11_calibration()`` — REBUILT ORPHAN, re-authored from legacy
    ``stage_D/D4_supp_t11_calibration.py``. The single re-authoring change is
    **parametrising the gene set**: legacy D4 hard-coded the filter
    ``call_df["four_criteria_flag"].isin(["robust_1234",
    "donor_sensitive_12"])``; here that flag-set is the function parameter
    ``flags``. The ``approach_b`` high-minus-low-bin sign aggregation and the
    signed-magnitude Spearman are copied verbatim from D4. One run produces
    both: ``supp_t11_calibration.json`` (n=3 robust set,
    ``{robust_1234, donor_sensitive_12}``) and
    ``supp_t11_calibration_called8.json`` (the 8-gene called set, adding
    ``weak_1``; T11 carries 7 of the 8 — TPSB2 absent — so n=7).
    results_verification_weak1_extended.md §B fixes the n=7 output:
    6/7 sign agreement, ρ=0.821, p=0.023.

  - ``supp_t6_lhd_p()`` — absorbed verbatim from ``stage_D/D6_supp_t6_lhd_p.py``
    (not an orphan). The §9.9 LHD-vs-P portal-bias calibration against Supp
    Table 6.

Reads:  RESULTS/steatosis_call_criteria.parquet,
        RESULTS/steatosis_mixed_effects.parquet, RESULTS/visium_human.h5ad,
        config.SUPP_T11, config.SUPP_T6.
Writes: RESULTS/supp_t11_calibration.json, RESULTS/supp_t11_calibration_called8.json,
        RESULTS/supp_t6_lhd_p_calibration.json.
Feeds:  §4.3 / results-weak1 — R4.7 (T11 n=3, Fig 4C), R5.5/R5.6 (T11 n=7,
        SAA1 disagreement), §9.9 LHD-vs-P (supp S7).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgd import config
from sgd.config import RESULTS, SUPP_T6, SUPP_T11
from sgd.panels import (CENTRAL_MARKERS, KNOWN_LIVER_DIRECTION, PORTAL_MARKERS,
                        _gene_score, analysis_gene_panel, cohort_mask)
from sgd.gradient import (N_BINS_DEFAULT, SIGMA_BINS, per_gene_gradient,
                          quantile_bin_per_donor_pool)

warnings.filterwarnings("ignore")
np.random.seed(0)

LHD_SAMPLES = config.LHD_SAMPLES
P_SAMPLES = config.P_SAMPLES

VISIUM = RESULTS / "visium_human.h5ad"
CALL = RESULTS / "steatosis_call_criteria.parquet"
OUT_T11 = RESULTS / "supp_t11_calibration.json"
OUT_T11_CALLED8 = RESULTS / "supp_t11_calibration_called8.json"
OUT_T6 = RESULTS / "supp_t6_lhd_p_calibration.json"


# ---------------------------------------------------------------------------
# D4 (REBUILT) — Supp Table 11 calibration
# ---------------------------------------------------------------------------

def load_supp_t11() -> pd.DataFrame:
    """
    Read pericentral-by-lipid-bin sheet. Header on row 1 (row 0 is
    the description prose).
    """
    df = pd.read_excel(SUPP_T11, sheet_name="Visium spot lipid content", header=1)
    df.columns = [str(c).strip() for c in df.columns]
    if "Gene" in df.columns:
        df = df.set_index("Gene")
    elif "gene" in df.columns:
        df = df.set_index("gene")
    bin_cols = [c for c in df.columns if c.startswith("mean lipid bin")]
    df[bin_cols] = df[bin_cols].apply(pd.to_numeric, errors="coerce")
    return df


def _t11_calibration_n3(call_df: pd.DataFrame, flags: list[str]) -> dict:
    """Legacy-D4 calibration on the 3-gene robust set — re-authored only by
    lifting the ``four_criteria_flag.isin`` filter to the ``flags`` parameter.
    Output is byte-identical to legacy D4's ``supp_t11_calibration.json``."""
    surviving = call_df[call_df["four_criteria_flag"].isin(flags)].copy()
    print(f"[D4]   surviving ({' or '.join(flags)}) genes: {len(surviving)}")

    print(f"[D4] Loading {SUPP_T11}")
    t11 = load_supp_t11()
    bin_cols = [c for c in t11.columns if c.startswith("mean lipid bin")]
    print(f"[D4]   Supp T11 panel size: {len(t11)}; bin columns: {bin_cols}")

    # Direction of paper's lipid response: sign(mean of high-lipid bins - mean of low-lipid bins)
    high_cols = [c for c in bin_cols if c.endswith("4") or c.endswith("5")]
    low_cols = [c for c in bin_cols if c.endswith("0") or c.endswith("1")]
    if not high_cols or not low_cols:
        print(f"[D4] WARNING: could not identify high/low bin columns in {bin_cols}")
        return {
            "status": "bin_columns_not_identifiable",
            "available_columns": bin_cols,
        }

    pub_high = t11[high_cols].mean(axis=1)
    pub_low = t11[low_cols].mean(axis=1)
    pub_diff = pub_high - pub_low
    pub_diff = pub_diff[~pub_diff.index.duplicated()]
    pub_dir = np.sign(pub_diff)

    if len(surviving) == 0:
        print(f"[D4] No surviving genes; calibration skipped.")
        return {
            "status": "no_surviving_genes",
            "n_surviving": 0,
        }

    common = sorted(set(surviving["gene"]) & set(pub_dir.index))
    print(f"[D4]   genes overlapping with Supp T11: {len(common)}/{len(surviving)}")
    if not common:
        return {
            "status": "no_overlap",
            "n_surviving": int(len(surviving)),
            "n_overlap": 0,
        }

    our_beta = surviving.set_index("gene").loc[common, "beta_2"].to_numpy()
    our_sign = np.sign(our_beta)
    pub_sign_arr = pub_dir.reindex(common).to_numpy()
    pub_diff_arr = pub_diff.reindex(common).to_numpy()

    valid = (our_sign != 0) & (pub_sign_arr != 0) & ~np.isnan(pub_sign_arr)
    n_compared = int(valid.sum())
    n_agree = int((our_sign[valid] == pub_sign_arr[valid]).sum())
    pct = 100.0 * n_agree / max(1, n_compared)
    rho, _ = spearmanr(our_beta[valid], pub_diff_arr[valid])

    out = {
        "status": "ok",
        "n_surviving_genes": int(len(surviving)),
        "n_overlap_with_supp_t11": len(common),
        "n_compared_directional": n_compared,
        "n_agree": n_agree,
        "sign_agreement_pct": float(pct),
        "signed_magnitude_spearman": float(rho) if not np.isnan(rho) else None,
        "method": "approach_b_high_minus_low_bin_aggregation",
        "high_bin_cols": high_cols,
        "low_bin_cols": low_cols,
    }
    print(f"[D4] Sign agreement: {n_agree}/{n_compared} = {pct:.1f}%; "
          f"Spearman = {rho:.3f}")
    return out


def _t11_calibration_called8(call_df: pd.DataFrame, flags: list[str]) -> dict:
    """Calibration on the 8-gene called set. Same ``approach_b`` high-minus-
    low-bin sign aggregation as legacy D4 (byte-identical), augmented with the
    documented weak_1 extension: a per-gene table and the robust/weak subset
    breakdown plus the Spearman significance test
    (results_verification_weak1_extended.md §B)."""
    surviving = call_df[call_df["four_criteria_flag"].isin(flags)].copy()
    print(f"[D4]   called set ({' or '.join(flags)}) size: {len(surviving)}")

    t11 = load_supp_t11()
    bin_cols = [c for c in t11.columns if c.startswith("mean lipid bin")]
    high_cols = [c for c in bin_cols if c.endswith("4") or c.endswith("5")]
    low_cols = [c for c in bin_cols if c.endswith("0") or c.endswith("1")]

    pub_high = t11[high_cols].mean(axis=1)
    pub_low = t11[low_cols].mean(axis=1)
    pub_diff = pub_high - pub_low
    pub_diff = pub_diff[~pub_diff.index.duplicated()]

    flag_by_gene = call_df.set_index("gene")["four_criteria_flag"].to_dict()
    beta_by_gene = surviving.set_index("gene")["beta_2"].to_dict()

    per_gene: dict = {}
    for gene in sorted(surviving["gene"]):
        beta_2 = float(beta_by_gene[gene])
        if gene in pub_diff.index:
            t11_diff = float(pub_diff.loc[gene])
            framework_sign = int(np.sign(beta_2))
            t11_sign = int(np.sign(t11_diff))
            per_gene[gene] = {
                "flag": flag_by_gene.get(gene),
                "beta_2": beta_2,
                "t11_diff": t11_diff,
                "framework_sign": framework_sign,
                "t11_sign": t11_sign,
                "agree": int(framework_sign == t11_sign),
                "in_t11": True,
            }
        else:
            per_gene[gene] = {
                "flag": flag_by_gene.get(gene),
                "beta_2": beta_2,
                "in_t11": False,
            }

    in_t11 = {g: v for g, v in per_gene.items() if v["in_t11"]}
    genes_t11 = sorted(in_t11)
    beta_arr = np.array([in_t11[g]["beta_2"] for g in genes_t11])
    diff_arr = np.array([in_t11[g]["t11_diff"] for g in genes_t11])
    agree_arr = np.array([in_t11[g]["agree"] for g in genes_t11])
    n_agree = int(agree_arr.sum())
    pct = 100.0 * n_agree / max(1, len(genes_t11))
    rho, p = spearmanr(beta_arr, diff_arr)

    def _subset(flag_name: str) -> dict:
        sub = {g: v for g, v in in_t11.items() if v["flag"] == flag_name}
        sg = sorted(sub)
        b = np.array([sub[g]["beta_2"] for g in sg])
        d = np.array([sub[g]["t11_diff"] for g in sg])
        sub_rho, _ = spearmanr(b, d) if len(sg) >= 2 else (np.nan, np.nan)
        return {
            "n": len(sg),
            "n_agree": int(sum(sub[g]["agree"] for g in sg)),
            "spearman": float(sub_rho) if not np.isnan(sub_rho) else None,
        }

    out = {
        "status": "ok",
        "called_set_size": int(len(surviving)),
        "called_set_in_t11": len(genes_t11),
        "n_agree_overall": n_agree,
        "sign_agreement_pct_overall": float(pct),
        "signed_magnitude_spearman_overall": float(rho) if not np.isnan(rho) else None,
        "spearman_p": float(p) if not np.isnan(p) else None,
        "robust_subset": _subset("robust_1234"),
        "weak_subset": _subset("weak_1"),
        "method": "approach_b_high_minus_low_bin_aggregation",
        "high_bin_cols": high_cols,
        "low_bin_cols": low_cols,
        "per_gene": per_gene,
    }
    print(f"[D4] Called-set calibration: {n_agree}/{len(genes_t11)} = {pct:.1f}%; "
          f"Spearman = {rho:.3f} (p={p:.3f})")
    return out


def supp_t11_calibration() -> None:
    """D4 rebuild — produce both the n=3 robust and the n=8 called calibration."""
    print(f"[D4] Loading {CALL}")
    call_df = pd.read_parquet(CALL)

    # 3-gene robust set — byte-identical to legacy D4.
    out3 = _t11_calibration_n3(call_df, ["robust_1234", "donor_sensitive_12"])
    OUT_T11.write_text(json.dumps(out3, indent=2, default=str))
    print(f"[D4] Wrote {OUT_T11}")

    # 8-gene called set — lifts the flag-set to add weak_1.
    out8 = _t11_calibration_called8(call_df,
                                    ["robust_1234", "donor_sensitive_12", "weak_1"])
    OUT_T11_CALLED8.write_text(json.dumps(out8, indent=2, default=str))
    print(f"[D4] Wrote {OUT_T11_CALLED8}")


# ---------------------------------------------------------------------------
# D6 — §9.9 LHD-vs-P calibration with Supp Table 6 (absorbed verbatim)
# ---------------------------------------------------------------------------

def cohort_slope(adata: sc.AnnData, samples: list, panel_mask: np.ndarray
                  ) -> tuple[np.ndarray, list[str]]:
    spot = cohort_mask(adata, samples)
    if "fibrotic_spot" in adata.obs.columns:
        spot = spot & ~adata.obs["fibrotic_spot"].astype(bool).to_numpy()
    sub = adata[spot, panel_mask].copy()
    Gs, bc, _, _ = quantile_bin_per_donor_pool(
        sub, s_col="s", n_bins=N_BINS_DEFAULT, sigma=SIGMA_BINS)
    valid = ~np.isnan(bc)
    if valid.sum() < 3:
        return np.full(sub.n_vars, np.nan), list(sub.var_names)
    Gs = Gs[valid]; bc = bc[valid]
    return per_gene_gradient(Gs, bc)["slope"], list(sub.var_names)


def supp_t6_lhd_p() -> None:
    """D6 — LHD-vs-P portal-bias calibration against Supp Table 6."""
    print(f"[D6] Loading {VISIUM}")
    adata = sc.read_h5ad(VISIUM)
    panel = analysis_gene_panel(adata, mode="strict")
    print(f"[D6]   strict-66 panel size: {int(panel.sum())}")

    # P-cohort needs an axis built from its own marker subset; for calibration
    # against a per-gene LHD-vs-P difference we need a comparable axis on each.
    # Stage B built obs.s on LHD samples only. For this calibration we
    # build obs.s on P samples ad-hoc using the same §5.1 recipe.
    print("[D6] Building obs.s on P-cohort spots (ad-hoc, §5.1 recipe)...")
    samples_arr = adata.obs["sample_id"].astype(str).to_numpy()
    s_full = adata.obs["s"].to_numpy().copy() if "s" in adata.obs.columns else np.full(adata.n_obs, np.nan)
    for sample in P_SAMPLES:
        m = samples_arr == sample
        if m.sum() < 10:
            continue
        sub = adata[m]
        P = _gene_score(sub, PORTAL_MARKERS)
        C = _gene_score(sub, CENTRAL_MARKERS)
        if np.isnan(P).all() or np.isnan(C).all():
            continue
        Pz = (P - np.nanmean(P)) / (np.nanstd(P) + 1e-9)
        Cz = (C - np.nanmean(C)) / (np.nanstd(C) + 1e-9)
        delta = Cz - Pz
        ranks = pd.Series(delta).rank(method="average").to_numpy() - 1
        s_p = ranks / max(1, len(ranks) - 1)
        cy = _gene_score(sub, ("CYP2E1", "CYP1A2"))
        if not np.isnan(cy).all() and cy[s_p > 0.5].mean() < cy[s_p <= 0.5].mean():
            s_p = 1.0 - s_p
        s_full[m] = s_p
    adata.obs["s"] = s_full

    print(f"[D6] Per-gene gradient on LHD cohort...")
    slope_lhd, var_lhd = cohort_slope(adata, list(LHD_SAMPLES), panel)
    print(f"[D6] Per-gene gradient on P cohort...")
    slope_p, var_p = cohort_slope(adata, list(P_SAMPLES), panel)
    assert var_lhd == var_p
    panel_var = var_lhd
    diff = slope_lhd - slope_p

    # Read Supp T6.
    print(f"[D6] Loading {SUPP_T6}")
    t6 = pd.read_excel(SUPP_T6, sheet_name="Portal_bias_comparison_LHD_adja", header=1)
    t6.columns = [str(c).strip() for c in t6.columns]
    if "Gene" in t6.columns:
        t6 = t6.set_index("Gene")
    t6.columns = [c.lower() for c in t6.columns]
    if "residual" not in t6.columns:
        OUT_T6.write_text(json.dumps({
            "status": "supp_t6_residual_column_not_found",
            "available_cols": list(t6.columns),
        }, indent=2))
        return
    pub_residual = pd.to_numeric(t6["residual"], errors="coerce")
    pub_residual = pub_residual[~pub_residual.index.duplicated()]
    pub_dir = np.sign(pub_residual)

    # Compare on the strict panel ∩ Supp T6.
    # ORIENTATION: our `obs.s` is portal→central (low s = portal, high s =
    # central, oriented per §5.3). So `slope > 0` = central-enriched.
    # `slope_lhd - slope_p > 0` = LHD more central-enriched than P.
    # Supp T6 "Residual" is portal-bias residual: positive = LHD more
    # PORTAL-biased than P. The two are opposite signs by definition.
    # We compare `sign(-(slope_lhd - slope_p))` to paper's `sign(residual)`,
    # equivalent to flipping our sign before the comparison.
    common = sorted(set(panel_var) & set(pub_dir.index))
    print(f"[D6]   panel ∩ Supp T6: {len(common)} genes")
    diff_arr = pd.Series(diff, index=panel_var).reindex(common).to_numpy()
    pub_arr = pub_dir.reindex(common).to_numpy()
    pub_residual_arr = pub_residual.reindex(common).to_numpy()
    # Flip our sign so positive = LHD more portal-biased than P, matching
    # the paper's residual orientation.
    our_portal_bias = -diff_arr
    valid = (our_portal_bias != 0) & (pub_arr != 0) & ~np.isnan(our_portal_bias) & ~np.isnan(pub_arr)
    n_compared = int(valid.sum())
    n_agree = int((np.sign(our_portal_bias[valid]) == pub_arr[valid]).sum())
    pct = 100.0 * n_agree / max(1, n_compared)
    rho, _ = spearmanr(our_portal_bias[valid], pub_residual_arr[valid])

    # Sanity check on a handful of literature markers to verify orientation.
    sanity_per_marker = {}
    for g, expected_dir in KNOWN_LIVER_DIRECTION.items():
        if g not in common:
            continue
        i = common.index(g)
        if not valid[i]:
            continue
        # expected_dir = -1 (portal) or +1 (central).
        # In LHD-vs-P, we have no a-priori expected sign for the residual —
        # but if the gene shifts in the same direction in both pipelines,
        # signs should match.
        sanity_per_marker[g] = {
            "our_portal_bias": float(our_portal_bias[i]),
            "paper_residual": float(pub_residual_arr[i]),
            "agree": int(np.sign(our_portal_bias[i]) == pub_arr[i]),
        }

    out = {
        "n_panel_genes": int(len(panel_var)),
        "n_overlap_with_supp_t6": len(common),
        "n_compared_directional": n_compared,
        "n_agree": n_agree,
        "sign_agreement_pct": float(pct),
        "signed_magnitude_spearman": float(rho) if not np.isnan(rho) else None,
        "passes_section_9_9_80pct": bool(pct >= 80.0),
        "orientation_note": (
            "Our s axis is portal→central (slope > 0 = central-enriched). "
            "Supp T6 residual is portal-bias (residual > 0 = LHD more "
            "portal-biased than P). Compared sign(-(slope_lhd - slope_p)) "
            "to sign(residual) so both sides are oriented as 'LHD more "
            "portal-biased than P'."
        ),
        "method": "sign(-(slope_lhd - slope_p)) vs sign(supp_t6_residual)",
        "sanity_per_marker": sanity_per_marker,
    }
    OUT_T6.write_text(json.dumps(out, indent=2, default=str))
    print(f"[D6] Wrote {OUT_T6}")
    print(f"[D6] Sign agreement: {n_agree}/{n_compared} = {pct:.1f}%; "
          f"Spearman = {rho:.3f}")


def main() -> None:
    supp_t11_calibration()
    supp_t6_lhd_p()


if __name__ == "__main__":
    main()
