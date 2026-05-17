"""
11 — Steatosis lipid join, mixed-effects model, and per-gene call criteria.

Absorbs legacy ``stage_D/D1_lipid_join.py``, ``stage_D/D2_steatosis_model.py``,
and ``stage_D/D3_call_criteria.py``. Three functions run in sequence:

  - ``lipid_join()`` (D1) reads the per-donor sheets of Supp Table 10,
    normalises spot names ``_1`` → ``-1``, and joins to ``obs.lipid_pct`` on
    ``visium_human.h5ad``.
  - ``steatosis_model()`` (D2) fits the per-gene OLS on the four steatotic
    donors: ``g ~ bs(s, df=4) + lipid_pct + s:lipid_pct + log_total_umi_raw +
    C(donor_id)`` with donor-clustered robust SE, plus the no-UMI / per-donor /
    leave-one-out variants.
  - ``call_criteria()`` (D3) evaluates each gene against the six §11.3
    criteria and assigns ``four_criteria_flag`` ∈ {robust_1234,
    donor_sensitive_12, weak_1, null_0}.

The OLS fitters (``fit_one_gene_cohort`` etc.), ``per_gene_full``, and
``steatotic_analysis_mask`` are imported from ``sgd.steatosis``; the
single-donor axis builder ``build_axis_for_p_donor`` from ``sgd.axis``.

Reads:  RESULTS/visium_human.h5ad, config.SUPP_T10 (Supp Table 10 xlsx).
Writes: RESULTS/visium_human.h5ad updated with obs.lipid_pct;
        RESULTS/lipid_join_summary.json, RESULTS/steatosis_mixed_effects.parquet,
        RESULTS/steatosis_model_summary.json,
        RESULTS/steatosis_call_criteria.parquet,
        RESULTS/steatosis_call_summary.json.
Feeds:  steatosis regression §4.9 — R4.1–R4.5, R5.1–R5.3; Fig 4A, supp S5.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from scipy.sparse import issparse
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgd import config
from sgd.config import RESULTS, STEATOTIC_DONORS, SUPP_T10
from sgd.io import normalise_supp_t10_barcode
from sgd.panels import (CENTRAL_MARKERS, PORTAL_MARKERS, _gene_score,
                        analysis_gene_panel)
from sgd.axis import build_axis_for_p_donor
from sgd.steatosis import (fit_one_gene_cohort, fit_one_gene_leave_out,
                           fit_one_gene_per_donor, per_gene_full,
                           steatotic_analysis_mask)

warnings.filterwarnings("ignore")
np.random.seed(0)

VISIUM = RESULTS / "visium_human.h5ad"

# D1 outputs.
LIPID_JOIN_SUMMARY = RESULTS / "lipid_join_summary.json"

# D2 outputs.
MIXED_EFFECTS = RESULTS / "steatosis_mixed_effects.parquet"
MODEL_SUMMARY = RESULTS / "steatosis_model_summary.json"

# D3 outputs.
CALL_CRITERIA = RESULTS / "steatosis_call_criteria.parquet"
CALL_SUMMARY = RESULTS / "steatosis_call_summary.json"

N_JOBS = 16

FDR_Q_THRESHOLD = 0.05
LEAVE_OUT_P_THRESHOLD = 0.05  # criterion 3/4: survives at p<0.05 same sign


# ---------------------------------------------------------------------------
# D1 — Lipid annotation join (Supp Table 10)
# ---------------------------------------------------------------------------

def read_t10_donor(sheet: str) -> pd.DataFrame:
    """
    Supp T10 has the title row in M2 only; other sheets have header on row 0.
    Try both; return DataFrame with columns spot_name, lipid_percentage.
    """
    for h in (0, 1):
        df = pd.read_excel(SUPP_T10, sheet_name=sheet, header=h)
        df.columns = [str(c).strip() for c in df.columns]
        if "spot_name" in df.columns and "lipid_percentage" in df.columns:
            df["lipid_percentage"] = pd.to_numeric(df["lipid_percentage"], errors="coerce")
            return df.dropna(subset=["spot_name", "lipid_percentage"]).copy()
    raise RuntimeError(f"Could not parse sheet {sheet} of Supp T10")


def lipid_join() -> None:
    """D1 — join Supp T10 lipid_pct onto visium_human.h5ad obs."""
    print(f"[D1] Loading {VISIUM}")
    adata = sc.read_h5ad(VISIUM)
    print(f"[D1]   {adata.n_obs} spots × {adata.n_vars} genes")

    # Build per-(sample, barcode) lipid lookup.
    lipid_lookup: dict[tuple[str, str], float] = {}
    per_donor: dict[str, dict] = {}
    for donor in STEATOTIC_DONORS:
        df = read_t10_donor(donor)
        df["barcode"] = df["spot_name"].map(normalise_supp_t10_barcode)
        n_supp = len(df)
        for _, row in df.iterrows():
            lipid_lookup[(donor, row["barcode"])] = float(row["lipid_percentage"])
        per_donor[donor] = {
            "n_supp_t10_rows": int(n_supp),
            "min_lipid": float(df["lipid_percentage"].min()),
            "max_lipid": float(df["lipid_percentage"].max()),
            "mean_lipid": float(df["lipid_percentage"].mean()),
            "median_lipid": float(df["lipid_percentage"].median()),
        }
        print(f"[D1]   {donor}: {n_supp} Supp T10 rows; "
              f"lipid range [{df['lipid_percentage'].min():.3f}, "
              f"{df['lipid_percentage'].max():.3f}], "
              f"median {df['lipid_percentage'].median():.3f}")

    # Join to obs.
    samples = adata.obs["sample_id"].astype(str).to_numpy()
    barcodes = adata.obs["barcode"].astype(str).to_numpy()
    lipid_pct = np.full(adata.n_obs, np.nan)
    n_joined_per_donor: dict[str, int] = {d: 0 for d in STEATOTIC_DONORS}
    for i in range(adata.n_obs):
        s, b = samples[i], barcodes[i]
        v = lipid_lookup.get((s, b))
        if v is not None:
            lipid_pct[i] = v
            n_joined_per_donor[s] += 1
    adata.obs["lipid_pct"] = lipid_pct
    adata.write_h5ad(VISIUM, compression="gzip")
    print(f"[D1] Updated {VISIUM} with obs.lipid_pct")

    # Coverage summary.
    summary: dict = {"per_donor": {}}
    for donor in STEATOTIC_DONORS:
        n_in_visium = int((samples == donor).sum())
        n_with_lipid = n_joined_per_donor[donor]
        summary["per_donor"][donor] = {
            **per_donor[donor],
            "n_spots_in_visium_after_QC": n_in_visium,
            "n_spots_with_lipid_after_join": n_with_lipid,
            "join_coverage_pct": (100.0 * n_with_lipid / max(1, n_in_visium)),
        }
        print(f"[D1]   {donor}: joined {n_with_lipid}/{n_in_visium} "
              f"({summary['per_donor'][donor]['join_coverage_pct']:.1f}%)")

    summary["n_total_steatotic_spots_with_lipid"] = int(np.isfinite(lipid_pct).sum())
    LIPID_JOIN_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"[D1] Wrote {LIPID_JOIN_SUMMARY}")


# ---------------------------------------------------------------------------
# D2 — Steatosis mixed-effects model
# ---------------------------------------------------------------------------

def steatosis_model() -> None:
    """D2 — per-gene OLS on the steatotic cohort."""
    print(f"[D2] Loading {VISIUM}")
    adata = sc.read_h5ad(VISIUM)

    # Build obs.s for P6 ad-hoc — Stage B built the axis on LHD samples
    # only, so P6 has all-NaN s. Without this, P6 drops out of the
    # steatosis model entirely (criterion 3 leave-P6 becomes trivially
    # vacuous). Use the §5.1 recipe per-donor, same as D6.
    s_full = adata.obs["s"].to_numpy().copy() if "s" in adata.obs.columns else np.full(adata.n_obs, np.nan)
    samples_arr = adata.obs["sample_id"].astype(str).to_numpy()
    for donor in STEATOTIC_DONORS:
        m = samples_arr == donor
        if m.sum() == 0:
            continue
        if np.isfinite(s_full[m]).any():
            continue  # already populated by Stage B
        s_donor_only = build_axis_for_p_donor(adata, donor)
        s_full[m] = s_donor_only[m]
        print(f"[D2]   built ad-hoc obs.s for {donor}: "
              f"{int(np.isfinite(s_full[m]).sum())}/{int(m.sum())} valid")
    adata.obs["s"] = s_full

    spot_mask = steatotic_analysis_mask(adata)
    print(f"[D2]   steatotic-cohort ∩ ¬fibrotic ∩ lipid_finite: "
          f"{int(spot_mask.sum())} spots")

    # Strict-66 panel.
    panel = analysis_gene_panel(adata, mode="strict")
    sub = adata[spot_mask, panel].copy()
    print(f"[D2]   strict-66 panel: {sub.n_vars} genes; cohort spots: {sub.n_obs}")

    # Build the metadata DataFrame once for the model.
    df_meta = pd.DataFrame({
        "s": sub.obs["s"].to_numpy(),
        "lipid_pct": sub.obs["lipid_pct"].to_numpy().astype(float),
        "log_total_umi_raw": np.log1p(sub.obs["total_umi_raw"].astype(float).to_numpy()),
        "donor_id": sub.obs["donor_id"].astype(str).to_numpy(),
    })
    # Ensure no NaN in covariates.
    keep = (~df_meta.isna().any(axis=1)).to_numpy()
    df_meta = df_meta.loc[keep].reset_index(drop=True)
    X = sub.X
    if issparse(X):
        X = X.toarray()
    X = X[keep]
    print(f"[D2]   after dropping rows with NaN covariates: {len(df_meta)} spots")
    print(f"[D2]   per-donor spot counts: "
          f"{df_meta['donor_id'].value_counts().to_dict()}")

    # Run per-gene in parallel.
    print(f"[D2] Running per-gene OLS (cohort + no-UMI + per-donor + leave-P6 + leave-M1) "
          f"on {sub.n_vars} genes (n_jobs={N_JOBS})...")
    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(per_gene_full)(i, sub.var_names[i], X[:, i], df_meta)
        for i in range(sub.n_vars)
    )
    df_out = pd.DataFrame(results)
    df_out.to_parquet(MIXED_EFFECTS, compression="zstd", index=False)
    print(f"[D2] Wrote {MIXED_EFFECTS} ({len(df_out)} rows)")

    # Quick summary.
    n_ok = int((df_out["cohort_status"] == "ok").sum())
    n_p_lt_05 = int((df_out["p_2"] < 0.05).sum())
    summary = {
        "n_genes_total": int(len(df_out)),
        "n_cohort_fit_ok": n_ok,
        "n_p2_lt_005_uncorrected": n_p_lt_05,
        "n_spots_per_donor": df_meta["donor_id"].value_counts().to_dict(),
        "n_spots_total": int(len(df_meta)),
    }
    MODEL_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[D2] Wrote {MODEL_SUMMARY}")
    print(f"[D2] Summary: {n_ok} fits ok, {n_p_lt_05} with raw p<0.05 "
          f"(FDR correction in D3)")


# ---------------------------------------------------------------------------
# D3 — Per-gene call criteria evaluation (§11.3)
# ---------------------------------------------------------------------------

def call_criteria() -> None:
    """D3 — evaluate the six §11.3 criteria and assign four_criteria_flag."""
    print(f"[D3] Loading {MIXED_EFFECTS}")
    df = pd.read_parquet(MIXED_EFFECTS)
    n = len(df)
    print(f"[D3]   {n} genes")

    ok_mask = df["cohort_status"].astype(str) == "ok"

    # --- Criterion 1: FDR-significant β2 ---
    pvals = df["p_2"].to_numpy()
    qvals = np.full(n, np.nan)
    finite = np.isfinite(pvals)
    if finite.any():
        _, q_finite, _, _ = multipletests(pvals[finite], alpha=FDR_Q_THRESHOLD,
                                           method="fdr_bh")
        qvals[finite] = q_finite
    df["q_2"] = qvals
    df["criterion_1"] = (qvals < FDR_Q_THRESHOLD) & ok_mask

    # --- Criterion 2: per-donor β2 sign agreement ---
    cohort_sign = np.sign(df["beta_2"].to_numpy())
    donor_signs_match = np.ones(n, dtype=bool)
    for donor in STEATOTIC_DONORS:
        col = f"beta_2_{donor}"
        if col not in df.columns:
            continue
        donor_beta = df[col].to_numpy()
        donor_sign = np.sign(donor_beta)
        # NaN donor values fail the sign check.
        agree = (donor_sign == cohort_sign) & np.isfinite(donor_beta)
        donor_signs_match &= agree
    df["criterion_2"] = donor_signs_match & ok_mask

    # --- Criterion 3: survives excluding P6 ---
    leave_p6_status = df["leave_p6_status"].astype(str) == "ok"
    leave_p6_beta = df["beta_2_leave_p6"].to_numpy()
    leave_p6_p = df["p_2_leave_p6"].to_numpy()
    leave_p6_sign = np.sign(leave_p6_beta)
    df["criterion_3"] = (
        leave_p6_status
        & (leave_p6_sign == cohort_sign)
        & (leave_p6_p < LEAVE_OUT_P_THRESHOLD)
        & ok_mask
    )

    # --- Criterion 4: survives excluding M1 ---
    leave_m1_status = df["leave_m1_status"].astype(str) == "ok"
    leave_m1_beta = df["beta_2_leave_m1"].to_numpy()
    leave_m1_p = df["p_2_leave_m1"].to_numpy()
    leave_m1_sign = np.sign(leave_m1_beta)
    df["criterion_4"] = (
        leave_m1_status
        & (leave_m1_sign == cohort_sign)
        & (leave_m1_p < LEAVE_OUT_P_THRESHOLD)
        & ok_mask
    )

    # --- Criterion 5: survives excluding M2 (largest per-donor magnitude
    # for all 3 robust genes) ---
    leave_m2_status = df["leave_m2_status"].astype(str) == "ok"
    leave_m2_beta = df["beta_2_leave_m2"].to_numpy()
    leave_m2_p = df["p_2_leave_m2"].to_numpy()
    leave_m2_sign = np.sign(leave_m2_beta)
    df["criterion_5"] = (
        leave_m2_status
        & (leave_m2_sign == cohort_sign)
        & (leave_m2_p < LEAVE_OUT_P_THRESHOLD)
        & ok_mask
    )

    # --- Criterion 6: survives excluding M3 ---
    leave_m3_status = df["leave_m3_status"].astype(str) == "ok"
    leave_m3_beta = df["beta_2_leave_m3"].to_numpy()
    leave_m3_p = df["p_2_leave_m3"].to_numpy()
    leave_m3_sign = np.sign(leave_m3_beta)
    df["criterion_6"] = (
        leave_m3_status
        & (leave_m3_sign == cohort_sign)
        & (leave_m3_p < LEAVE_OUT_P_THRESHOLD)
        & ok_mask
    )

    # --- Conjunction flag ---
    c1 = df["criterion_1"].to_numpy()
    c2 = df["criterion_2"].to_numpy()
    c3 = df["criterion_3"].to_numpy()
    c4 = df["criterion_4"].to_numpy()
    c5 = df["criterion_5"].to_numpy()
    c6 = df["criterion_6"].to_numpy()
    flag = np.zeros(n, dtype="O")
    for i in range(n):
        if c1[i] and c2[i] and c3[i] and c4[i]:
            flag[i] = "robust_1234"
        elif c1[i] and c2[i]:
            flag[i] = "donor_sensitive_12"
        elif c1[i]:
            flag[i] = "weak_1"
        else:
            flag[i] = "null_0"
    df["four_criteria_flag"] = flag

    # --- Strengthened "robust to any single donor" flag ---
    # All four §11.3 criteria + leave-M2 + leave-M3 surviving.
    df["robust_to_any_single_donor"] = (
        c1 & c2 & c3 & c4 & c5 & c6
    )

    # --- Stage A §14 #9 second-half check ---
    # Did β2 magnitude shift substantially when log(UMI) was removed?
    # Restrict the sensitivity flag to **called genes** (robust_1234 or
    # donor_sensitive_12) only — null genes with tiny β2 can produce
    # spuriously large relative deltas and falsely trigger §14 #9b.
    beta_umi = df["beta_2"].to_numpy()
    beta_no_umi = df["beta_2_no_umi"].to_numpy()
    delta = np.abs(beta_no_umi - beta_umi) / np.maximum(np.abs(beta_umi), 1e-9)
    df["c2_log_umi_delta_relative"] = delta
    called_mask = np.isin(flag, ("robust_1234", "donor_sensitive_12"))
    df["c2_log_umi_sensitive"] = (delta > 0.3) & ok_mask & called_mask

    df.to_parquet(CALL_CRITERIA, compression="zstd", index=False)
    print(f"[D3] Wrote {CALL_CRITERIA}")

    # --- Summary ---
    counts = pd.Series(flag).value_counts().to_dict()
    n_robust = int(counts.get("robust_1234", 0))
    n_donor_sens = int(counts.get("donor_sensitive_12", 0))
    n_weak = int(counts.get("weak_1", 0))
    n_null = int(counts.get("null_0", 0))
    n_c2_sens = int(df["c2_log_umi_sensitive"].sum())

    if n_robust >= 5:
        scenario = "robust_set"
    elif n_robust >= 1:
        scenario = "small_robust_set"
    elif n_donor_sens > 0:
        scenario = "donor_sensitive"
    else:
        scenario = "null"

    sev14_6 = n_robust == 0
    sev14_9b = bool(n_c2_sens / max(1, n_robust + n_donor_sens) > 0.5) if (n_robust + n_donor_sens) > 0 else False

    n_any_donor = int(df["robust_to_any_single_donor"].sum())
    any_donor_genes = df[df["robust_to_any_single_donor"]]["gene"].tolist()

    summary = {
        "n_genes": int(n),
        "n_cohort_fit_ok": int(ok_mask.sum()),
        "criterion_counts": {
            "criterion_1_FDR_q_lt_005": int(c1.sum()),
            "criterion_2_donor_signs_agree": int(c2.sum()),
            "criterion_3_survives_no_P6": int(c3.sum()),
            "criterion_4_survives_no_M1": int(c4.sum()),
            "criterion_5_survives_no_M2": int(c5.sum()),
            "criterion_6_survives_no_M3": int(c6.sum()),
        },
        "flag_counts": {
            "robust_1234": n_robust,
            "donor_sensitive_12": n_donor_sens,
            "weak_1": n_weak,
            "null_0": n_null,
        },
        "robust_to_any_single_donor_count": n_any_donor,
        "robust_to_any_single_donor_genes": any_donor_genes,
        "outcome_scenario": scenario,
        "sev14_6_no_robust_genes": bool(sev14_6),
        "c2_log_umi_sensitive_count": n_c2_sens,
        "sev14_9b_majority_c2_sensitive": sev14_9b,
        "thresholds": {"fdr_q": FDR_Q_THRESHOLD,
                        "leave_out_p": LEAVE_OUT_P_THRESHOLD,
                        "c2_relative_delta": 0.3},
    }
    CALL_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[D3] Wrote {CALL_SUMMARY}")
    print(f"[D3] Counts: c1={int(c1.sum())} c2={int(c2.sum())} "
          f"c3={int(c3.sum())} c4={int(c4.sum())} c5={int(c5.sum())} c6={int(c6.sum())}")
    print(f"[D3] Robust to any single donor (1234+5+6): "
          f"{n_any_donor} genes — {any_donor_genes}")
    print(f"[D3] Flag counts: robust_1234={n_robust} donor_sensitive_12={n_donor_sens} "
          f"weak_1={n_weak} null_0={n_null}")
    print(f"[D3] Outcome scenario: {scenario}")
    if n_robust > 0:
        robust_genes = df[df["four_criteria_flag"] == "robust_1234"]["gene"].tolist()
        print(f"[D3] Robust genes: {robust_genes}")


def main() -> None:
    lipid_join()
    steatosis_model()
    call_criteria()


if __name__ == "__main__":
    main()
