"""Steatosis-model spot mask, per-gene OLS fitters, and per-bin bootstrap."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

from sgd.config import STEATOTIC_DONORS

# Per-bin bootstrap default (D11).
N_BOOT = 200


# ---------------------------------------------------------------------------
# Steatotic cohort spot mask
# ---------------------------------------------------------------------------

def steatotic_analysis_mask(adata: ad.AnnData) -> np.ndarray:
    """
    Steatotic cohort filter: sample_id ∈ {M1, M2, M3, P6}, not fibrotic,
    lipid_pct finite. P6 is NOT in lhd_analysis_mask, so we cannot reuse
    that helper.
    """
    samples = adata.obs["sample_id"].astype(str).to_numpy()
    in_cohort = np.isin(samples, list(STEATOTIC_DONORS))
    if "fibrotic_spot" in adata.obs.columns:
        not_fibrotic = ~adata.obs["fibrotic_spot"].astype(bool).to_numpy()
    else:
        not_fibrotic = np.ones(adata.n_obs, dtype=bool)
    if "lipid_pct" not in adata.obs.columns:
        raise RuntimeError("obs.lipid_pct missing — run D1 first")
    has_lipid = ~adata.obs["lipid_pct"].isna().to_numpy()
    return in_cohort & not_fibrotic & has_lipid


# ---------------------------------------------------------------------------
# Per-gene OLS fitters (§11.2 / §11.3)
# ---------------------------------------------------------------------------

def fit_one_gene_cohort(g_values: np.ndarray, df_meta: pd.DataFrame,
                         include_log_umi: bool = True) -> dict:
    """
    Cohort-level OLS with **donor-clustered robust standard errors** for
    the s:lipid_pct term. Spot-level OLS p-values would be anti-conservative
    because the true independent unit is closer to donor/slide than spot.
    Cluster-robust SE on `donor_id` accounts for the within-donor correlation.
    """
    df = df_meta.copy()
    df["g"] = g_values
    formula = "g ~ bs(s, df=4) + lipid_pct + s:lipid_pct + C(donor_id)"
    if include_log_umi:
        formula += " + log_total_umi_raw"
    try:
        donor_groups = df["donor_id"].astype(str).to_numpy()
        model = ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": donor_groups}
        )
        term = "s:lipid_pct"
        if term in model.params.index:
            beta_2 = float(model.params[term])
            se_2 = float(model.bse[term])
            p_2 = float(model.pvalues[term])
        else:
            return {"status": "term_not_found", "params": list(model.params.index)}
        return {
            "status": "ok",
            "beta_2": beta_2,
            "se_2": se_2,
            "p_2": p_2,
            "n_obs": int(model.nobs),
            "rsquared": float(model.rsquared),
            "se_method": "cluster_robust_donor",
        }
    except Exception as e:
        return {"status": "fit_failed", "error": f"{type(e).__name__}: {str(e)[:80]}"}


def fit_one_gene_per_donor(g_values: np.ndarray, df_meta: pd.DataFrame
                            ) -> dict[str, dict]:
    """Per-donor OLS for one gene; returns β2 per donor."""
    out: dict[str, dict] = {}
    for donor in STEATOTIC_DONORS:
        m = df_meta["donor_id"].to_numpy() == donor
        if m.sum() < 30:
            out[donor] = {"status": "too_few_spots", "n": int(m.sum())}
            continue
        df = df_meta.loc[m].copy()
        df["g"] = g_values[m]
        formula = "g ~ bs(s, df=4) + lipid_pct + s:lipid_pct + log_total_umi_raw"
        try:
            model = ols(formula, data=df).fit()
            term = "s:lipid_pct"
            if term in model.params.index:
                out[donor] = {
                    "status": "ok",
                    "beta_2": float(model.params[term]),
                    "se_2": float(model.bse[term]),
                    "p_2": float(model.pvalues[term]),
                    "n_obs": int(model.nobs),
                }
            else:
                out[donor] = {"status": "term_not_found"}
        except Exception as e:
            out[donor] = {"status": "fit_failed",
                            "error": f"{type(e).__name__}: {str(e)[:80]}"}
    return out


def fit_one_gene_leave_out(g_values: np.ndarray, df_meta: pd.DataFrame,
                            leave_out: str) -> dict:
    """
    Cohort fit with one donor excluded — for §11.3 criteria 3 and 4.
    Uses cluster-robust SE on the remaining 3 donors (degrees of freedom
    will be small but the method handles it).
    """
    keep = df_meta["donor_id"].to_numpy() != leave_out
    if keep.sum() < 50:
        return {"status": "too_few_spots", "n": int(keep.sum())}
    return fit_one_gene_cohort(g_values[keep], df_meta.loc[keep].reset_index(drop=True),
                                include_log_umi=True)


def per_gene_full(idx: int, gene: str, X_col: np.ndarray,
                   df_meta: pd.DataFrame) -> dict:
    cohort = fit_one_gene_cohort(X_col, df_meta, include_log_umi=True)
    cohort_no_umi = fit_one_gene_cohort(X_col, df_meta, include_log_umi=False)
    per_donor = fit_one_gene_per_donor(X_col, df_meta)
    leave_p6 = fit_one_gene_leave_out(X_col, df_meta, "P6")
    leave_m1 = fit_one_gene_leave_out(X_col, df_meta, "M1")
    leave_m2 = fit_one_gene_leave_out(X_col, df_meta, "M2")
    leave_m3 = fit_one_gene_leave_out(X_col, df_meta, "M3")
    return {
        "gene": gene,
        "cohort_status": cohort.get("status"),
        "beta_2": cohort.get("beta_2"),
        "se_2": cohort.get("se_2"),
        "p_2": cohort.get("p_2"),
        "n_obs": cohort.get("n_obs"),
        "rsquared": cohort.get("rsquared"),
        "beta_2_no_umi": cohort_no_umi.get("beta_2"),
        "p_2_no_umi": cohort_no_umi.get("p_2"),
        **{f"beta_2_{d}": per_donor[d].get("beta_2") for d in STEATOTIC_DONORS},
        **{f"p_2_{d}": per_donor[d].get("p_2") for d in STEATOTIC_DONORS},
        **{f"per_donor_status_{d}": per_donor[d].get("status") for d in STEATOTIC_DONORS},
        "beta_2_leave_p6": leave_p6.get("beta_2"),
        "p_2_leave_p6": leave_p6.get("p_2"),
        "leave_p6_status": leave_p6.get("status"),
        "beta_2_leave_m1": leave_m1.get("beta_2"),
        "p_2_leave_m1": leave_m1.get("p_2"),
        "leave_m1_status": leave_m1.get("status"),
        "beta_2_leave_m2": leave_m2.get("beta_2"),
        "p_2_leave_m2": leave_m2.get("p_2"),
        "leave_m2_status": leave_m2.get("status"),
        "beta_2_leave_m3": leave_m3.get("beta_2"),
        "p_2_leave_m3": leave_m3.get("p_2"),
        "leave_m3_status": leave_m3.get("status"),
    }


# ---------------------------------------------------------------------------
# Per-bin mean with bootstrap CI (D11)
# ---------------------------------------------------------------------------

def per_bin_mean_with_bootstrap(s: np.ndarray, x: np.ndarray, n_bins: int,
                                  n_boot: int = N_BOOT, rng=None
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin centers, means, bootstrap CI low, CI high)."""
    rng = rng or np.random.default_rng(0)
    edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    ci_lo = np.full(n_bins, np.nan)
    ci_hi = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (s >= edges[i]) & (s < edges[i + 1])
        if i == n_bins - 1:
            m |= (s == 1.0)
        if m.sum() < 5:
            continue
        xs = x[m]
        means[i] = xs.mean()
        boot = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, len(xs), len(xs))
            boot[b] = xs[idx].mean()
        ci_lo[i] = np.percentile(boot, 2.5)
        ci_hi[i] = np.percentile(boot, 97.5)
    return centers, means, ci_lo, ci_hi
