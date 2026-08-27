"""Exhaustive wild-cluster bootstrap primitives for four-donor inference."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import norm


WEBB_WEIGHTS = np.array([
    -np.sqrt(1.5), -1.0, -np.sqrt(0.5),
    np.sqrt(0.5), 1.0, np.sqrt(1.5),
])
RADEMACHER_WEIGHTS = np.array([-1.0, 1.0])


def build_steatosis_design(df_meta: pd.DataFrame, restricted: bool = False) -> pd.DataFrame:
    """Build the unrestricted or null-restricted steatosis design matrix."""
    if restricted:
        formula = "1 + bs(s, df=4) + lipid_pct + log_total_umi_raw + C(donor_id)"
    else:
        formula = (
            "1 + bs(s, df=4) + lipid_pct + s:lipid_pct "
            "+ log_total_umi_raw + C(donor_id)"
        )
    return dmatrix(formula, data=df_meta, return_type="dataframe")


def interaction_column(design: pd.DataFrame) -> int:
    """Locate the single ``s:lipid_pct`` column in an unrestricted design."""
    candidates = [
        i for i, column in enumerate(design.columns)
        if "lipid_pct" in column and ":" in column
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected one s:lipid_pct interaction column; found "
            f"{[design.columns[i] for i in candidates]}"
        )
    return candidates[0]


def exhaustive_weight_grid(weights: np.ndarray, n_clusters: int) -> np.ndarray:
    """Return every assignment of a finite weight set to ``n_clusters``."""
    return np.asarray(list(itertools.product(weights, repeat=n_clusters)), dtype=float)


def _fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    return beta, fitted, y - fitted


def _cluster_covariance(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: np.ndarray,
    xtx_inverse: np.ndarray | None = None,
) -> np.ndarray:
    """CR1 cluster-robust covariance with the statsmodels small-sample factor."""
    inverse = xtx_inverse
    if inverse is None:
        try:
            inverse = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(X.T @ X)
    clusters = np.unique(cluster_ids)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for cluster in clusters:
        score = X[cluster_ids == cluster].T @ residuals[cluster_ids == cluster]
        meat += np.outer(score, score)
    n_obs, n_parameters = X.shape
    correction = 1.0
    if len(clusters) > 1 and n_obs > n_parameters:
        correction = (
            (n_obs - 1) / (n_obs - n_parameters)
            * len(clusters) / (len(clusters) - 1)
        )
    return inverse @ meat @ inverse * correction


def wild_cluster_bootstrap(
    outcome: np.ndarray,
    unrestricted_design: np.ndarray,
    restricted_design: np.ndarray,
    interaction_index: int,
    cluster_ids: np.ndarray,
    cluster_order: Sequence[str],
    webb_grid: np.ndarray | None = None,
    rademacher_grid: np.ndarray | None = None,
) -> dict:
    """Test the interaction coefficient using exhaustive restricted-null WCR.

    The bootstrap outcome is the restricted fitted value plus the restricted
    residual multiplied by one weight per donor. Each bootstrap sample is
    studentized with the same CR1 covariance estimator used for the observed
    statistic. The add-one two-sided p-value convention is reported explicitly
    in the output metadata written by the caller.
    """
    y = np.asarray(outcome, dtype=float)
    X = np.asarray(unrestricted_design, dtype=float)
    X0 = np.asarray(restricted_design, dtype=float)
    clusters = np.asarray(cluster_ids).astype(str)
    order = tuple(str(c) for c in cluster_order)
    unknown = sorted(set(np.unique(clusters)) - set(order))
    if unknown:
        raise ValueError(f"cluster_order is missing observed clusters: {unknown}")

    beta, _, residuals = _fit_ols(X, y)
    try:
        xtx_inverse = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        xtx_inverse = np.linalg.pinv(X.T @ X)
    covariance = _cluster_covariance(X, residuals, clusters, xtx_inverse)
    variance = max(float(covariance[interaction_index, interaction_index]), 0.0)
    se_observed = float(np.sqrt(variance))
    beta_observed = float(beta[interaction_index])
    if not np.isfinite(se_observed) or se_observed <= 0:
        return {"status": "zero_or_invalid_observed_se"}
    t_observed = beta_observed / se_observed
    p_asymptotic = float(2.0 * norm.sf(abs(t_observed)))

    _, fitted_null, residuals_null = _fit_ols(X0, y)
    projection = xtx_inverse @ X.T
    cluster_index = np.array([order.index(c) for c in clusters], dtype=int)
    grids = {
        "webb": webb_grid if webb_grid is not None
        else exhaustive_weight_grid(WEBB_WEIGHTS, len(order)),
        "rademacher": rademacher_grid if rademacher_grid is not None
        else exhaustive_weight_grid(RADEMACHER_WEIGHTS, len(order)),
    }

    pvalues: dict[str, float] = {}
    valid_counts: dict[str, int] = {}
    for label, grid in grids.items():
        extreme = 0
        valid = 0
        threshold = abs(t_observed)
        for weights in grid:
            y_boot = fitted_null + weights[cluster_index] * residuals_null
            beta_boot = projection @ y_boot
            residuals_boot = y_boot - X @ beta_boot
            covariance_boot = _cluster_covariance(
                X, residuals_boot, clusters, xtx_inverse
            )
            variance_boot = max(
                float(covariance_boot[interaction_index, interaction_index]), 0.0
            )
            se_boot = np.sqrt(variance_boot)
            if not np.isfinite(se_boot) or se_boot <= 0:
                continue
            valid += 1
            t_boot = float(beta_boot[interaction_index] / se_boot)
            # Exhaustive grids contain algebraically tied sign-symmetric
            # assignments. Count ties with a small numerical tolerance so the
            # p-value does not depend on BLAS-level rounding.
            extreme += int(
                abs(t_boot) > threshold
                or np.isclose(abs(t_boot), threshold, rtol=1e-10, atol=1e-12)
            )
        if valid == 0:
            pvalues[label] = np.nan
        else:
            pvalues[label] = float((extreme + 1) / (valid + 1))
        valid_counts[label] = valid

    return {
        "status": "ok",
        "beta_2": beta_observed,
        "se_2": se_observed,
        "t_observed": float(t_observed),
        "p_cluster_robust_asymptotic": p_asymptotic,
        "p_webb": pvalues["webb"],
        "p_rademacher": pvalues["rademacher"],
        "n_webb_valid": valid_counts["webb"],
        "n_rademacher_valid": valid_counts["rademacher"],
    }
