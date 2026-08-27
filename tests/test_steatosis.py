import numpy as np
import pandas as pd

from sgd.steatosis import apply_calling_rule
from sgd.wild_cluster import (
    build_steatosis_design,
    interaction_column,
    wild_cluster_bootstrap,
)


def _screening_row(leave_m3_p: float) -> pd.DataFrame:
    row = {
        "gene": "TEST", "cohort_status": "ok", "beta_2": 1.0,
        "beta_2_no_umi": 1.0, "p_2": 1e-4,
    }
    for donor in ("M1", "M2", "M3", "P6"):
        row[f"beta_2_{donor}"] = 0.5
        row[f"beta_2_leave_{donor.lower()}"] = 0.8
        row[f"p_2_leave_{donor.lower()}"] = (
            leave_m3_p if donor == "M3" else 0.01
        )
        row[f"leave_{donor.lower()}_status"] = "ok"
    return pd.DataFrame([row])


def test_robust_label_requires_all_six_documented_criteria():
    failed = apply_calling_rule(_screening_row(leave_m3_p=0.2)).iloc[0]
    assert failed["criterion_6"] == False  # noqa: E712
    assert failed["four_criteria_flag"] == "donor_sensitive_12"
    assert failed["calling_status"] == "donor_sensitive"
    passed = apply_calling_rule(_screening_row(leave_m3_p=0.01)).iloc[0]
    assert passed["four_criteria_flag"] == "robust_1234"
    assert passed["calling_status"] == "robust_all_six"


def test_wild_cluster_bootstrap_enumerates_all_four_cluster_weights():
    rng = np.random.default_rng(2)
    donors = np.repeat(["M1", "M2", "M3", "P6"], 12)
    s = np.tile(np.linspace(0.05, 0.95, 12), 4)
    lipid = np.repeat([0.2, 0.5, 0.8, 1.1], 12) + rng.normal(0, 0.03, len(s))
    meta = pd.DataFrame({
        "s": s,
        "lipid_pct": lipid,
        "log_total_umi_raw": rng.normal(8, 0.2, len(s)),
        "donor_id": donors,
    })
    X = build_steatosis_design(meta, restricted=False)
    X0 = build_steatosis_design(meta, restricted=True)
    y = 0.8 * s * lipid + rng.normal(0, 0.1, len(s))
    result = wild_cluster_bootstrap(
        y, X.to_numpy(), X0.to_numpy(), interaction_column(X), donors,
        ("M1", "M2", "M3", "P6"),
    )
    assert result["status"] == "ok"
    assert result["n_webb_valid"] == 6 ** 4
    assert result["n_rademacher_valid"] == 2 ** 4
    assert 0.0 < result["p_webb"] <= 1.0
