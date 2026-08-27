import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

from sgd.wmatrix import block_bootstrap_W, select_ridge_lambda_loocv


def test_response_gene_permutation_only_permutes_w_rows():
    """Documents why response-label shuffling cannot be a matrix-wide null."""
    rng = np.random.default_rng(4)
    G = rng.normal(size=(18, 5))
    dgds = rng.normal(size=(18, 5))
    permutation = np.array([3, 0, 4, 1, 2])
    mean, std = block_bootstrap_W(
        G, dgds, lam=0.1, n_boot=12, block_size=2, rng_seed=9
    )
    permuted_mean, permuted_std = block_bootstrap_W(
        G, dgds[:, permutation], lam=0.1, n_boot=12,
        block_size=2, rng_seed=9,
    )
    np.testing.assert_allclose(permuted_mean, mean[permutation])
    np.testing.assert_allclose(permuted_std, std[permutation])
    original_fraction = np.mean(np.abs(mean) / (std + 1e-12) > 1.96)
    permuted_fraction = np.mean(
        np.abs(permuted_mean) / (permuted_std + 1e-12) > 1.96
    )
    assert permuted_fraction == original_fraction


def test_analytic_ridge_loocv_matches_explicit_refits():
    rng = np.random.default_rng(7)
    G = rng.normal(size=(11, 4))
    dgds = rng.normal(size=(11, 3))
    lambdas = (0.01, 0.1, 1.0)
    best, analytic = select_ridge_lambda_loocv(G, dgds, lambdas=lambdas)
    explicit = []
    for lam in lambdas:
        predictions = np.zeros_like(dgds)
        for train, test in LeaveOneOut().split(G):
            for gene in range(dgds.shape[1]):
                model = Ridge(alpha=lam, fit_intercept=True).fit(
                    G[train], dgds[train, gene]
                )
                predictions[test[0], gene] = model.predict(G[test])[0]
        explicit.append(np.mean((predictions - dgds) ** 2))
    np.testing.assert_allclose(analytic, explicit, rtol=1e-10, atol=1e-12)
    assert best == lambdas[int(np.argmin(explicit))]
