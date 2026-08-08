import jax
import jax.numpy as jnp

from world_models.probe import probe, r2_score, ridge_fit


def _split(X, y, n_fit):
    return X[:n_fit], y[:n_fit], X[n_fit:], y[n_fit:]


def test_probe_recovers_a_linear_map():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    X = jax.random.normal(k1, (500, 16))
    w = jax.random.normal(k2, (16, 2))
    y = X @ w + 1.5 + 0.01 * jax.random.normal(k3, (500, 2))
    r2 = probe(*_split(X, y, 250))
    assert float(r2.min()) > 0.99


def test_probe_reports_nothing_on_noise_targets():
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    X = jax.random.normal(k1, (500, 16))
    y = jax.random.normal(k2, (500, 2))  # independent of X
    r2 = probe(*_split(X, y, 250))
    assert float(r2.max()) < 0.1


def test_ridge_handles_intercept():
    X = jax.random.normal(jax.random.PRNGKey(2), (100, 4))
    y = 3.0 * X[:, :1] + 7.0
    w, b = ridge_fit(X, y, lam=1e-6)
    pred = X @ w + b
    assert float(r2_score(y, pred)[0]) > 0.999
    assert abs(float(b[0]) - 7.0) < 0.1
