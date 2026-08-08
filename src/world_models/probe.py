"""Linear probes: does a latent linearly encode a ground-truth quantity?

Closed-form ridge regression, deliberately no sklearn: the whole probe is
four lines of linear algebra and a tutorial reader should see all of them.
"""

from __future__ import annotations

import jax.numpy as jnp


def ridge_fit(X, y, lam: float = 1e-3):
    """Ridge regression with intercept. X: (N, D), y: (N, K) -> (w, b)."""
    x_mean = X.mean(axis=0)
    y_mean = y.mean(axis=0)
    Xc = X - x_mean
    yc = y - y_mean
    d = X.shape[1]
    w = jnp.linalg.solve(Xc.T @ Xc + lam * jnp.eye(d), Xc.T @ yc)
    b = y_mean - x_mean @ w
    return w, b


def r2_score(y_true, y_pred) -> jnp.ndarray:
    """Per-target R^2, shape (K,). 1 is perfect, 0 is predicting the mean."""
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / ss_tot


def probe(latents_fit, targets_fit, latents_eval, targets_eval, lam: float = 1e-3):
    """Fit on one set of episodes, report per-target R^2 on another."""
    w, b = ridge_fit(latents_fit, targets_fit, lam)
    return r2_score(targets_eval, latents_eval @ w + b)
