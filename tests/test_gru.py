import jax
import jax.numpy as jnp
import numpy as np

from world_models.models.gru_dynamics import (
    GRUDynamics,
    gaussian_nll,
    sequence_nll,
)
from world_models.rollout import open_loop_predict
from world_models.train_gru import fit_standardizer, standardize, unstandardize


def _init(model, batch=3):
    return model.init(
        jax.random.PRNGKey(0), model.initial_state(batch),
        jnp.zeros((batch, model.latent_dim)),
        jnp.zeros((batch, model.action_dim)),
    )


def test_cell_shapes_and_sigma_floor():
    model = GRUDynamics(latent_dim=16, hidden=32)
    params = _init(model)
    h, mu, sigma = model.apply(
        params, model.initial_state(3),
        jnp.ones((3, 16)), jnp.zeros((3, 2)),
    )
    assert h.shape == (3, 32)
    assert mu.shape == (3, 16)
    assert sigma.shape == (3, 16)
    assert float(sigma.min()) > 0


def test_residual_flag_offsets_the_mean():
    # direct and residual share the same parameter tree; residual only
    # adds z_prev to the predicted mean.
    direct = GRUDynamics(latent_dim=8, hidden=16, residual=False)
    residual = GRUDynamics(latent_dim=8, hidden=16, residual=True)
    params = _init(direct)
    z_prev = jnp.arange(24, dtype=jnp.float32).reshape(3, 8)
    a = jnp.zeros((3, 2))
    _, mu_d, _ = direct.apply(params, direct.initial_state(3), z_prev, a)
    _, mu_r, _ = residual.apply(params, residual.initial_state(3), z_prev, a)
    assert jnp.allclose(mu_r, mu_d + z_prev, atol=1e-5)


def test_gaussian_nll_matches_standard_normal():
    z = jnp.zeros((5, 4))
    nll = gaussian_nll(z, jnp.zeros((5, 4)), jnp.ones((5, 4)))
    expected = 0.5 * 4 * float(jnp.log(2 * jnp.pi))
    assert jnp.allclose(nll, expected, atol=1e-5)


def test_open_loop_predict_indexing():
    # Ground-truth dynamics z_{t+1} = z_t + a_{t+1}; a perfect model must
    # reproduce the trajectory exactly, which pins down every index.
    T, B, D = 12, 2, 2
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.normal(size=(T + 1, B, D)).astype(np.float32))
    z = [jnp.zeros((B, D))]
    for t in range(1, T + 1):
        z.append(z[-1] + a[t])
    z = jnp.stack(z)                       # (T+1, B, D)

    def step_fn(carry, z_in, act):
        return carry, z_in + act

    W, K = 4, 6
    preds = open_loop_predict(
        step_fn, None, z[:W], a[1:W + 1], a[W + 1:W + K]
    )
    assert preds.shape == (K, B, D)
    assert jnp.allclose(preds, z[W:W + K], atol=1e-5)


def test_overfits_ar1_dynamics():
    import optax
    from flax.training import train_state

    T, B, D = 16, 8, 4
    rng = np.random.default_rng(1)
    z = np.zeros((T + 1, B, D), dtype=np.float32)
    z[0] = rng.normal(size=(B, D))
    for t in range(1, T + 1):
        z[t] = 0.9 * z[t - 1] + 0.05 * rng.normal(size=(B, D))
    z = jnp.asarray(z)
    a = jnp.zeros((T + 1, B, 2))
    mask = jnp.ones(T)

    model = GRUDynamics(latent_dim=D, hidden=32)
    params = _init(model, batch=B)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(3e-3)
    )

    @jax.jit
    def step(state):
        def loss_fn(p):
            nll, _ = sequence_nll(model, p, z, a, mask)
            return nll
        nll, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), nll

    _, initial = step(state)
    for _ in range(300):
        state, nll = step(state)
    assert float(nll) < float(initial) - 2.0


def test_standardizer_roundtrip():
    rng = np.random.default_rng(2)
    latents = rng.normal(3.0, 5.0, size=(10, 7, 4)).astype(np.float32)
    stats = fit_standardizer(latents, slice(0, 8))
    z = standardize(latents, stats)
    back = unstandardize(z, stats)
    assert np.allclose(back, latents, atol=1e-4)
    train_flat = z[:8].reshape(-1, 4)
    assert np.allclose(train_flat.mean(0), 0, atol=1e-4)
    assert np.allclose(train_flat.std(0), 1, atol=1e-3)
