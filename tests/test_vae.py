import jax
import jax.numpy as jnp

from world_models.models.vae import VAE, elbo_terms


def test_vae_shapes():
    model = VAE(latent_dim=16)
    x = jnp.zeros((4, 32, 32, 1))
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    params = model.init(k1, x, k2)
    recon, mu, logvar = model.apply(params, x, k2)
    assert recon.shape == x.shape
    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)


def test_kl_is_zero_for_standard_normal_posterior():
    mu = jnp.zeros((2, 16))
    logvar = jnp.zeros((2, 16))
    x = jnp.zeros((2, 32, 32, 1))
    rec, kl = elbo_terms(x, x, mu, logvar)
    assert float(rec) == 0.0
    assert abs(float(kl)) < 1e-5


def test_kl_is_positive_away_from_prior():
    mu = jnp.ones((2, 16)) * 2.0
    logvar = jnp.zeros((2, 16))
    x = jnp.zeros((2, 32, 32, 1))
    _, kl = elbo_terms(x, x, mu, logvar)
    assert float(kl) > 1.0


def test_overfits_a_single_batch():
    # Broken gradients hide well; a tiny overfit run flushes them out.
    import optax
    from flax.training import train_state

    from world_models.envs import EnvParams, collect_trajectory

    traj = collect_trajectory(jax.random.PRNGKey(0), None, 31, EnvParams())
    batch = traj["obs"]  # (32, 32, 32, 1)

    model = VAE(latent_dim=16)
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    params = model.init(k1, batch, k2)
    tx = optax.adam(1e-3)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

    @jax.jit
    def train_step(state, key):
        def loss_fn(p):
            recon, mu, logvar = model.apply(p, batch, key)
            rec, kl = elbo_terms(recon, batch, mu, logvar)
            return rec + kl
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    _, initial = train_step(state, jax.random.PRNGKey(2))
    for i in range(200):
        state, loss = train_step(state, jax.random.fold_in(jax.random.PRNGKey(3), i))
    assert float(loss) < 0.5 * float(initial)
