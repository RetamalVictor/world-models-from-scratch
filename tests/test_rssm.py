import jax
import jax.numpy as jnp
import numpy as np

from world_models.models.rssm import RSSM, kl_balanced, kl_gauss
from world_models.train_rssm import Config, make_losses, rollout_prior


def _tiny_model_and_params(latent=8, hidden=32):
    model = RSSM(latent_dim=latent, hidden=hidden)
    params = model.init(
        jax.random.PRNGKey(0), jnp.zeros((2, 32, 32, 1)),
        model.initial_state(2), jnp.zeros((2, latent)), jnp.zeros((2, 2)),
    )
    return model, params


def test_cell_shapes_and_sigma_floor():
    model, params = _tiny_model_and_params()
    o_hat, (mu_p, sig_p), (mu_q, sig_q) = model.apply(
        params, jnp.zeros((2, 32, 32, 1)), model.initial_state(2),
        jnp.zeros((2, 8)), jnp.zeros((2, 2)),
    )
    assert o_hat.shape == (2, 32, 32, 1)
    assert mu_p.shape == (2, 8) and mu_q.shape == (2, 8)
    assert float(sig_p.min()) >= 0.1
    assert float(sig_q.min()) >= 0.1


def test_kl_gauss_is_zero_for_identical_gaussians():
    mu = jnp.ones((3, 8))
    sig = jnp.full((3, 8), 0.7)
    assert jnp.allclose(kl_gauss(mu, sig, mu, sig), 0.0, atol=1e-6)


def test_kl_balancing_routes_gradients():
    mu_q = jnp.ones(4) * 0.5
    mu_p = jnp.zeros(4)
    sig = jnp.ones(4)

    def kl_wrt_q(m_q, alpha):
        return kl_balanced(m_q, sig, mu_p, sig, alpha).sum()

    def kl_wrt_p(m_p, alpha):
        return kl_balanced(mu_q, sig, m_p, sig, alpha).sum()

    # alpha = 1: all gradient goes to the prior, none to the posterior.
    assert jnp.allclose(jax.grad(kl_wrt_q)(mu_q, 1.0), 0.0)
    assert float(jnp.abs(jax.grad(kl_wrt_p)(mu_p, 1.0)).max()) > 0
    # alpha = 0: the reverse.
    assert jnp.allclose(jax.grad(kl_wrt_p)(mu_p, 0.0), 0.0)
    assert float(jnp.abs(jax.grad(kl_wrt_q)(mu_q, 0.0)).max()) > 0


def test_rollout_prior_shapes():
    model, params = _tiny_model_and_params()
    frames = rollout_prior(
        model, params, model.initial_state(2), jnp.zeros((2, 8)),
        jnp.zeros((5, 2, 2)),
    )
    assert frames.shape == (5, 2, 32, 32, 1)
    noisy = rollout_prior(
        model, params, model.initial_state(2), jnp.zeros((2, 8)),
        jnp.zeros((5, 2, 2)), noise=jnp.ones((5, 2, 8)),
    )
    assert noisy.shape == (5, 2, 32, 32, 1)
    assert not np.allclose(np.asarray(frames), np.asarray(noisy))


def test_overfits_a_tiny_batch():
    import optax
    from flax.training import train_state

    from world_models.envs import EnvParams, collect_trajectory

    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    traj = jax.vmap(
        lambda k: collect_trajectory(k, None, 6, EnvParams())
    )(keys)
    obs_seq = jnp.transpose(traj["obs"], (1, 0, 2, 3, 4))   # (7, 4, 32, 32, 1)
    act_seq = jnp.transpose(traj["action"], (1, 0, 2))

    model, params = _tiny_model_and_params()
    config = Config(latent_dim=8, alpha=0.8)
    sequence_loss = make_losses(model, config)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(3e-3)
    )

    @jax.jit
    def step(state, key):
        grad_fn = jax.value_and_grad(sequence_loss, has_aux=True)
        (loss, _), grads = grad_fn(state.params, obs_seq, act_seq, key,
                                   jnp.float32(1.0))
        return state.apply_gradients(grads=grads), loss

    _, initial = step(state, jax.random.PRNGKey(1))
    for i in range(150):
        state, loss = step(state, jax.random.fold_in(jax.random.PRNGKey(2), i))
    assert float(loss) < 0.5 * float(initial)
