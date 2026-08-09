import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from world_models.models.rssm import (
    RSSM,
    continue_bce,
    kl_balanced,
    kl_gauss,
    load_rssm,
)
from world_models.train_rssm import Config, make_losses, rollout_prior

# The Step 3 parameter tree, dumped before obs_size and the continue
# head existed. Every trained checkpoint from Steps 3-4 restores against
# exactly this, so the defaults are not free to move.
DEFAULT_PARAM_SHAPES = {
    "core/hn/bias": (128,),
    "core/hn/kernel": (128, 128),
    "core/hr/kernel": (128, 128),
    "core/hz/kernel": (128, 128),
    "core/in/bias": (128,),
    "core/in/kernel": (18, 128),
    "core/ir/bias": (128,),
    "core/ir/kernel": (18, 128),
    "core/iz/bias": (128,),
    "core/iz/kernel": (18, 128),
    "decoder/ConvTranspose_0/bias": (64,),
    "decoder/ConvTranspose_0/kernel": (4, 4, 128, 64),
    "decoder/ConvTranspose_1/bias": (32,),
    "decoder/ConvTranspose_1/kernel": (4, 4, 64, 32),
    "decoder/ConvTranspose_2/bias": (1,),
    "decoder/ConvTranspose_2/kernel": (4, 4, 32, 1),
    "decoder/Dense_0/bias": (256,),
    "decoder/Dense_0/kernel": (144, 256),
    "decoder/Dense_1/bias": (2048,),
    "decoder/Dense_1/kernel": (256, 2048),
    "encoder/Conv_0/bias": (32,),
    "encoder/Conv_0/kernel": (4, 4, 1, 32),
    "encoder/Conv_1/bias": (64,),
    "encoder/Conv_1/kernel": (4, 4, 32, 64),
    "encoder/Conv_2/bias": (128,),
    "encoder/Conv_2/kernel": (4, 4, 64, 128),
    "encoder/Dense_0/bias": (256,),
    "encoder/Dense_0/kernel": (2048, 256),
    "post_head/Dense_0/bias": (128,),
    "post_head/Dense_0/kernel": (384, 128),
    "post_head/Dense_1/bias": (16,),
    "post_head/Dense_1/kernel": (128, 16),
    "post_head/Dense_2/bias": (16,),
    "post_head/Dense_2/kernel": (128, 16),
    "prior_head/Dense_0/bias": (128,),
    "prior_head/Dense_0/kernel": (128, 128),
    "prior_head/Dense_1/bias": (16,),
    "prior_head/Dense_1/kernel": (128, 16),
    "prior_head/Dense_2/bias": (16,),
    "prior_head/Dense_2/kernel": (128, 16),
    "reward_head/Dense_0/bias": (128,),
    "reward_head/Dense_0/kernel": (144, 128),
    "reward_head/Dense_1/bias": (1,),
    "reward_head/Dense_1/kernel": (128, 1),
}


def _param_shapes(params):
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    return {"/".join(str(k.key) for k in path[1:]): tuple(v.shape)
            for path, v in flat}


def _tiny_model_and_params(latent=8, hidden=32):
    model = RSSM(latent_dim=latent, hidden=hidden)
    params = model.init(
        jax.random.PRNGKey(0), jnp.zeros((2, 32, 32, 1)),
        model.initial_state(2), jnp.zeros((2, latent)), jnp.zeros((2, 2)),
    )
    return model, params


def _default_params():
    model = RSSM()
    return model, model.init(
        jax.random.PRNGKey(0), jnp.zeros((1, 32, 32, 1)),
        model.initial_state(1), jnp.zeros((1, 16)), jnp.zeros((1, 2)),
    )


def test_cell_shapes_and_sigma_floor():
    model, params = _tiny_model_and_params()
    o_hat, (mu_p, sig_p), (mu_q, sig_q), r_hat = model.apply(
        params, jnp.zeros((2, 32, 32, 1)), model.initial_state(2),
        jnp.zeros((2, 8)), jnp.zeros((2, 2)),
    )
    assert o_hat.shape == (2, 32, 32, 1)
    assert mu_p.shape == (2, 8) and mu_q.shape == (2, 8)
    assert r_hat.shape == (2,)
    assert float(sig_p.min()) >= 0.1
    assert float(sig_q.min()) >= 0.1


def test_rgb_channels():
    model = RSSM(latent_dim=8, hidden=32, obs_channels=3)
    params = model.init(
        jax.random.PRNGKey(0), jnp.zeros((2, 32, 32, 3)),
        model.initial_state(2), jnp.zeros((2, 8)), jnp.zeros((2, 2)),
    )
    o_hat, _, _, _ = model.apply(
        params, jnp.zeros((2, 32, 32, 3)), model.initial_state(2),
        jnp.zeros((2, 8)), jnp.zeros((2, 2)),
    )
    assert o_hat.shape == (2, 32, 32, 3)


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
    rew_seq = jnp.zeros(obs_seq.shape[:2])
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(3e-3)
    )

    @jax.jit
    def step(state, key):
        grad_fn = jax.value_and_grad(sequence_loss, has_aux=True)
        (loss, _), grads = grad_fn(state.params, obs_seq, act_seq, rew_seq,
                                   key, jnp.float32(1.0))
        return state.apply_gradients(grads=grads), loss

    _, initial = step(state, jax.random.PRNGKey(1))
    for i in range(150):
        state, loss = step(state, jax.random.fold_in(jax.random.PRNGKey(2), i))
    assert float(loss) < 0.5 * float(initial)


def test_default_param_tree_is_frozen():
    _, params = _default_params()
    assert _param_shapes(params) == DEFAULT_PARAM_SHAPES


def test_load_rssm_reads_a_pre_doom_run(tmp_path):
    """config.json from Steps 3-4 has none of the new keys."""
    model, params = _default_params()
    (tmp_path / "config.json").write_text(json.dumps(
        {"latent_dim": 16, "hidden": 128, "min_sigma": 0.1,
         "obs_channels": 1}
    ))
    (tmp_path / "checkpoint.msgpack").write_bytes(
        serialization.to_bytes(params))

    loaded, loaded_params = load_rssm(tmp_path)
    assert loaded.obs_size == 32
    assert loaded.action_dim == 2
    assert not loaded.predict_continue
    jax.tree.map(np.testing.assert_array_equal, params, loaded_params)


def test_continue_head_only_exists_when_asked():
    _, off = _tiny_model_and_params()
    assert "continue_head" not in off["params"]

    model = RSSM(latent_dim=8, hidden=32, predict_continue=True)
    args = (jnp.zeros((2, 32, 32, 1)), model.initial_state(2),
            jnp.zeros((2, 8)), jnp.zeros((2, 2)))
    params = model.init(jax.random.PRNGKey(0), *args)
    assert "continue_head" in params["params"]
    assert model.apply(params, *args)[4].shape == (2,)
    logit = model.apply(params, model.initial_state(2), jnp.zeros((2, 8)),
                        method=RSSM.continue_logit)
    assert logit.shape == (2,)


def test_doom_scale_model_runs():
    model = RSSM(latent_dim=32, action_dim=3, hidden=256, obs_size=64,
                 predict_continue=True)
    args = (jnp.zeros((2, 64, 64, 1)), model.initial_state(2),
            jnp.zeros((2, 32)), jnp.zeros((2, 3)))
    params = model.init(jax.random.PRNGKey(0), *args)
    o_hat, (mu_p, _), (mu_q, _), r_hat, c_logit = model.apply(params, *args)
    assert o_hat.shape == (2, 64, 64, 1)
    assert mu_p.shape == (2, 32) and mu_q.shape == (2, 32)
    assert r_hat.shape == (2,) and c_logit.shape == (2,)
    # the extra stride-2 stage, mirrored, keeps the bottleneck at 4x4
    shapes = _param_shapes(params)
    assert shapes["encoder/Conv_3/kernel"] == (4, 4, 128, 256)
    assert shapes["decoder/Dense_1/kernel"] == (256, 4096)
    assert shapes["decoder/ConvTranspose_3/kernel"] == (4, 4, 32, 1)


def test_obs_size_must_reach_a_4x4_bottleneck():
    model = RSSM(obs_size=48)
    with pytest.raises(ValueError):
        model.init(jax.random.PRNGKey(0), jnp.zeros((1, 48, 48, 1)),
                   model.initial_state(1), jnp.zeros((1, 16)),
                   jnp.zeros((1, 2)))


def test_overfits_a_tiny_batch_at_64():
    import optax
    from flax.training import train_state

    from world_models.envs import EnvParams, collect_trajectory

    params_64 = EnvParams(img_h=64, img_w=64)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    traj = jax.vmap(
        lambda k: collect_trajectory(k, None, 3, params_64)
    )(keys)
    obs_seq = jnp.transpose(traj["obs"], (1, 0, 2, 3, 4))   # (4, 2, 64, 64, 1)
    act_seq = jnp.transpose(traj["action"], (1, 0, 2))

    model = RSSM(latent_dim=8, hidden=32, obs_size=64)
    params = model.init(
        jax.random.PRNGKey(0), obs_seq[0], model.initial_state(2),
        jnp.zeros((2, 8)), act_seq[0],
    )
    sequence_loss = make_losses(model, Config(latent_dim=8, alpha=0.8))
    rew_seq = jnp.zeros(obs_seq.shape[:2])
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(3e-3)
    )

    @jax.jit
    def step(state, key):
        grad_fn = jax.value_and_grad(sequence_loss, has_aux=True)
        (loss, _), grads = grad_fn(state.params, obs_seq, act_seq, rew_seq,
                                   key, jnp.float32(1.0))
        return state.apply_gradients(grads=grads), loss

    _, initial = step(state, jax.random.PRNGKey(1))
    for i in range(80):
        state, loss = step(state, jax.random.fold_in(jax.random.PRNGKey(2), i))
    assert float(loss) < 0.5 * float(initial)


def test_continue_bce_matches_the_definition():
    logits = jnp.array([-4.0, -0.5, 0.0, 2.0])
    p = jax.nn.sigmoid(logits)
    for target in (0.0, 1.0):
        naive = -(target * jnp.log(p) + (1.0 - target) * jnp.log1p(-p))
        assert jnp.allclose(continue_bce(logits, target), naive, atol=1e-5)
    assert float(continue_bce(jnp.float32(0.0), 1.0)) == pytest.approx(
        float(jnp.log(2.0)), abs=1e-6)
    # the naive form overflows to inf out here; the stable one does not
    assert float(continue_bce(jnp.float32(60.0), 0.0)) == pytest.approx(60.0)
    assert float(continue_bce(jnp.float32(60.0), 1.0)) == pytest.approx(0.0)
