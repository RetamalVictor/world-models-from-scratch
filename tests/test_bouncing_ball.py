import jax
import jax.numpy as jnp

from world_models.envs import (
    BouncingBallEnv,
    EnvParams,
    collect_trajectory,
    random_nudge_policy,
)

PARAMS = EnvParams()


def test_reset_shapes():
    obs, state = BouncingBallEnv.reset(jax.random.PRNGKey(0), PARAMS)
    assert obs.shape == (PARAMS.img_h, PARAMS.img_w, 1)
    assert obs.dtype == jnp.float32
    assert 0.0 <= float(state.x) < PARAMS.img_w
    assert 0.0 <= float(state.y) < PARAMS.img_h


def test_spawn_positions_are_independent():
    # Regression test: x and y were once drawn with the same PRNG key, which
    # put every spawn on the diagonal x == y.
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    _, states = jax.vmap(lambda k: BouncingBallEnv.reset(k, PARAMS))(keys)
    assert not jnp.allclose(states.x, states.y, atol=1e-3)


def test_blob_peaks_at_ball_position():
    obs, state = BouncingBallEnv.reset(jax.random.PRNGKey(3), PARAMS)
    row, col = jnp.unravel_index(jnp.argmax(obs[:, :, 0]), obs.shape[:2])
    assert abs(float(row) - float(state.y)) <= 1.0
    assert abs(float(col) - float(state.x)) <= 1.0


def test_ball_stays_in_bounds_under_random_actions():
    traj = collect_trajectory(
        jax.random.PRNGKey(1), random_nudge_policy(0.3), 500, PARAMS
    )
    assert jnp.all((traj["x"] >= 0) & (traj["x"] < PARAMS.img_w))
    assert jnp.all((traj["y"] >= 0) & (traj["y"] < PARAMS.img_h))


def test_random_policy_actions_are_recorded():
    # Regression test: collect_trajectory used to accept policy_fn and then
    # silently roll out with zero actions anyway.
    traj = collect_trajectory(
        jax.random.PRNGKey(2), random_nudge_policy(0.3), 50, PARAMS
    )
    assert float(jnp.abs(traj["action"][1:]).max()) > 0.0


def test_trajectory_shapes():
    n = 10
    traj = collect_trajectory(jax.random.PRNGKey(4), None, n, PARAMS)
    assert traj["obs"].shape == (n + 1, PARAMS.img_h, PARAMS.img_w, 1)
    assert traj["action"].shape == (n + 1, 2)
    assert traj["vx"].shape == (n + 1,)
