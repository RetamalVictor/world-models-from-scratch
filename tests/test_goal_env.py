import jax
import jax.numpy as jnp

from world_models.envs import BallGoalEnv, GoalEnvParams, GoalEnvState
from world_models.envs.bouncing_ball_goal import collect_trajectory

HOVER = GoalEnvParams(goal_speed=0.0)
FOLLOW = GoalEnvParams(goal_speed=1.0)


def _state(x, y, gx, gy):
    z = jnp.float32
    return GoalEnvState(x=z(x), y=z(y), vx=z(0), vy=z(0),
                        gx=z(gx), gy=z(gy), gvx=z(0), gvy=z(0),
                        step_count=jnp.int32(0))


def test_reset_shapes_rgb():
    obs, state = BallGoalEnv.reset(jax.random.PRNGKey(0), HOVER)
    assert obs.shape == (32, 32, 3)
    assert float(state.gvx) == 0.0 and float(state.gvy) == 0.0


def test_goal_is_visible_and_red():
    # A frame with only the goal (agent parked far away) should be
    # red-dominant at the goal pixel.
    obs, _, _, _, _ = BallGoalEnv.step(
        jax.random.PRNGKey(0), _state(5, 5, 25, 25), jnp.zeros(2), HOVER)
    r, g = float(obs[25, 25, 0]), float(obs[25, 25, 1])
    assert r > 0.9
    assert g < 0.3


def test_reward_matches_kernel():
    # Agent and goal at rest: reward is exactly the Gaussian kernel.
    _, _, reward, _, _ = BallGoalEnv.step(
        jax.random.PRNGKey(0), _state(10, 10, 10, 10), jnp.zeros(2), HOVER)
    assert abs(float(reward) - 1.0) < 1e-5
    _, _, reward, _, _ = BallGoalEnv.step(
        jax.random.PRNGKey(0), _state(10, 10, 16, 10), jnp.zeros(2), HOVER)
    expected = jnp.exp(-(6.0 ** 2) / (2 * HOVER.reward_sigma ** 2))
    assert abs(float(reward) - float(expected)) < 1e-5


def test_hover_goal_stays_put_and_follow_goal_moves():
    traj_h = collect_trajectory(jax.random.PRNGKey(1), None, 50, HOVER)
    assert float(jnp.ptp(traj_h["gx"])) < 1e-5
    assert float(jnp.ptp(traj_h["gy"])) < 1e-5

    traj_f = collect_trajectory(jax.random.PRNGKey(1), None, 200, FOLLOW)
    assert float(jnp.ptp(traj_f["gx"])) > 1.0
    assert jnp.all((traj_f["gx"] >= 0) & (traj_f["gx"] < FOLLOW.img_w))
    assert jnp.all((traj_f["gy"] >= 0) & (traj_f["gy"] < FOLLOW.img_h))


def test_collect_trajectory_contract():
    traj = collect_trajectory(jax.random.PRNGKey(2), None, 10, HOVER)
    assert traj["obs"].shape == (11, 32, 32, 3)
    assert traj["reward"].shape == (11,)
    assert float(traj["reward"].min()) > 0.0
    assert float(traj["reward"].max()) <= 1.0
