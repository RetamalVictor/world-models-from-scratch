"""Bouncing ball with a rendered goal — env v2, for Step 4 (control).

Two entities in the box: the agent ball (white, action-controlled, same
physics as v1) and a goal ball (red, no actions). One knob covers both
tasks: goal_speed = 0 is the hover task (goal spawns at a random spot
and stays put), goal_speed > 0 is the follow task (the goal cruises and
bounces off walls like everything else).

Observation: (H, W, 3) float32 RGB. Agent white, goal red, additive and
clipped where they overlap. Reward is a dense Gaussian kernel on the
agent-goal distance, computed by the env from its internal state — the
agent only ever receives the scalar.

The v1 env (bouncing_ball.py) is deliberately untouched so Steps 0-3
stay reproducible.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from world_models.envs.bouncing_ball import _reflect


class GoalEnvParams(NamedTuple):
    img_h: int = 32
    img_w: int = 32
    ball_radius: float = 2.0
    goal_radius: float = 2.0
    max_speed: float = 2.0
    max_nudge: float = 0.3
    goal_speed: float = 0.0        # 0 = hover task, > 0 = follow task
    reward_sigma: float = 3.0
    dt: float = 1.0
    max_steps: int = 200


class GoalEnvState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    gx: jnp.ndarray
    gy: jnp.ndarray
    gvx: jnp.ndarray
    gvy: jnp.ndarray
    step_count: jnp.ndarray


def _blob(x, y, sigma, params) -> jnp.ndarray:
    ys = jnp.arange(params.img_h, dtype=jnp.float32)
    xs = jnp.arange(params.img_w, dtype=jnp.float32)
    gx, gy = jnp.meshgrid(xs, ys)
    return jnp.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * sigma ** 2))


def _render_rgb(state: GoalEnvState, params: GoalEnvParams) -> jnp.ndarray:
    agent = _blob(state.x, state.y, params.ball_radius, params)
    goal = _blob(state.gx, state.gy, params.goal_radius, params)
    r = jnp.clip(agent + goal, 0.0, 1.0)
    g = jnp.clip(agent + 0.15 * goal, 0.0, 1.0)
    b = jnp.clip(agent + 0.15 * goal, 0.0, 1.0)
    return jnp.stack([r, g, b], axis=-1)


def _reward(state: GoalEnvState, params: GoalEnvParams) -> jnp.ndarray:
    d2 = (state.x - state.gx) ** 2 + (state.y - state.gy) ** 2
    return jnp.exp(-d2 / (2.0 * params.reward_sigma ** 2))


class BallGoalEnv:
    """Stateless, purely functional two-entity environment."""

    @staticmethod
    def default_params() -> GoalEnvParams:
        return GoalEnvParams()

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def reset(key: jax.Array, params: GoalEnvParams):
        kx, ky, kv, kgx, kgy, kgv = jax.random.split(key, 6)
        margin = 4.0
        x = jax.random.uniform(kx, minval=margin, maxval=params.img_w - margin)
        y = jax.random.uniform(ky, minval=margin, maxval=params.img_h - margin)
        angle = jax.random.uniform(kv, minval=0.0, maxval=2.0 * jnp.pi)
        speed = params.max_speed * 0.8
        gx = jax.random.uniform(kgx, minval=margin, maxval=params.img_w - margin)
        gy = jax.random.uniform(kgy, minval=margin, maxval=params.img_h - margin)
        gangle = jax.random.uniform(kgv, minval=0.0, maxval=2.0 * jnp.pi)
        state = GoalEnvState(
            x=x, y=y,
            vx=speed * jnp.cos(angle), vy=speed * jnp.sin(angle),
            gx=gx, gy=gy,
            gvx=params.goal_speed * jnp.cos(gangle),
            gvy=params.goal_speed * jnp.sin(gangle),
            step_count=jnp.int32(0),
        )
        return _render_rgb(state, params), state

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def step(key: jax.Array, state: GoalEnvState, action: jax.Array,
             params: GoalEnvParams):
        ax = jnp.clip(action[0], -params.max_nudge, params.max_nudge)
        ay = jnp.clip(action[1], -params.max_nudge, params.max_nudge)
        vx = state.vx + ax
        vy = state.vy + ay
        speed = jnp.sqrt(vx ** 2 + vy ** 2)
        scale = jnp.where(speed > params.max_speed, params.max_speed / speed, 1.0)
        vx, vy = vx * scale, vy * scale
        x = state.x + vx * params.dt
        y = state.y + vy * params.dt
        x, vx = _reflect(x, vx, jnp.float32(params.img_w))
        y, vy = _reflect(y, vy, jnp.float32(params.img_h))

        gx = state.gx + state.gvx * params.dt
        gy = state.gy + state.gvy * params.dt
        gx, gvx = _reflect(gx, state.gvx, jnp.float32(params.img_w))
        gy, gvy = _reflect(gy, state.gvy, jnp.float32(params.img_h))

        step_count = state.step_count + 1
        new_state = GoalEnvState(x=x, y=y, vx=vx, vy=vy,
                                 gx=gx, gy=gy, gvx=gvx, gvy=gvy,
                                 step_count=step_count)
        obs = _render_rgb(new_state, params)
        reward = _reward(new_state, params)
        done = step_count >= params.max_steps
        info = {"true_x": x, "true_y": y, "true_vx": vx, "true_vy": vy,
                "true_gx": gx, "true_gy": gy, "true_gvx": gvx,
                "true_gvy": gvy}
        return obs, new_state, reward, done, info


@partial(jax.jit, static_argnums=(1, 2, 3))
def collect_trajectory(
    key: jax.Array,
    policy_fn=None,
    n_steps: int = 200,
    params: GoalEnvParams = GoalEnvParams(),
) -> dict:
    """Roll out env v2. Same contract as the v1 collector, plus reward
    and the goal's ground truth. The reset frame's reward is computed
    (it's a pure function of state), not zero-filled."""
    env = BallGoalEnv
    k_reset, k_steps = jax.random.split(key)
    obs0, state0 = env.reset(k_reset, params)

    def scan_fn(state, k):
        k_act, k_env = jax.random.split(k)
        if policy_fn is None:
            action = jnp.zeros(2)
        else:
            action = policy_fn(k_act, state)
        obs, state, reward, done, info = env.step(k_env, state, action, params)
        return state, {
            "obs": obs, "action": action, "reward": reward,
            "x": info["true_x"], "y": info["true_y"],
            "vx": info["true_vx"], "vy": info["true_vy"],
            "gx": info["true_gx"], "gy": info["true_gy"],
            "gvx": info["true_gvx"], "gvy": info["true_gvy"],
        }

    keys = jax.random.split(k_steps, n_steps)
    _, trajectory = jax.lax.scan(scan_fn, state0, keys)

    init_frame = {
        "obs": obs0, "action": jnp.zeros(2),
        "reward": _reward(state0, params),
        "x": state0.x, "y": state0.y, "vx": state0.vx, "vy": state0.vy,
        "gx": state0.gx, "gy": state0.gy,
        "gvx": state0.gvx, "gvy": state0.gvy,
    }
    return jax.tree.map(
        lambda first, rest: jnp.concatenate([first[None], rest]),
        init_frame, trajectory,
    )
