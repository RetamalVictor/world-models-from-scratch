"""Bouncing ball in a box, as a gymnax-style JAX environment.

State: (x, y, vx, vy) in continuous coords on [0, W) x [0, H).
Action: (ax, ay) nudge added to the velocity, clipped to [-max_nudge, max_nudge].
Observation: (H, W, 1) float32 grayscale image with a Gaussian blob centred
on the ball.

The API mirrors gymnax so swapping in a real env later is a small change:

    obs, state = BouncingBallEnv.reset(key, params)
    obs, state, reward, done, info = BouncingBallEnv.step(key, state, action, params)

The info dict carries the true state. That is the whole point of this env:
the ground truth is available, so we can probe learned latents against it.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class EnvParams(NamedTuple):
    img_h: int = 32
    img_w: int = 32
    ball_radius: float = 2.0       # sigma of the Gaussian blob (pixels)
    max_speed: float = 2.0         # clamp on velocity magnitude
    max_nudge: float = 0.3         # action clamp
    dt: float = 1.0
    max_steps: int = 200


class EnvState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    step_count: jnp.ndarray


def _render(x: jnp.ndarray, y: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
    """Render the ball as a Gaussian blob -> (H, W, 1) float32 in [0, 1]."""
    ys = jnp.arange(params.img_h, dtype=jnp.float32)
    xs = jnp.arange(params.img_w, dtype=jnp.float32)
    gx, gy = jnp.meshgrid(xs, ys)
    sigma = params.ball_radius
    img = jnp.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * sigma ** 2))
    return img[:, :, None]


def _reflect(pos, vel, limit):
    """Reflect position and velocity so the ball stays inside [0, limit).

    One floor and one ceiling reflection is enough as long as the ball can't
    cross the whole box in a single step (max_speed << box size).
    """
    below = pos < 0.0
    pos = jnp.where(below, -pos, pos)
    vel = jnp.where(below, -vel, vel)
    above = pos >= limit
    pos = jnp.where(above, 2.0 * limit - pos, pos)
    vel = jnp.where(above, -vel, vel)
    return pos, vel


class BouncingBallEnv:
    """Stateless, purely functional bouncing ball environment."""

    @staticmethod
    def default_params() -> EnvParams:
        return EnvParams()

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def reset(key: jax.Array, params: EnvParams) -> tuple[jnp.ndarray, EnvState]:
        kx, ky, kv = jax.random.split(key, 3)
        margin = 4.0  # keep the ball away from the walls at spawn
        x = jax.random.uniform(kx, minval=margin, maxval=params.img_w - margin)
        y = jax.random.uniform(ky, minval=margin, maxval=params.img_h - margin)
        angle = jax.random.uniform(kv, minval=0.0, maxval=2.0 * jnp.pi)
        speed = params.max_speed * 0.8  # start fairly fast so bounces happen
        vx = speed * jnp.cos(angle)
        vy = speed * jnp.sin(angle)
        state = EnvState(x=x, y=y, vx=vx, vy=vy, step_count=jnp.int32(0))
        obs = _render(x, y, params)
        return obs, state

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def step(
        key: jax.Array,
        state: EnvState,
        action: jax.Array,        # shape (2,): (ax, ay)
        params: EnvParams,
    ) -> tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, dict]:
        ax = jnp.clip(action[0], -params.max_nudge, params.max_nudge)
        ay = jnp.clip(action[1], -params.max_nudge, params.max_nudge)
        vx = state.vx + ax
        vy = state.vy + ay

        speed = jnp.sqrt(vx ** 2 + vy ** 2)
        scale = jnp.where(speed > params.max_speed, params.max_speed / speed, 1.0)
        vx = vx * scale
        vy = vy * scale

        x = state.x + vx * params.dt
        y = state.y + vy * params.dt

        x, vx = _reflect(x, vx, jnp.float32(params.img_w))
        y, vy = _reflect(y, vy, jnp.float32(params.img_h))

        step_count = state.step_count + 1
        new_state = EnvState(x=x, y=y, vx=vx, vy=vy, step_count=step_count)
        obs = _render(x, y, params)
        reward = jnp.float32(0.0)  # no reward in the toy env
        done = step_count >= params.max_steps
        info = {"true_x": x, "true_y": y, "true_vx": vx, "true_vy": vy}
        return obs, new_state, reward, done, info


def random_nudge_policy(scale: float = 0.3):
    """Uniform random nudges in [-scale, scale]^2.

    Build the policy once and reuse it: collect_trajectory treats the policy
    as a static jit argument, so every fresh closure triggers a recompile.
    """
    def policy_fn(key: jax.Array, state: EnvState) -> jnp.ndarray:
        return jax.random.uniform(key, (2,), minval=-scale, maxval=scale)
    return policy_fn


@partial(jax.jit, static_argnums=(1, 2, 3))
def collect_trajectory(
    key: jax.Array,
    policy_fn=None,
    n_steps: int = 200,
    params: EnvParams = EnvParams(),
) -> dict:
    """Roll out the env for n_steps and stack the results.

    policy_fn is a callable (key, state) -> (2,) action, or None for zero
    actions. Data for training the dynamics models should come from a random
    policy, otherwise the action channel carries no information and the
    action-conditioned part of the model has nothing to learn from.

    Returns a dict of arrays with leading dim n_steps + 1 (the reset frame is
    prepended, with a zero action):
        obs: (T, H, W, 1)
        action: (T, 2)
        x, y, vx, vy: (T,)   ground truth for probing
    """
    env = BouncingBallEnv
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
            "obs": obs,
            "action": action,
            "x": info["true_x"],
            "y": info["true_y"],
            "vx": info["true_vx"],
            "vy": info["true_vy"],
        }

    keys = jax.random.split(k_steps, n_steps)
    _, trajectory = jax.lax.scan(scan_fn, state0, keys)

    init_frame = {
        "obs": obs0,
        "action": jnp.zeros(2),
        "x": state0.x,
        "y": state0.y,
        "vx": state0.vx,
        "vy": state0.vy,
    }
    trajectory = jax.tree.map(
        lambda first, rest: jnp.concatenate([first[None], rest]),
        init_frame,
        trajectory,
    )
    return trajectory
