"""Episode collection: the plain-Python loop that feeds the replay buffer.

No jit, no scan. env.step is a call into a stateful engine (Doom's C++
process, or any other env with the same surface) that Python holds a
handle to, not a pure function JAX can trace or batch — so this loop
pays one host round-trip per policy step, deliberately, the same way
envs/doom.py already does. Everything upstream of the buffer (the RSSM,
the actor-critic) stays jittable; this file is where that boundary is.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from world_models.models.rssm import RSSM


def _as_action_vector(action, action_dim: int) -> np.ndarray:
    """A scalar index becomes a one-hot; a length-action_dim vector
    passes through. The inverse of DoomTakeCover._index: policies may
    hand back either form, but storage always wants the vector, matching
    the (T+1, action_dim) convention every other env's dataset uses."""
    a = np.asarray(action)
    if a.ndim == 0:
        v = np.zeros(action_dim, np.float32)
        v[int(a)] = 1.0
        return v
    return a.astype(np.float32)


def collect_episode(env, policy, max_steps: int) -> dict:
    """Run one episode of `env` under `policy`.

    `env` needs reset() -> obs, step(action) -> (obs, reward, done),
    action_dim, and died (True iff the episode ended in death rather
    than a step cap). `policy` needs reset(obs0), called once with the
    reset frame, and __call__(obs) -> action, called once per step.

    Returns the dict ReplayBuffer.add_episode unpacks: obs uint8
    (T+1, H, W, C) — input frames are float32 in [0, 1], rounded rather
    than truncated so the uint8 round-trip doesn't bias every pixel
    down by up to 1; action float32 (T+1, action_dim) with an all-zero
    first row, since no action preceded the reset frame; reward float32
    (T+1,) with reward[0] pinned to 0.01 — alive at reset pays exactly
    what alive pays on every later frame, it's just not the consequence
    of an action; terminated, True only when the env reports the agent
    died. T <= max_steps: shorter when the episode ends first, and
    hitting the cap without dying is a time limit, not the world
    ending, so terminated stays False.
    """
    obs0 = env.reset()
    policy.reset(obs0)

    obs = [obs0]
    action = [np.zeros(env.action_dim, np.float32)]
    reward = [np.float32(0.01)]
    terminated = False

    frame = obs0
    for _ in range(max_steps):
        a = policy(frame)
        frame, r, done = env.step(a)
        obs.append(frame)
        action.append(_as_action_vector(a, env.action_dim))
        reward.append(np.float32(r))
        if done:
            terminated = bool(env.died)
            break

    return {
        "obs": np.round(np.stack(obs) * 255.0).astype(np.uint8),
        "action": np.stack(action).astype(np.float32),
        "reward": np.asarray(reward, np.float32),
        "terminated": terminated,
    }


class RandomPolicy:
    """Uniform random action index, for seed episodes before any world
    model exists to filter states with.

    Its own np.random.default_rng, independent of env or jax rng
    streams. reset() does not reseed: seeding is the caller's job,
    typically by constructing a fresh RandomPolicy per episode.
    """

    def __init__(self, action_dim: int, seed: int):
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed)

    def reset(self, obs0):
        pass

    def __call__(self, obs) -> int:
        return int(self._rng.integers(self.action_dim))


class RSSMPolicy:
    """Acts from an RSSM belief state filtered online from raw frames.

    encode/core_step/post_dist are each jitted once here, at
    construction, rather than left to retrace on every Python-level
    step — this runs once per env step, so re-tracing would dominate.
    wm/wm_params are frozen for the policy's lifetime, closed over by
    those jits.

    act_fn(actor_params, s, key) -> (action_dim,) float action is
    supplied by the caller and is the only place a policy (e.g. the
    actor) enters this file; nothing here imports one, so this module
    stays usable before an actor exists (random collection) and after
    (actor collection) without caring which.

    Params are call-time arguments, not closures: online training
    changes them every round, and a closure would force a fresh jit
    trace per episode. Construct once, then point set_params at the
    current params before each episode.

    z is always the posterior mean (post_dist's mu, sigma discarded):
    this filters a belief for acting, it does not train, so there is
    nothing to sample noise for.

    Ordering matters: each call folds the incoming frame into the
    belief first, acts from the updated state, and only then advances
    h with the chosen action — the same correct-then-act order the
    jitted ball runners use. Acting before the update would give every
    action a one-step-stale belief.
    """

    def __init__(self, wm: RSSM, act_fn, key, action_dim: int):
        self.action_dim = action_dim
        self._wm = wm
        self._act_fn = act_fn
        self._key = key
        self._step = 0
        self._wm_params = None
        self._actor_params = None
        self._encode = jax.jit(
            lambda p, o: wm.apply(p, o, method=RSSM.encode))
        self._core_step = jax.jit(
            lambda p, h, z, a: wm.apply(p, h, z, a, method=RSSM.core_step))
        self._post_dist = jax.jit(
            lambda p, h, e: wm.apply(p, h, e, method=RSSM.post_dist))

    def set_params(self, wm_params, actor_params=None):
        self._wm_params = wm_params
        self._actor_params = actor_params

    def reseed(self, key):
        """New action-sampling key; per-episode, from the caller's lane."""
        self._key = key
        self._step = 0

    def reset(self, obs0):
        # h starts blank; the first __call__ receives the reset frame
        # and folds it in, so nothing else happens here.
        self._h = self._wm.initial_state(1)
        self._step = 0

    def __call__(self, obs) -> np.ndarray:
        e = self._encode(self._wm_params, jnp.asarray(obs)[None])
        z, _ = self._post_dist(self._wm_params, self._h, e)
        s = jnp.concatenate([self._h[0], z[0]])
        key_t = jax.random.fold_in(self._key, self._step)
        self._step += 1
        action = jnp.asarray(
            self._act_fn(self._actor_params, s, key_t), jnp.float32)
        self._h = self._core_step(self._wm_params, self._h, z, action[None])
        return np.asarray(action)
