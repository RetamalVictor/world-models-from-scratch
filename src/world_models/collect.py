"""Episode collection: the plain-Python loop that feeds the replay buffer.

No jit, no scan. env.step is a call into a stateful engine (Doom's C++
process, or any other env with the same surface) that Python holds a
handle to, not a pure function JAX can trace or batch — so this loop
pays one host round-trip per policy step, deliberately, the same way
envs/doom.py already does. Everything upstream of the buffer (the RSSM,
the actor-critic) stays jittable; this file is where that boundary is.
"""

from __future__ import annotations

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

from world_models.models.rssm import RSSM


def _as_action_vector(action, action_dim: int) -> np.ndarray:
    """A scalar index becomes a one-hot; a length-action_dim vector
    passes through. The inverse of envs/doom.py's _action_index:
    policies may hand back either form, but storage always wants the
    vector, matching the (T+1, action_dim) convention every other env's
    dataset uses."""
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

    The step that kills you is not stored. Doom throws its screen
    buffer away the moment the player dies, so the wrapper has nothing
    to hand back but a repeat of the last live frame, and in the buffer
    I measured every single terminal frame was pixel-identical to its
    predecessor. Storing them taught the continue head "death = the
    picture froze" — a state the prior never dreams, which leaves
    continue-based return truncation inert exactly where it should
    bite. Dropping the repeat puts replay.py's zero continue target on
    the last frame the engine really rendered, so the head keys on what
    the frame shows — the fireball one step away — instead of on
    frame-freezing, and that is something imagination can and does
    produce. Nothing is lost with the row: the killing step pays no
    reward. Timeouts are untouched; their final frame is real.

    Campaign envs take a second path, taken whenever the env exposes
    finished_reason (DoomTakeCover and the ball envs do not, so their
    dicts come out of here exactly as they always have). An exit is the
    map ending in the way the whole campaign is about, and the engine
    drops its screen buffer there just as it does at death, so the
    exiting step's frame is another wrapper repeat and gets dropped the
    same way. What differs is the reward: that step pays the exit
    bonus, and the row it would have been stored against no longer
    exists, so the bonus is added onto the last stored row. The
    terminal label then sits on a real frame showing the exit switch up
    close, a cue the prior can dream, instead of on a duplicate, and
    the bonus arrives one step early, attached to standing at the
    switch rather than to pressing use at it. That asymmetry is the
    price of keeping the continue head off frame-freezing, and it is
    the same trade the death case already makes with the killing
    action. A "timeout" ending stores its final frame and leaves
    terminated False, and so does simply running out of max_steps: the
    clock belongs to the collector, not to the world, and a row whose
    continue target is 1 either way teaches nothing about termination.

    Those envs also carry state worth recording, so the dict grows two
    keys for them and only for them: variables (T+1, n_vars) float32
    when the env exposes variables(), one row per stored frame and
    dropped alongside any row that is dropped, and reason, the flavor
    the episode ended with.

    reward[0] follows the env. take_cover pays for being alive, so its
    reset frame pays the 0.01 every later frame pays; the campaign has
    no living reward, so its reset frame pays nothing and the reward
    head is never handed a value only reset frames have.
    """
    obs0 = env.reset()
    policy.reset(obs0)

    campaign = hasattr(env, "finished_reason")
    obs = [obs0]
    action = [np.zeros(env.action_dim, np.float32)]
    reward = [np.float32(0.0 if campaign else 0.01)]
    variables = [env.variables()] if hasattr(env, "variables") else None
    terminated = False
    reason = None

    frame = obs0
    for _ in range(max_steps):
        a = policy(frame)
        frame, r, done = env.step(a)
        if done:
            reason = (env.finished_reason if campaign
                      else "death" if env.died else "timeout")
        if reason in ("death", "exit"):
            # This step's frame is the wrapper's repeat, not an
            # observation, so it never enters the buffer. The row
            # already stored carries the 0 continue target, and on an
            # exit it also takes the bonus this step paid, since the
            # row that earned it is the one being dropped.
            terminated = True
            if reason == "exit":
                reward[-1] = np.float32(reward[-1] + r)
            break
        obs.append(frame)
        action.append(_as_action_vector(a, env.action_dim))
        reward.append(np.float32(r))
        if variables is not None:
            variables.append(env.variables())
        if done:
            break

    episode = {
        "obs": np.round(np.stack(obs) * 255.0).astype(np.uint8),
        "action": np.stack(action).astype(np.float32),
        "reward": np.asarray(reward, np.float32),
        "terminated": terminated,
    }
    if variables is not None:
        episode["variables"] = np.stack(variables).astype(np.float32)
    if campaign:
        # Reaching max_steps is the collector's own clock running out,
        # which is a timeout by any other name.
        episode["reason"] = reason or "timeout"
    return episode


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


class ScriptedExplorer:
    """Walk forward, turn when something is in the way, shoot and press
    use now and then: the seed-data policy for campaign maps.

    A uniformly random policy never leaves the first room of a Doom
    map, it spends its budget turning on the spot, and a world model
    cannot dream corridors nobody has ever walked down. So campaign
    pretraining data comes from this instead, and the use presses are
    what make an exit switch reachable at all before any reward exists
    to aim at it.

    It holds the env, deliberately. Bumps are read from position, which
    means calling env.variables() rather than looking at frames: this
    is a collection script, not a learned policy, and reading the
    simulator's own state is exactly the shortcut a data source is
    allowed to take. Nothing trained on the data it collects ever sees
    more than the frames.

    A bump is total distance traveled over the last bump_window steps
    coming in under bump_distance, in Doom map units. A walking player
    covers tens of them per policy step, so only something solid
    triggers it. The response is a random number of turn steps in one
    random direction, and the position history restarts empty
    afterwards, so the next bump has to be earned by fresh evidence
    rather than by the stall that was just dealt with.

    Actions come back as indices into env.ACTIONS, only ever from the
    enabled part of the vocabulary. All randomness is the constructor's
    seed; reset() does not reseed, matching RandomPolicy.
    """

    def __init__(self, env, seed: int, bump_window: int = 3,
                 bump_distance: float = 10.0, max_turn_steps: int = 6,
                 attack_prob: float = 0.05, use_prob: float = 0.05):
        self._env = env
        self._rng = np.random.default_rng(seed)
        self.bump_window = bump_window
        self.bump_distance = bump_distance
        self.max_turn_steps = max_turn_steps
        self.attack_prob = attack_prob
        self.use_prob = use_prob
        self._px = env.VARIABLE_NAMES.index("position_x")
        self._py = env.VARIABLE_NAMES.index("position_y")
        self._forward = env.ACTIONS.index("forward")
        self._attack = env.ACTIONS.index("attack")
        self._use = env.ACTIONS.index("use")
        self._turn_actions = (env.ACTIONS.index("turn_left"),
                              env.ACTIONS.index("turn_right"))
        self.reset(None)

    def reset(self, obs0):
        self._history = deque(maxlen=self.bump_window + 1)
        self._turn = self._forward
        self._turning = 0

    def __call__(self, obs) -> int:
        if self._turning > 0:
            # Mid-turn: no position is recorded, because turning in
            # place does not move and would look like a fresh bump.
            self._turning -= 1
            return self._turn
        v = self._env.variables()
        self._history.append((float(v[self._px]), float(v[self._py])))
        if self._bumped():
            self._history.clear()
            self._turn = self._turn_actions[int(self._rng.integers(2))]
            self._turning = int(
                self._rng.integers(1, self.max_turn_steps + 1)) - 1
            return self._turn
        draw = float(self._rng.random())
        if draw < self.attack_prob:
            return self._attack
        if draw < self.attack_prob + self.use_prob:
            return self._use
        return self._forward

    def _bumped(self) -> bool:
        if len(self._history) <= self.bump_window:
            return False
        steps = np.diff(np.asarray(self._history), axis=0)
        return float(np.linalg.norm(steps, axis=1).sum()) < self.bump_distance


class RSSMPolicy:
    """Acts from an RSSM belief state filtered online from raw frames.

    The whole step — encode, posterior, act, advance h — is a single
    jitted function, compiled once at construction. Four separate jits
    would be four host round-trips per env step, and at Doom's frame
    size that costs more than the engine does; only wm and act_fn are
    closed over, so one trace serves the policy's whole lifetime.

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
        self._key = key
        self._step = 0
        self._wm_params = None
        self._actor_params = None

        def policy_step(wm_params, actor_params, h, obs, key, step):
            e = wm.apply(wm_params, obs[None], method=RSSM.encode)
            z, _ = wm.apply(wm_params, h, e, method=RSSM.post_dist)
            s = jnp.concatenate([h[0], z[0]])
            action = jnp.asarray(
                act_fn(actor_params, s, jax.random.fold_in(key, step)),
                jnp.float32)
            h = wm.apply(wm_params, h, z, action[None],
                         method=RSSM.core_step)
            return action, h

        self._policy_step = jax.jit(policy_step)

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
        action, self._h = self._policy_step(
            self._wm_params, self._actor_params, self._h,
            jnp.asarray(obs), self._key, self._step)
        self._step += 1
        return np.asarray(action)
