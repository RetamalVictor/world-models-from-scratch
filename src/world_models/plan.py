"""Decision-time planning in the RSSM's imagination (model-predictive
control over a trained world model).

The actor answers "what now" with one forward pass. The planner spends
imagination instead: at every real frame it dreams n_candidates action
sequences horizon steps deep, scores each dream with the reward and
continue heads (and the critic, on the tail), and plays the first
action of the winner. Nothing here trains. World model, actor and
critic arrive frozen as call-time params, exactly as they do for
RSSMPolicy, and the only thing that changes between steps is the
belief.

Two properties are worth stating before the code, because both are
measured facts about this stack rather than preferences.

The whole decision is ONE jitted function. Per-frame cost on this
stack floors out at dispatch latency, not at compute, so a plan split
across several jitted calls pays more in round-trips than the candidate
rollout itself costs. Everything from the encoder to the advanced
belief lives inside `plan_step`.

Candidates are a mixture, not one distribution. Most sample from the
actor at a temperature, so the planner starts from what the policy
already knows and is never worse informed than it; the rest sample
uniformly, so a bad habit can be outvoted by something the actor would
never have proposed.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from world_models.imagination import imagine_rollout
from world_models.models.rssm import RSSM

# Disabled actions get a large finite negative logit instead of -inf.
# Sampling adds Gumbel noise, which is single digits in practice, so
# -1e9 is already unreachable, and a finite offset cannot turn into a
# NaN if this path is ever differentiated through.
_DISABLED_LOGIT = -1e9


@dataclass(frozen=True)
class MPCConfig:
    """Planner knobs. The defaults are the first evaluation configuration.

    temperature divides the actor's logits before sampling candidates,
    so it must be positive; small values approach the greedy actor and
    large ones approach uniform. prior_temperature is the separate
    lever that multiplies the prior's sigma inside the dream, the same
    tau train_doom exposes as imagination_temperature, neutral at 1.0.

    replan_every > 1 executes that many actions from the winning plan
    before planning again, which buys throughput at the cost of acting
    on a stale plan; it must not exceed horizon, since the plan is only
    that long.
    """
    n_candidates: int = 64
    horizon: int = 12
    actor_frac: float = 0.8
    temperature: float = 1.0
    prior_temperature: float = 1.0
    discount: float = 0.99
    use_value_tail: bool = True
    replan_every: int = 1


def score_rollout(rewards, continues, discount: float, tail_value=None):
    """Discounted imagined return, truncated by predicted continues.

    rewards and continues are (H, B) as imagine_rollout returns them,
    so index i belongs to state s_{i+1}. Weight of reward i is
    discount**i times the product of every earlier continue: a step the
    model expects not to survive to contributes little, and nothing
    behind it contributes at all. This is what lambda_returns computes
    at lam=1 (its per-step discount is gamma * continue), written
    forward because a planner wants one number per candidate rather
    than a return per state.

    tail_value (B,) is the critic's value of the final imagined state,
    standing in for everything past the horizon. It is weighted by
    discount**H and the full continue product, the same factor the
    reward one step earlier would have carried. None omits it, which is
    the no-value-tail arm of the planner ablation.
    """
    horizon = rewards.shape[0]
    # survival[i] = probability of reaching s_i, so survival[0] is 1 and
    # survival[H] is the whole product: one array serves the rewards and
    # the tail.
    survival = jnp.concatenate(
        [jnp.ones_like(continues[:1]), jnp.cumprod(continues, axis=0)],
        axis=0)
    powers = discount ** jnp.arange(horizon + 1, dtype=rewards.dtype)
    weights = powers[:, None] * survival
    score = (weights[:horizon] * rewards).sum(axis=0)
    if tail_value is not None:
        score = score + weights[horizon] * tail_value
    return score


def make_planner(wm: RSSM, logits_fn, value_fn, action_dim: int,
                 config: MPCConfig, action_mask=None):
    """Close the model and the scoring rules over their config.

    Returns (score_candidates, plan), both pure functions of params and
    a belief, so a test can plan against a hand-built RSSM with no
    environment, no policy object and no jit anywhere in sight.

        score_candidates(wm_params, actor_params, critic_params,
                         h, z, key) -> (actions, scores)
        plan(wm_params, actor_params, critic_params, h, z, key)
                                    -> actions of the winner

    h (hidden,) and z (latent_dim,) are one belief, tiled to
    n_candidates inside; actions is (horizon, n_candidates, action_dim)
    and scores is (n_candidates,); plan returns (horizon, action_dim).

    logits_fn(actor_params, s) -> (B, action_dim) logits and
    value_fn(critic_params, s) -> (B,) values are supplied by the
    caller, the way RSSMPolicy takes act_fn: this module imports no
    actor and no critic, so it stays usable with whatever heads a run
    happens to have trained. Raw logits rather than actions, because
    the planner needs to sample them at its own temperature. Either may
    be None when the config makes it unreachable (actor_frac 0,
    use_value_tail off); asking for one without supplying it raises
    here rather than at the first frame.

    action_mask (action_dim,) booleans marks which slots the env will
    accept, defaulting to all of them. It is applied to BOTH candidate
    sources, not just the uniform one: an actor trained with its own
    availability mask already avoids the disabled slots and loses
    nothing here, and an actor trained without one would otherwise be
    able to make the planner propose an action the env rejects.
    """
    n = config.n_candidates
    n_actor = int(round(n * config.actor_frac))
    if n_actor > 0 and logits_fn is None:
        raise ValueError("actor_frac > 0 needs a logits_fn to sample from")
    if config.use_value_tail and value_fn is None:
        raise ValueError("use_value_tail needs a value_fn")
    if not 1 <= config.replan_every <= config.horizon:
        raise ValueError(
            f"replan_every must be in 1..horizon ({config.horizon}), "
            f"got {config.replan_every}")

    mask = None if action_mask is None else jnp.asarray(action_mask, bool)
    if mask is not None and mask.shape != (action_dim,):
        raise ValueError(
            f"action_mask must be ({action_dim},), got {mask.shape}")
    base = (jnp.zeros(action_dim, jnp.float32) if mask is None
            else jnp.where(mask, 0.0, _DISABLED_LOGIT).astype(jnp.float32))
    uniform_logits = jnp.broadcast_to(base, (n, action_dim))
    # Candidate 0 is an actor candidate whenever there is one, so the
    # policy's own preference is always on the ballot.
    is_actor = jnp.arange(n) < n_actor

    def score_candidates(wm_params, actor_params, critic_params, h, z, key):
        def action_fn(step_key, s):
            # Both branches draw from their own key, so the uniform
            # candidates do not shift when the actor's logits change.
            k_actor, k_uniform = jax.random.split(step_key)
            if n_actor == 0:
                return jax.nn.one_hot(
                    jax.random.categorical(k_uniform, uniform_logits),
                    action_dim)
            logits = jnp.asarray(logits_fn(actor_params, s), jnp.float32)
            if mask is not None:
                logits = jnp.where(mask, logits, _DISABLED_LOGIT)
            i = jax.random.categorical(k_actor, logits / config.temperature)
            if n_actor < n:
                i = jnp.where(is_actor, i,
                              jax.random.categorical(k_uniform,
                                                     uniform_logits))
            return jax.nn.one_hot(i, action_dim)

        h0 = jnp.broadcast_to(h, (n,) + h.shape)
        z0 = jnp.broadcast_to(z, (n,) + z.shape)
        states, actions, rewards, continues = imagine_rollout(
            wm, wm_params, action_fn, h0, z0, key, config.horizon,
            config.prior_temperature)
        tail = (value_fn(critic_params, states[-1])
                if config.use_value_tail else None)
        scores = score_rollout(rewards, continues, config.discount, tail)
        return actions, scores

    def plan(wm_params, actor_params, critic_params, h, z, key):
        actions, scores = score_candidates(
            wm_params, actor_params, critic_params, h, z, key)
        return actions[:, jnp.argmax(scores), :]

    return score_candidates, plan


class MPCPolicy:
    """Plans each action from an RSSM belief filtered online from frames.

    Drop-in for RSSMPolicy in collect.py: reset(obs0), __call__(obs) ->
    action, plus set_params and reseed, so collect_episode and every
    eval script take it without knowing which policy they hold. What it
    costs is compute per frame, n_candidates times horizon prior steps
    of it, which is why the training loop keeps collecting with the
    actor and the planner earns its place at evaluation first.

    Constructed like RSSMPolicy and for the same reasons: the model and
    the head functions are closed over so one trace serves the policy's
    whole lifetime, while every set of params is a call-time argument
    so pointing it at a newer checkpoint never forces a fresh trace.

    Ordering is RSSMPolicy's correct-then-act: fold the incoming frame
    into the belief, plan and act from the updated state, and only then
    advance h with the action that was played. z is the posterior mean
    while filtering (this is acting, not training, so there is nothing
    to sample noise for); z inside the dream is sampled, which is a
    different question and is answered in imagination.py.

    With replan_every > 1 the winning plan is carried between steps and
    the next few actions come out of it unchanged. The belief still
    updates on every frame, so only the plan is stale, never the state
    it was planned from.
    """

    def __init__(self, wm: RSSM, logits_fn, value_fn, key, action_dim: int,
                 config: MPCConfig | None = None, action_mask=None):
        self.action_dim = action_dim
        self.config = config or MPCConfig()
        self._wm = wm
        self._key = key
        self._step = 0
        self._wm_params = None
        self._actor_params = None
        self._critic_params = None
        cfg = self.config

        _, plan = make_planner(wm, logits_fn, value_fn, action_dim, cfg,
                               action_mask)

        def plan_step(wm_params, actor_params, critic_params, h, cached,
                      obs, key, step):
            e = wm.apply(wm_params, obs[None], method=RSSM.encode)
            z, _ = wm.apply(wm_params, h, e, method=RSSM.post_dist)
            plan_key = jax.random.fold_in(key, step)
            if cfg.replan_every == 1:
                # The common case keeps the cond out of the graph
                # entirely: the phase is a compile-time zero.
                cached = plan(wm_params, actor_params, critic_params,
                              h[0], z[0], plan_key)
                phase = 0
            else:
                phase = step % cfg.replan_every
                cached = jax.lax.cond(
                    phase == 0,
                    lambda: plan(wm_params, actor_params, critic_params,
                                 h[0], z[0], plan_key),
                    lambda: cached)
            action = cached[phase]
            h = wm.apply(wm_params, h, z, action[None], method=RSSM.core_step)
            return action, h, cached

        self._plan_step = jax.jit(plan_step)

    def set_params(self, wm_params, actor_params=None, critic_params=None):
        self._wm_params = wm_params
        self._actor_params = actor_params
        self._critic_params = critic_params

    def reseed(self, key):
        """New planning key; per-episode, from the caller's lane."""
        self._key = key
        self._step = 0

    def reset(self, obs0):
        # h starts blank and the first __call__ folds in the reset frame,
        # as in RSSMPolicy. The cached plan starts at zeros and is never
        # read: step 0 is phase 0, which always replans.
        self._h = self._wm.initial_state(1)
        self._plan = jnp.zeros((self.config.horizon, self.action_dim),
                               jnp.float32)
        self._step = 0

    def __call__(self, obs) -> np.ndarray:
        action, self._h, self._plan = self._plan_step(
            self._wm_params, self._actor_params, self._critic_params,
            self._h, self._plan, jnp.asarray(obs), self._key, self._step)
        self._step += 1
        return np.asarray(action)
