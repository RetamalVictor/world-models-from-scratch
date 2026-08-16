"""The prior rollout every imagination path shares.

Rolling the RSSM's prior forward under some action rule, reading the
reward and continue heads at each imagined state, is one scan whether
the actions come from an actor being trained or from a planner's
candidate sequence. train_ac.py and train_doom.py each carry a copy of
that scan inside their loss factories; this module holds the copy new
code calls, so the planner does not become a third.

Those two copies stay where they are on purpose. Both are under
experimental comparison and their numbers must not move, so folding
them into this function is a separate change that needs an
output-equality guard in front of it. Everything here is therefore
what train_doom's make_imagination does, in the same order: the key
splits first into an action key and a prior key, the action keys come
from splitting the first into `horizon` pieces, the prior noise is one
normal draw of shape (horizon,) + z0.shape, and the temperature
multiplies sigma rather than the sampled value. Change any of that and
the retrofit can no longer be proved equal.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from world_models.models.rssm import RSSM


def imagine_rollout(wm: RSSM, wm_params, action_fn, h0, z0, key,
                    horizon: int, temperature: float = 1.0):
    """Roll the frozen prior forward for `horizon` steps under action_fn.

    h0 (B, hidden) and z0 (B, latent_dim) are the start states, usually
    from posterior filtering on real frames. action_fn(step_key, s) ->
    (B, action_dim) float supplies the action for every row at once,
    from that step's key and the current model state s = [h, z]; the
    actor path and a planner's candidate path differ only in what they
    pass here. The rollout is batched over B, and rows never interact:
    each carries its own state, its own action and its own noise row.

    Returns (states, actions, rewards, continues) with states
    (H+1, B, s_dim) starting at s0 so states[i] is the state the
    action in actions[i] was taken from, actions (H, B, action_dim),
    and rewards and continues (H, B), where index i is the quantity of
    state s_{i+1}: the reward collected entering it and the
    probability the episode continues past it. continues comes out of
    the sigmoid, not as a logit, so `wm` must have been built with
    predict_continue=True.

    z is SAMPLED from the prior at every step, never taken as the
    mean: a mean rollout smears the stochastic events this model is
    supposed to dream (measured on take_cover, where mean rollouts
    almost never spawn a fireball), which would hand a planner a
    world nothing ever happens in. temperature multiplies the prior
    sigma, the 2018 World Models tau lever, neutral at 1.0 and exactly
    the prior mean at 0.0.

    No entropy term comes back, unlike train_doom's inner imagine. It
    belongs to the actor's loss rather than to the rollout, and it is
    recoverable without rerunning anything: the states the actor acted
    from are states[:-1].
    """
    s0 = jnp.concatenate([h0, z0], axis=-1)
    k_act, k_prior = jax.random.split(key)
    act_keys = jax.random.split(k_act, horizon)
    prior_noise = jax.random.normal(k_prior, (horizon,) + z0.shape)

    def step(carry, xs):
        h, z = carry
        k_t, pn = xs
        s = jnp.concatenate([h, z], axis=-1)
        a = jnp.asarray(action_fn(k_t, s), jnp.float32)
        h = wm.apply(wm_params, h, z, a, method=RSSM.core_step)
        mu_p, sig_p = wm.apply(wm_params, h, method=RSSM.prior_dist)
        z = mu_p + sig_p * temperature * pn
        r = wm.apply(wm_params, h, z, method=RSSM.reward)
        c = jax.nn.sigmoid(
            wm.apply(wm_params, h, z, method=RSSM.continue_logit))
        return (h, z), (jnp.concatenate([h, z], axis=-1), a, r, c)

    _, (states, actions, rewards, continues) = jax.lax.scan(
        step, (h0, z0), (act_keys, prior_noise))
    states = jnp.concatenate([s0[None], states], axis=0)
    return states, actions, rewards, continues
