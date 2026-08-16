"""Actor and critic for imagination training (Step 4).

Both consume the RSSM's model state s = [h, z]. The actor emits a
tanh-squashed Gaussian scaled to the env's action bound, so everything
stays reparameterized and value gradients can flow through the dynamics.

DiscreteActor is the same idea for an env whose actions are buttons:
a categorical head, kept differentiable by a straight-through sample.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


class Actor(nn.Module):
    action_dim: int = 2
    max_action: float = 0.3
    min_sigma: float = 0.1

    @nn.compact
    def __call__(self, s):
        x = nn.silu(nn.Dense(128)(s))
        x = nn.silu(nn.Dense(128)(x))
        mu = nn.Dense(self.action_dim)(x)
        sigma = jax.nn.softplus(nn.Dense(self.action_dim)(x)) + self.min_sigma
        return mu, sigma

    def act(self, s, key=None):
        """Sampled (training) or deterministic (key=None) bounded action."""
        mu, sigma = self(s)
        pre = mu if key is None else mu + sigma * jax.random.normal(key, mu.shape)
        return self.max_action * jnp.tanh(pre)


# A large finite offset rather than -inf: -inf pushed straight into
# softmax/log_softmax leaves the gradient at that logit as inf - inf,
# i.e. NaN, and that NaN spreads through the whole batch on the next
# backward pass even though the disabled action itself never needed a
# gradient. 1e9 dwarfs anything a trunk this size ever emits as a raw
# logit, so the masked slots still carry effectively zero probability.
MASKED_LOGIT_OFFSET = -1e9


def _mask_logits(logits, action_mask):
    """Push disabled actions' logits far down without using -inf.

    action_mask is boolean, broadcastable against logits' last axis,
    True marking an enabled action. None means every action is enabled
    and logits comes back untouched, so a caller that never passes a
    mask sees byte-identical behavior to before masking existed.
    """
    if action_mask is None:
        return logits
    return logits + jnp.where(action_mask, 0.0, MASKED_LOGIT_OFFSET)


class DiscreteActor(nn.Module):
    """Categorical actor over one-hot actions.

    Doom's take_cover offers {left, right, noop} and nothing in between,
    which a tanh-Gaussian cannot express. Same trunk as Actor; the head
    emits logits instead of (mu, sigma).

    Every action-distribution method takes an optional action_mask
    (boolean, action_dim, default None): DoomCampaign's action space
    reserves a few one-hot slots that are not choosable yet, and the
    mask keeps the network's output width fixed at action_dim while
    still excluding those slots from sampling, the greedy action,
    log-prob, and entropy. With action_mask left at None, nothing about
    this class's behavior or its param tree changes from before masking
    existed.
    """
    action_dim: int = 3

    @nn.compact
    def __call__(self, s):
        x = nn.silu(nn.Dense(128)(s))
        x = nn.silu(nn.Dense(128)(x))
        return nn.Dense(self.action_dim)(x)

    def act(self, s, key=None, action_mask=None):
        """Sampled (training) or greedy (key=None) one-hot action.

        Plain action selection for talking to the env — no gradient
        tricks; use sample_st inside imagination.
        """
        logits = _mask_logits(self(s), action_mask)
        i = (jnp.argmax(logits, axis=-1) if key is None
             else jax.random.categorical(key, logits))
        return jax.nn.one_hot(i, self.action_dim)

    def sample_st(self, s, key, action_mask=None):
        """Straight-through one-hot sample, for imagination.

        Forward pass carries the hard one-hot, so the dynamics see a
        real action; backward pass carries the softmax probabilities,
        so value gradients reach the logits even though the sample
        itself is discrete and has no derivative of its own. A masked
        action never wins the categorical draw and its near-zero
        softmax share contributes nothing worth mentioning to the
        backward pass either.
        """
        logits = _mask_logits(self(s), action_mask)
        probs = jax.nn.softmax(logits)
        one_hot = jax.nn.one_hot(jax.random.categorical(key, logits),
                                 self.action_dim)
        return one_hot + probs - jax.lax.stop_gradient(probs)

    def log_prob(self, s, action, action_mask=None):
        """Log probability the (masked) distribution assigns to a
        one-hot action, for policy-gradient style losses that need the
        probability of the action actually taken rather than a sample."""
        log_p = jax.nn.log_softmax(_mask_logits(self(s), action_mask))
        return (log_p * action).sum(axis=-1)

    def entropy(self, s, action_mask=None):
        """Categorical entropy, via log_softmax so confident logits
        can't take a log of zero. A masked action's near-zero
        probability makes its own term vanish, so this is the entropy
        of the smaller distribution over the remaining actions."""
        log_p = jax.nn.log_softmax(_mask_logits(self(s), action_mask))
        return -(jnp.exp(log_p) * log_p).sum(axis=-1)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, s):
        x = nn.silu(nn.Dense(128)(s))
        x = nn.silu(nn.Dense(128)(x))
        return nn.Dense(1)(x)[..., 0]


def lambda_returns(rewards, values, gamma: float, lam: float,
                   continues=None):
    """TD(lambda) returns, computed backwards.

    rewards[i] is the reward received entering state s_{i+1}; values[i]
    is the critic's value of s_{i+1}. Both (H, B). Returns (H, B) where
    out[i] is the lambda-return of state s_i.

    continues[i] (H, B) is the probability the episode survives into
    s_{i+1}, so the per-step discount becomes gamma * continues[i] and
    a predicted death truncates the return instead of bootstrapping
    through it — without it, imagination happily values what happens
    after you die. None means the episode always continues, which is
    the ball and every caller written before the continue head.
    """
    if continues is None:
        discounts = jnp.full_like(rewards, gamma)
    else:
        discounts = gamma * continues

    def step(carry, xs):
        next_return = carry
        r, v, d = xs
        ret = r + d * ((1.0 - lam) * v + lam * next_return)
        return ret, ret

    bootstrap = values[-1]
    _, returns = jax.lax.scan(
        step, bootstrap, (rewards, values, discounts), reverse=True
    )
    return returns
