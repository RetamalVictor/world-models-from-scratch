"""Actor and critic for imagination training (Step 4).

Both consume the RSSM's model state s = [h, z]. The actor emits a
tanh-squashed Gaussian scaled to the env's action bound, so everything
stays reparameterized and value gradients can flow through the dynamics.
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
