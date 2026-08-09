"""Frozen-encoder GRU dynamics (Step 2, Ha-style).

Single-step cell: consumes the previous latent and the action for the
transition, updates the hidden state, and emits a Gaussian over the next
latent. Sequences are handled with lax.scan in the training/eval code so
the same cell serves teacher-forced training and open-loop rollout.

With residual=True the head predicts a mean *offset* and the likelihood
becomes N(z_t | z_{t-1} + mu_delta, sigma). Both variants put a density
on the same variable, so their NLLs are directly comparable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


class GRUDynamics(nn.Module):
    latent_dim: int = 16
    action_dim: int = 2
    hidden: int = 128
    residual: bool = False

    @nn.compact
    def __call__(self, h, z_prev, action):
        x = jnp.concatenate([z_prev, action], axis=-1)
        h, _ = nn.GRUCell(features=self.hidden)(h, x)
        y = nn.silu(nn.Dense(128)(h))
        mu = nn.Dense(self.latent_dim)(y)
        sigma = jax.nn.softplus(nn.Dense(self.latent_dim)(y)) + 1e-3
        if self.residual:
            mu = z_prev + mu
        return h, mu, sigma

    def initial_state(self, batch_size: int) -> jnp.ndarray:
        return jnp.zeros((batch_size, self.hidden))


def gaussian_nll(z, mu, sigma):
    """-log N(z | mu, sigma), summed over the latent dims. Shape (...,)."""
    return 0.5 * jnp.sum(
        ((z - mu) / sigma) ** 2 + 2.0 * jnp.log(sigma) + jnp.log(2.0 * jnp.pi),
        axis=-1,
    )


def sequence_nll(model, params, z_seq, a_seq, loss_mask):
    """Teacher-forced NLL over time-major sequences.

    z_seq: (T+1, B, D) latents, a_seq: (T+1, B, A) actions where a_seq[t]
    is the action of the transition t-1 -> t (the env stores it that way,
    with a zero action on the reset frame). loss_mask: (T,) with zeros on
    the warm-up transitions.

    Returns (mean masked NLL, h_seq of shape (T, B, hidden)) — h_seq[t]
    is the state that predicted z_{t+1}, i.e. history up to frame t.
    """
    h0 = model.initial_state(z_seq.shape[1])

    def step(h, inp):
        z_prev, a, z_next = inp
        h, mu, sigma = model.apply(params, h, z_prev, a)
        return h, (gaussian_nll(z_next, mu, sigma), h)

    _, (nlls, h_seq) = jax.lax.scan(
        step, h0, (z_seq[:-1], a_seq[1:], z_seq[1:])
    )
    total = (nlls * loss_mask[:, None]).sum()
    return total / (loss_mask.sum() * z_seq.shape[1]), h_seq
