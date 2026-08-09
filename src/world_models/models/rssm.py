"""Recurrent State-Space Model (Step 3, Dreamer-style).

One cell with four parts: a deterministic GRU core, a prior head that
must predict the next stochastic latent without seeing the frame, a
posterior head that gets to look, and a decoder over [h, z]. Everything
trains jointly against reconstruction + balanced KL; the KL between
prior and posterior is where the dynamics get learned.

The training/eval scans live in train_rssm.py and call the methods here
via model.apply(..., method=...), so the cell stays a plain single-step
object like the GRU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from world_models.models.vae import Decoder


class ConvEncoder(nn.Module):
    """Step-1 conv trunk, but emitting an embedding instead of (mu, logvar)."""
    embed_dim: int = 256

    @nn.compact
    def __call__(self, o):  # (B, 32, 32, 1)
        x = nn.silu(nn.Conv(32, (4, 4), strides=(2, 2))(o))
        x = nn.silu(nn.Conv(64, (4, 4), strides=(2, 2))(x))
        x = nn.silu(nn.Conv(128, (4, 4), strides=(2, 2))(x))
        x = x.reshape((x.shape[0], -1))
        return nn.silu(nn.Dense(self.embed_dim)(x))


class GaussianHead(nn.Module):
    latent_dim: int = 16
    min_sigma: float = 0.1

    @nn.compact
    def __call__(self, x):
        x = nn.silu(nn.Dense(128)(x))
        mu = nn.Dense(self.latent_dim)(x)
        sigma = jax.nn.softplus(nn.Dense(self.latent_dim)(x)) + self.min_sigma
        return mu, sigma


class RSSM(nn.Module):
    latent_dim: int = 16
    action_dim: int = 2
    hidden: int = 128
    min_sigma: float = 0.1

    def setup(self):
        self.core = nn.GRUCell(features=self.hidden)
        self.encoder = ConvEncoder()
        self.prior_head = GaussianHead(self.latent_dim, self.min_sigma)
        self.post_head = GaussianHead(self.latent_dim, self.min_sigma)
        self.decoder = Decoder()

    def __call__(self, o, h, z, a):
        """Exercise every submodule once — used only for init."""
        e = self.encoder(o)
        h = self.core_step(h, z, a)
        mu_p, sig_p = self.prior_dist(h)
        mu_q, sig_q = self.post_dist(h, e)
        return self.decode(h, mu_q), (mu_p, sig_p), (mu_q, sig_q)

    def encode(self, o):
        return self.encoder(o)

    def core_step(self, h, z_prev, action):
        x = jnp.concatenate([z_prev, action], axis=-1)
        h, _ = self.core(h, x)
        return h

    def prior_dist(self, h):
        return self.prior_head(h)

    def post_dist(self, h, e):
        return self.post_head(jnp.concatenate([h, e], axis=-1))

    def decode(self, h, z):
        return self.decoder(jnp.concatenate([h, z], axis=-1))

    def initial_state(self, batch_size: int) -> jnp.ndarray:
        return jnp.zeros((batch_size, self.hidden))


def kl_gauss(mu_q, sig_q, mu_p, sig_p):
    """KL( N(mu_q, sig_q) || N(mu_p, sig_p) ), summed over latent dims."""
    return jnp.sum(
        jnp.log(sig_p / sig_q)
        + (sig_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sig_p ** 2)
        - 0.5,
        axis=-1,
    )


def kl_balanced(mu_q, sig_q, mu_p, sig_p, alpha: float = 0.8):
    """KL balancing (Dreamer): the same KL, but the gradient is split so
    the prior is pulled toward the posterior with weight alpha and the
    posterior regularized toward the prior with weight 1 - alpha."""
    sg = jax.lax.stop_gradient
    toward_prior = kl_gauss(sg(mu_q), sg(sig_q), mu_p, sig_p)
    toward_post = kl_gauss(mu_q, sig_q, sg(mu_p), sg(sig_p))
    return alpha * toward_prior + (1.0 - alpha) * toward_post
