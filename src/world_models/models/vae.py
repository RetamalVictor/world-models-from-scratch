"""Convolutional VAE for 32x32x1 frames (Step 1).

Encoder halves the resolution three times (32 -> 16 -> 8 -> 4), the decoder
mirrors it with transposed convs. The decoder output is linear and read as
the mean of a fixed-variance Gaussian, so the reconstruction loss is MSE
summed over pixels.
"""

from __future__ import annotations

import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn


class Encoder(nn.Module):
    latent_dim: int = 16

    @nn.compact
    def __call__(self, x):  # (B, 32, 32, 1)
        x = nn.silu(nn.Conv(32, (4, 4), strides=(2, 2))(x))    # (B, 16, 16, 32)
        x = nn.silu(nn.Conv(64, (4, 4), strides=(2, 2))(x))    # (B, 8, 8, 64)
        x = nn.silu(nn.Conv(128, (4, 4), strides=(2, 2))(x))   # (B, 4, 4, 128)
        x = x.reshape((x.shape[0], -1))
        x = nn.silu(nn.Dense(256)(x))
        mu = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        return mu, logvar


class Decoder(nn.Module):
    out_channels: int = 1

    @nn.compact
    def __call__(self, z):  # (B, latent_dim)
        x = nn.silu(nn.Dense(256)(z))
        x = nn.silu(nn.Dense(4 * 4 * 128)(x))
        x = x.reshape((-1, 4, 4, 128))
        x = nn.silu(nn.ConvTranspose(64, (4, 4), strides=(2, 2))(x))  # (B, 8, 8, 64)
        x = nn.silu(nn.ConvTranspose(32, (4, 4), strides=(2, 2))(x))  # (B, 16, 16, 32)
        x = nn.ConvTranspose(self.out_channels, (4, 4), strides=(2, 2))(x)
        return x                                                      # (B, 32, 32, C)


class VAE(nn.Module):
    latent_dim: int = 16

    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder()

    def __call__(self, x, key):
        mu, logvar = self.encoder(x)
        std = jnp.exp(0.5 * logvar)
        z = mu + std * jax.random.normal(key, mu.shape)  # reparameterization
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_vae(run_dir):
    """Load a trained VAE from a run directory -> (model, params)."""
    import json
    from pathlib import Path

    from flax import serialization

    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    model = VAE(latent_dim=cfg["latent_dim"])
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    template = model.init(k1, jnp.zeros((1, 32, 32, 1)), k2)
    params = serialization.from_bytes(
        template, (run_dir / "checkpoint.msgpack").read_bytes()
    )
    return model, params


def elbo_terms(recon, x, mu, logvar):
    """(recon_error, kl), each averaged over the batch.

    Reconstruction is summed over pixels, KL summed over latent dims, so the
    two terms live on comparable scales and beta stays interpretable.
    """
    recon_err = ((recon - x) ** 2).sum(axis=(1, 2, 3)).mean()
    q = distrax.Independent(distrax.Normal(mu, jnp.exp(0.5 * logvar)), 1)
    p = distrax.Independent(
        distrax.Normal(jnp.zeros_like(mu), jnp.ones_like(mu)), 1
    )
    kl = q.kl_divergence(p).mean()
    return recon_err, kl
