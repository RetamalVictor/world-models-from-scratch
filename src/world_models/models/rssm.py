"""Recurrent State-Space Model (Step 3, Dreamer-style).

One cell with four parts: a deterministic GRU core, a prior head that
must predict the next stochastic latent without seeing the frame, a
posterior head that gets to look, and a decoder over [h, z]. Everything
trains jointly against reconstruction + balanced KL; the KL between
prior and posterior is where the dynamics get learned.

The training/eval scans live in train_rssm.py and call the methods here
via model.apply(..., method=...), so the cell stays a plain single-step
object like the GRU.

Two knobs exist for Doom, both defaulting to the Step 3 model: obs_size
picks the conv ladder (32 -> three stride-2 stages, 64 -> four) and
predict_continue adds the Bernoulli death head. On the defaults the
parameter tree is the one Steps 3-4 trained, so their checkpoints load
unchanged.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import linen as nn


def _channel_ladder(obs_size: int) -> tuple[int, ...]:
    """Stride-2 stages from obs_size down to a 4x4 bottleneck.

    32 -> (32, 64, 128), 64 -> (32, 64, 128, 256). The bottleneck is
    fixed at 4x4 because the decoder's first Dense is sized from it.
    """
    stages = int(round(math.log2(obs_size / 4.0)))
    if stages < 1 or 4 * 2 ** stages != obs_size:
        raise ValueError(
            f"obs_size must be 4 * 2**k with k >= 1, got {obs_size}")
    return tuple(32 * 2 ** i for i in range(stages))


class ConvEncoder(nn.Module):
    """Step-1 conv trunk, but emitting an embedding instead of (mu, logvar)."""
    embed_dim: int = 256
    obs_size: int = 32

    @nn.compact
    def __call__(self, o):  # (B, obs_size, obs_size, C)
        x = o
        for channels in _channel_ladder(self.obs_size):
            x = nn.silu(nn.Conv(channels, (4, 4), strides=(2, 2))(x))
        x = x.reshape((x.shape[0], -1))
        return nn.silu(nn.Dense(self.embed_dim)(x))


class ConvDecoder(nn.Module):
    """Mirror of ConvEncoder: 4x4 bottleneck back up to obs_size.

    Same layer sequence as the Step-1 VAE decoder at obs_size 32, so an
    RSSM checkpoint from Steps 3-4 restores into it unchanged.
    """
    out_channels: int = 1
    embed_dim: int = 256
    obs_size: int = 32

    @nn.compact
    def __call__(self, z):  # (B, hidden + latent_dim)
        ladder = _channel_ladder(self.obs_size)
        x = nn.silu(nn.Dense(self.embed_dim)(z))
        x = nn.silu(nn.Dense(4 * 4 * ladder[-1])(x))
        x = x.reshape((-1, 4, 4, ladder[-1]))
        for channels in ladder[-2::-1]:
            x = nn.silu(nn.ConvTranspose(channels, (4, 4), strides=(2, 2))(x))
        return nn.ConvTranspose(self.out_channels, (4, 4), strides=(2, 2))(x)


class GaussianHead(nn.Module):
    latent_dim: int = 16
    min_sigma: float = 0.1

    @nn.compact
    def __call__(self, x):
        x = nn.silu(nn.Dense(128)(x))
        mu = nn.Dense(self.latent_dim)(x)
        sigma = jax.nn.softplus(nn.Dense(self.latent_dim)(x)) + self.min_sigma
        return mu, sigma


class RewardHead(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.silu(nn.Dense(128)(x))
        return nn.Dense(1)(x)[..., 0]


class ContinueHead(nn.Module):
    """Bernoulli continue flag: emits a logit, not a probability."""
    @nn.compact
    def __call__(self, x):
        x = nn.silu(nn.Dense(128)(x))
        return nn.Dense(1)(x)[..., 0]


class RSSM(nn.Module):
    latent_dim: int = 16
    action_dim: int = 2
    hidden: int = 128
    min_sigma: float = 0.1
    obs_channels: int = 1
    obs_size: int = 32
    predict_continue: bool = False

    def setup(self):
        self.core = nn.GRUCell(features=self.hidden)
        self.encoder = ConvEncoder(obs_size=self.obs_size)
        self.prior_head = GaussianHead(self.latent_dim, self.min_sigma)
        self.post_head = GaussianHead(self.latent_dim, self.min_sigma)
        self.decoder = ConvDecoder(out_channels=self.obs_channels,
                                   obs_size=self.obs_size)
        self.reward_head = RewardHead()
        if self.predict_continue:
            self.continue_head = ContinueHead()

    def __call__(self, o, h, z, a):
        """Exercise every submodule once — used only for init.

        The continue logit is appended as a fifth output when the head
        exists, so callers that predate it keep unpacking four.
        """
        e = self.encoder(o)
        h = self.core_step(h, z, a)
        mu_p, sig_p = self.prior_dist(h)
        mu_q, sig_q = self.post_dist(h, e)
        out = (self.decode(h, mu_q), (mu_p, sig_p), (mu_q, sig_q),
               self.reward(h, mu_q))
        if self.predict_continue:
            out += (self.continue_logit(h, mu_q),)
        return out

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

    def reward(self, h, z):
        return self.reward_head(jnp.concatenate([h, z], axis=-1))

    def continue_logit(self, h, z):
        return self.continue_head(jnp.concatenate([h, z], axis=-1))

    def initial_state(self, batch_size: int) -> jnp.ndarray:
        return jnp.zeros((batch_size, self.hidden))


def load_rssm(run_dir):
    """Load a trained RSSM from a run directory -> (model, params)."""
    import json
    from pathlib import Path

    from flax import serialization

    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    # The .get defaults are the Steps 0-4 architecture: runs recorded
    # before these knobs existed rebuild exactly as they were trained.
    model = RSSM(
        latent_dim=cfg["latent_dim"],
        action_dim=cfg.get("action_dim", 2),
        hidden=cfg["hidden"],
        min_sigma=cfg["min_sigma"],
        obs_channels=cfg.get("obs_channels", 1),
        obs_size=cfg.get("obs_size", 32),
        predict_continue=cfg.get("predict_continue", False),
    )
    c, s = model.obs_channels, model.obs_size
    template = model.init(
        jax.random.PRNGKey(0), jnp.zeros((1, s, s, c)),
        model.initial_state(1), jnp.zeros((1, model.latent_dim)),
        jnp.zeros((1, model.action_dim)),
    )
    params = serialization.from_bytes(
        template, (run_dir / "checkpoint.msgpack").read_bytes()
    )
    return model, params


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


def continue_bce(logits, targets, death_weight: float = 1.0):
    """Bernoulli NLL of the continue flag, elementwise, from logits.

    targets: 1.0 where the episode continues past this frame, 0.0 on the
    frame where death ends it. Written in the stable max-form rather
    than log(sigmoid(x)) so a confident head can't produce a NaN.

    death_weight scales the zero-target frames only. Deaths are one
    frame per episode against dozens of alive ones, so unweighted the
    cheapest head is the one that says "alive" everywhere and never
    learns the cue; upweighting the rare class is the standard fix. At
    the default 1.0 the multiplier is exactly one, so every caller that
    predates the knob gets the same float it always did.
    """
    bce = (jnp.maximum(logits, 0.0) - logits * targets
           + jnp.log1p(jnp.exp(-jnp.abs(logits))))
    return bce * (1.0 + (death_weight - 1.0) * (1.0 - targets))
