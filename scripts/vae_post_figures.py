"""Figures for the first two posts (docs/posts/).

Run from the repo root of the clone that has the runs/ directory (the WSL
clone):

    uv run python scripts/vae_post_figures.py

Reads the base / beta01 / warmup5k checkpoints, recomputes the probe
predictions on the test episodes, and writes pngs + a rollout gif into
docs/media/.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from flax import serialization
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world_models import data as data_lib
from world_models.models.vae import VAE
from world_models.probe import r2_score, ridge_fit

MEDIA = Path("docs/media")
RUNS = Path("runs/vae")

# Chart chrome (reference palette, light mode)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"    # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2
SURFACE = "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def load_run(name):
    run_dir = RUNS / name
    cfg = json.loads((run_dir / "config.json").read_text())
    model = VAE(latent_dim=cfg["latent_dim"])
    dummy = jnp.zeros((1, 32, 32, 1))
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    template = model.init(k1, dummy, k2)
    params = serialization.from_bytes(
        template, (run_dir / "checkpoint.msgpack").read_bytes()
    )
    return model, params


def encode_test(model, params, dataset, test_split):
    obs = dataset["obs"][test_split]
    n_ep, n_t = obs.shape[:2]
    frames = data_lib.to_float(obs.reshape((-1,) + obs.shape[2:]))
    mu, _ = model.apply(params, frames, method=VAE.encode)
    return np.asarray(mu).reshape(n_ep, n_t, -1)


def probe_predictions(latents, targets):
    """Same protocol as train_vae: fit first half of episodes, eval second."""
    half = latents.shape[0] // 2
    d = latents.shape[-1]
    k = targets.shape[-1]
    w, b = ridge_fit(
        jnp.asarray(latents[:half].reshape(-1, d)),
        jnp.asarray(targets[:half].reshape(-1, k)),
    )
    y_eval = jnp.asarray(targets[half:].reshape(-1, k))
    pred = jnp.asarray(latents[half:].reshape(-1, d)) @ w + b
    return np.asarray(y_eval), np.asarray(pred), np.asarray(r2_score(y_eval, pred))


def scatter_panel(ax, true, pred, r2_mean, title):
    lim = (0, 32)
    ax.plot(lim, lim, ls="--", lw=1, color=BASELINE, zorder=1)
    ax.scatter(true, pred, s=4, alpha=0.12, color=BLUE, edgecolors="none", zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("true position (px, x and y pooled)")
    ax.set_ylabel("linear probe prediction (px)")
    ax.set_title(title, fontsize=10)
    ax.text(0.05, 0.92, f"R² = {r2_mean:.2f}", transform=ax.transAxes,
            color=INK2, fontsize=10)
    style(ax)


def rollout_gif(model, params, episode_obs, out_path, scale=4):
    """Side-by-side gif: data on the left, mean reconstruction on the right."""
    frames = data_lib.to_float(episode_obs)
    mu, _ = model.apply(params, frames, method=VAE.encode)
    recon = np.clip(np.asarray(model.apply(params, mu, method=VAE.decode)), 0, 1)
    data = np.asarray(frames)
    divider = np.full((data.shape[1], 2), 1.0)
    imgs = []
    for t in range(data.shape[0]):
        strip = np.concatenate([data[t, :, :, 0], divider, recon[t, :, :, 0]], axis=1)
        img = Image.fromarray((strip * 255).astype(np.uint8), mode="L")
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        imgs.append(img)
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=50, loop=0)


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    dataset = data_lib.load()
    _, _, test_split = data_lib.splits(dataset["obs"].shape[0])
    pos = np.stack(
        [dataset["x"][test_split], dataset["y"][test_split]], axis=-1
    )

    runs = {}
    for name in ("base", "beta01", "warmup5k"):
        model, params = load_run(name)
        latents = encode_test(model, params, dataset, test_split)
        true, pred, r2 = probe_predictions(latents, pos)
        runs[name] = dict(model=model, params=params, latents=latents,
                          true=true, pred=pred, r2=r2)
        print(f"{name}: position R^2 {r2.mean():.3f}")

    # Post 1: same reconstruction quality, different linear readability
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2))
    scatter_panel(axes[0], runs["beta01"]["true"].ravel(),
                  runs["beta01"]["pred"].ravel(),
                  runs["beta01"]["r2"].mean(), "beta = 0.1")
    scatter_panel(axes[1], runs["warmup5k"]["true"].ravel(),
                  runs["warmup5k"]["pred"].ravel(),
                  runs["warmup5k"]["r2"].mean(), "beta = 1 (5k warmup)")
    fig.tight_layout()
    fig.savefig(MEDIA / "probe-scatter-beta01-vs-warmup.png", dpi=150,
                facecolor=SURFACE, bbox_inches="tight")

    # Post 2: the dead latent - probe scatter + the microscopic signal
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
    scatter_panel(axes[0], runs["base"]["true"].ravel(),
                  runs["base"]["pred"].ravel(),
                  runs["base"]["r2"].mean(), "collapsed run (KL = 0)")
    ax = axes[1]
    d = runs["base"]["latents"].shape[-1]
    dims = np.arange(1, d + 1)
    std_warm = np.sort(runs["warmup5k"]["latents"].reshape(-1, d).std(0))[::-1]
    std_base = np.sort(runs["base"]["latents"].reshape(-1, d).std(0))[::-1]
    ax.plot(dims, std_warm, "o-", ms=5, lw=1.5, color=BLUE,
            label="healthy (warmup5k)")
    ax.plot(dims, std_base, "o-", ms=5, lw=1.5, color=ORANGE,
            label="collapsed (base)")
    ax.set_yscale("log")
    ax.set_xlabel("latent dimension, sorted by spread")
    ax.set_ylabel("std of posterior mean over test set")
    ax.set_title("the \"signal\" the probe amplified", fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    style(ax)
    fig.tight_layout()
    fig.savefig(MEDIA / "dead-latent-probe.png", dpi=150,
                facecolor=SURFACE, bbox_inches="tight")

    # Rollout gif from the first eval-half test episode (warmup5k model)
    n_test = dataset["obs"][test_split].shape[0]
    episode = dataset["obs"][test_split][n_test // 2]
    rollout_gif(runs["warmup5k"]["model"], runs["warmup5k"]["params"],
                episode, MEDIA / "vae-recon-rollout.gif")
    print(f"wrote figures to {MEDIA}/")


if __name__ == "__main__":
    main()
