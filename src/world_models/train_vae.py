"""Train the Step 1 VAE and run the probe baseline.

    uv run train-vae --run-name base
    uv run train-vae --steps 2000 --run-name smoke

Artifacts land in runs/vae/<run-name>/: config.json, metrics.jsonl,
checkpoint.msgpack, reconstruction and prior-sample grids, loss curves,
and probe.json with the R^2 numbers.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from flax import serialization
from flax.training import train_state

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world_models import data as data_lib
from world_models import probe as probe_lib
from world_models.models.vae import VAE, elbo_terms
from world_models.tracking import Tracker


@dataclass(frozen=True)
class Config:
    seed: int = 0
    data_path: str = "data/ball.npz"
    latent_dim: int = 16
    beta: float = 1.0
    # 0 means constant beta. Linear warmup from 0 over this many steps; the
    # ball doesn't need it, but Doom-scale models will, so the knob exists.
    beta_warmup_steps: int = 0
    lr: float = 1e-3
    grad_clip: float = 1.0
    batch_size: int = 128
    steps: int = 20_000
    log_every: int = 100
    val_every: int = 500
    run_name: str = "base"
    run_root: str = "runs/vae"
    wandb_project: str | None = None


def beta_at(config: Config, step) -> jnp.ndarray:
    beta = jnp.asarray(config.beta, jnp.float32)
    if config.beta_warmup_steps > 0:
        frac = jnp.minimum(1.0, step / config.beta_warmup_steps)
        beta = beta * frac
    return beta


def make_steps(model: VAE, config: Config):
    def loss_fn(params, batch, key, beta):
        recon, mu, logvar = model.apply(params, batch, key)
        rec, kl = elbo_terms(recon, batch, mu, logvar)
        return rec + beta * kl, (rec, kl)

    @jax.jit
    def train_step(state, batch, key, beta):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (rec, kl)), grads = grad_fn(state.params, batch, key, beta)
        state = state.apply_gradients(grads=grads)
        return state, {"loss": loss, "recon": rec, "kl": kl}

    @jax.jit
    def val_recon(params, batch):
        # Deterministic eval: decode the posterior mean, no sampling.
        mu, _ = model.apply(params, batch, method=VAE.encode)
        recon = model.apply(params, mu, method=VAE.decode)
        return ((recon - batch) ** 2).sum(axis=(1, 2, 3)).mean()

    return train_step, val_recon


def recon_grid(model, params, frames, n: int = 8):
    """Originals on top, mean reconstructions below."""
    batch = data_lib.to_float(frames[:n])
    mu, _ = model.apply(params, batch, method=VAE.encode)
    recon = model.apply(params, mu, method=VAE.decode)
    fig, axes = plt.subplots(2, n, figsize=(1.2 * n, 2.6))
    for i in range(n):
        axes[0, i].imshow(batch[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(recon[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_title("data", loc="left", fontsize=9)
    axes[1, 0].set_title("recon", loc="left", fontsize=9)
    return fig


def prior_sample_grid(model, params, key, latent_dim: int, n: int = 16):
    z = jax.random.normal(key, (n, latent_dim))
    imgs = model.apply(params, z, method=VAE.decode)
    cols = 8
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.2 * cols, 1.3 * rows))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle("decoded prior samples z ~ N(0, I)", fontsize=9)
    return fig


def loss_curves(run_dir: Path):
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    train = [r for r in rows if "loss" in r]
    val = [r for r in rows if "val_recon" in r]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot([r["step"] for r in train], [r["recon"] for r in train], label="recon")
    if val:
        ax1.plot([r["step"] for r in val], [r["val_recon"] for r in val],
                 label="val recon")
    ax1.set_xlabel("step")
    ax1.legend()
    ax2.plot([r["step"] for r in train], [r["kl"] for r in train], label="kl")
    ax2.set_xlabel("step")
    ax2.legend()
    fig.tight_layout()
    return fig


def run_probe(model, params, dataset, test_split: slice) -> dict:
    """Encode test episodes, fit ridge on the first half, R^2 on the second.

    This protocol (same split, same lam) is shared by every later model, so
    the numbers stay comparable across Steps 1-3.
    """
    obs = dataset["obs"][test_split]
    n_ep, n_t = obs.shape[:2]
    frames = data_lib.to_float(obs.reshape((-1,) + obs.shape[2:]))
    mu, _ = model.apply(params, frames, method=VAE.encode)
    latents = mu.reshape(n_ep, n_t, -1)

    targets = {
        "position": np.stack([dataset["x"][test_split], dataset["y"][test_split]], -1),
        "velocity": np.stack([dataset["vx"][test_split], dataset["vy"][test_split]], -1),
    }
    half = n_ep // 2
    results = {}
    for name, t in targets.items():
        t = jnp.asarray(t)
        r2 = probe_lib.probe(
            latents[:half].reshape(-1, latents.shape[-1]),
            t[:half].reshape(-1, t.shape[-1]),
            latents[half:].reshape(-1, latents.shape[-1]),
            t[half:].reshape(-1, t.shape[-1]),
        )
        results[f"r2_{name}"] = [float(v) for v in r2]
        results[f"r2_{name}_mean"] = float(r2.mean())
    return results


def train(config: Config) -> dict:
    run_dir = Path(config.run_root) / config.run_name
    if run_dir.exists():
        raise SystemExit(
            f"{run_dir} already exists; pick another --run-name or delete it"
        )
    tracker = Tracker(
        run_dir,
        {
            **dataclasses.asdict(config),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
        },
        wandb_project=config.wandb_project,
        run_name=config.run_name,
    )

    dataset = data_lib.load(config.data_path)
    train_split, val_split, test_split = data_lib.splits(dataset["obs"].shape[0])
    train_frames = data_lib.frames_of(dataset, train_split)
    val_frames = data_lib.frames_of(dataset, val_split)
    val_batch = data_lib.to_float(val_frames[:512])

    model = VAE(latent_dim=config.latent_dim)
    root_key = jax.random.PRNGKey(config.seed)
    init_key, sample_key, artifact_key = jax.random.split(root_key, 3)
    dummy = jnp.zeros((1, *train_frames.shape[1:]), jnp.float32)
    params = model.init(init_key, dummy, sample_key)

    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.lr),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )
    train_step, val_recon = make_steps(model, config)

    rng = np.random.default_rng(config.seed)
    for step in range(1, config.steps + 1):
        idx = rng.integers(0, train_frames.shape[0], config.batch_size)
        batch = data_lib.to_float(train_frames[idx])
        step_key = jax.random.fold_in(root_key, step)
        state, metrics = train_step(state, batch, step_key, beta_at(config, step))
        if step % config.log_every == 0:
            tracker.log(step, **metrics)
        if step % config.val_every == 0:
            v = val_recon(state.params, val_batch)
            tracker.log(step, val_recon=v)
            print(f"step {step:6d}  loss {metrics['loss']:8.3f}  "
                  f"recon {metrics['recon']:8.3f}  kl {metrics['kl']:6.3f}  "
                  f"val {v:8.3f}")

    (run_dir / "checkpoint.msgpack").write_bytes(
        serialization.to_bytes(state.params)
    )
    tracker.log_figure("recon_grid", recon_grid(model, state.params, val_frames))
    tracker.log_figure(
        "prior_samples",
        prior_sample_grid(model, state.params, artifact_key, config.latent_dim),
    )
    tracker.log_figure("loss_curves", loss_curves(run_dir))

    probe_results = run_probe(model, state.params, dataset, test_split)
    tracker.log_json("probe", probe_results)
    tracker.finish()
    print("probe results:")
    for k, v in probe_results.items():
        print(f"  {k}: {v}")
    return probe_results


def main():
    parser = argparse.ArgumentParser(description="Train the Step 1 VAE")
    defaults = Config()
    parser.add_argument("--run-name", type=str, default=defaults.run_name)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--latent-dim", type=int, default=defaults.latent_dim)
    parser.add_argument("--beta", type=float, default=defaults.beta)
    parser.add_argument("--beta-warmup-steps", type=int,
                        default=defaults.beta_warmup_steps)
    parser.add_argument("--data", type=str, default=defaults.data_path)
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="mirror metrics to this wandb project")
    args = parser.parse_args()
    config = Config(
        seed=args.seed,
        data_path=args.data,
        latent_dim=args.latent_dim,
        beta=args.beta,
        beta_warmup_steps=args.beta_warmup_steps,
        steps=args.steps,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
    )
    train(config)


if __name__ == "__main__":
    main()
