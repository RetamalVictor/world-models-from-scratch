"""Train the Step 2 GRU dynamics model on frozen VAE latents.

The 2x2 ablation grid from docs/design/gru-dynamics.md:

    uv run train-gru --predict direct   --latent-source mean
    uv run train-gru --predict residual --latent-source mean
    uv run train-gru --predict direct   --latent-source sample
    uv run train-gru --predict residual --latent-source sample

Artifacts land in runs/gru/<run-name>/: checkpoint, standardizer,
metrics.jsonl, drift.json + curve, probe.json, filmstrip and rollout gif.
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
from world_models import rollout as rollout_lib
from world_models.models.gru_dynamics import GRUDynamics, sequence_nll
from world_models.models.vae import VAE, load_vae
from world_models.plotstyle import BLUE, style
from world_models.tracking import Tracker


@dataclass(frozen=True)
class Config:
    seed: int = 0
    data_path: str = "data/ball.npz"
    vae_run: str = "runs/vae/warmup5k"
    hidden: int = 128
    predict: str = "direct"          # direct | residual
    latent_source: str = "mean"      # mean | sample
    transitions: int = 32            # subsequence length, in transitions
    warmup: int = 4                  # loss-masked transitions & rollout context
    horizon: int = 30                # open-loop rollout length
    lr: float = 1e-3
    grad_clip: float = 1.0
    batch_size: int = 64
    steps: int = 20_000
    log_every: int = 100
    val_every: int = 500
    run_name: str | None = None      # default: "<predict>-<latent_source>"
    run_root: str = "runs/gru"
    wandb_project: str | None = None


def encode_dataset(vae_model, vae_params, dataset, source: str, seed: int):
    """Encode every frame with the frozen VAE -> (E, T, D) float32.

    source "mean" uses the posterior mean; "sample" draws one fixed,
    seeded sample per frame, so the encoded dataset is still
    deterministic given the config.
    """
    obs = dataset["obs"]
    n_ep, n_t = obs.shape[:2]
    frames = obs.reshape((-1,) + obs.shape[2:])
    chunks = []
    chunk = 8192
    for i in range(0, frames.shape[0], chunk):
        batch = data_lib.to_float(frames[i:i + chunk])
        mu, logvar = vae_model.apply(vae_params, batch, method=VAE.encode)
        if source == "sample":
            key = jax.random.fold_in(jax.random.PRNGKey(seed), i)
            mu = mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
        chunks.append(np.asarray(mu))
    latents = np.concatenate(chunks, axis=0)
    return latents.reshape(n_ep, n_t, -1).astype(np.float32)


def fit_standardizer(latents, train_split):
    flat = latents[train_split].reshape(-1, latents.shape[-1])
    return {"mean": flat.mean(0), "std": flat.std(0) + 1e-6}


def standardize(latents, stats):
    return (latents - stats["mean"]) / stats["std"]


def unstandardize(latents, stats):
    return latents * stats["std"] + stats["mean"]


def sample_batch(rng, latents, actions, episodes, transitions, batch_size):
    """Random subsequences -> time-major (T+1, B, D) and (T+1, B, A)."""
    ep = rng.integers(episodes.start, episodes.stop, batch_size)
    t0 = rng.integers(0, latents.shape[1] - (transitions + 1), batch_size)
    t_idx = t0[:, None] + np.arange(transitions + 1)[None, :]
    z = latents[ep[:, None], t_idx]      # (B, T+1, D)
    a = actions[ep[:, None], t_idx]      # (B, T+1, A)
    return (jnp.asarray(z.transpose(1, 0, 2)),
            jnp.asarray(a.transpose(1, 0, 2)))


def drift_eval(model, params, vae_model, vae_params, dataset, latents_norm,
               stats, test_split, warmup, horizon):
    """Open-loop drift over all test episodes -> per-horizon pixel MSE."""
    z = jnp.asarray(latents_norm[test_split].transpose(1, 0, 2))  # (T, B, D)
    a = jnp.asarray(dataset["action"][test_split].transpose(1, 0, 2))
    n_ep = z.shape[1]

    def step_fn(h, z_in, act):
        h, mu, _ = model.apply(params, h, z_in, act)
        return h, mu

    preds_norm = rollout_lib.open_loop_predict(
        step_fn, model.initial_state(n_ep),
        z[:warmup], a[1:warmup + 1], a[warmup + 1:warmup + horizon],
    )                                                     # (K, B, D)
    preds = unstandardize(np.asarray(preds_norm), stats)
    flat = jnp.asarray(preds.reshape(-1, preds.shape[-1]))
    decoded = vae_model.apply(vae_params, flat, method=VAE.decode)
    pred_frames = np.asarray(decoded).reshape(
        horizon, n_ep, *decoded.shape[1:])

    true = data_lib.to_float(
        dataset["obs"][test_split][:, warmup:warmup + horizon]
    )                                                     # (B, K, H, W, 1)
    true_frames = np.asarray(true).transpose(1, 0, 2, 3, 4)
    mse = np.asarray(rollout_lib.pixel_mse_per_horizon(
        jnp.asarray(pred_frames), jnp.asarray(true_frames)))
    return mse, pred_frames, true_frames


def probe_hidden_state(model, params, latents_norm, dataset, test_split,
                       warmup):
    """Probe h_t against ground truth at frame t, chaperone protocol."""
    z = jnp.asarray(latents_norm[test_split].transpose(1, 0, 2))
    a = jnp.asarray(dataset["action"][test_split].transpose(1, 0, 2))
    mask = jnp.ones(z.shape[0] - 1)
    _, h_seq = sequence_nll(model, params, z, a, mask)    # (T, B, H)
    # h_seq[t] has consumed frames 0..t -> align with truth at frame t.
    h = np.asarray(h_seq).transpose(1, 0, 2)              # (B, T, H)
    h = h[:, warmup:]                                     # drop short-history states
    n_ep, n_t, dim = h.shape
    half = n_ep // 2

    results = {}
    targets = {
        "position": np.stack([dataset["x"][test_split], dataset["y"][test_split]], -1),
        "velocity": np.stack([dataset["vx"][test_split], dataset["vy"][test_split]], -1),
    }
    for name, t in targets.items():
        t = t[:, warmup:h.shape[1] + warmup]              # frames warmup..T-1
        r2 = probe_lib.probe(
            jnp.asarray(h[:half].reshape(-1, dim)),
            jnp.asarray(t[:half].reshape(-1, t.shape[-1])),
            jnp.asarray(h[half:].reshape(-1, dim)),
            jnp.asarray(t[half:].reshape(-1, t.shape[-1])),
        )
        results[f"r2_{name}"] = [float(v) for v in r2]
        results[f"r2_{name}_mean"] = float(r2.mean())
    return results


def drift_figure(mse, warmup):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(np.arange(1, len(mse) + 1), mse, color=BLUE, lw=1.5)
    ax.set_xlabel(f"open-loop horizon (steps past {warmup} real transitions)")
    ax.set_ylabel("pixel MSE vs real continuation")
    ax.set_title("open-loop drift", fontsize=10)
    style(ax)
    fig.tight_layout()
    return fig


def filmstrip_figure(pred_frames, true_frames, episode: int):
    ks = [1, 2, 4, 6, 10, 15, 22, 30]
    fig, axes = plt.subplots(2, len(ks), figsize=(1.2 * len(ks), 2.8))
    for i, k in enumerate(ks):
        axes[0, i].imshow(true_frames[k - 1, episode, :, :, 0],
                          cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(np.clip(pred_frames[k - 1, episode, :, :, 0], 0, 1),
                          cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"k={k}", fontsize=8)
        for row in (0, 1):
            axes[row, i].axis("off")
    axes[0, 0].set_ylabel("real", fontsize=8)
    axes[1, 0].set_ylabel("open loop", fontsize=8)
    return fig


def train(config: Config) -> dict:
    run_name = config.run_name or f"{config.predict}-{config.latent_source}"
    run_dir = Path(config.run_root) / run_name
    if run_dir.exists():
        raise SystemExit(
            f"{run_dir} already exists; pick another --run-name or delete it"
        )
    tracker = Tracker(
        run_dir,
        {
            **dataclasses.asdict(config),
            "run_name": run_name,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
        },
        wandb_project=config.wandb_project,
        run_name=run_name,
    )

    dataset = data_lib.load(config.data_path)
    train_split, val_split, test_split = data_lib.splits(dataset["obs"].shape[0])

    vae_model, vae_params = load_vae(config.vae_run)
    latents = encode_dataset(vae_model, vae_params, dataset,
                             config.latent_source, config.seed)
    stats = fit_standardizer(latents, train_split)
    latents_norm = standardize(latents, stats)
    actions = dataset["action"]
    latent_dim = latents.shape[-1]

    model = GRUDynamics(
        latent_dim=latent_dim,
        action_dim=actions.shape[-1],
        hidden=config.hidden,
        residual=(config.predict == "residual"),
    )
    root_key = jax.random.PRNGKey(config.seed)
    params = model.init(
        root_key, model.initial_state(1),
        jnp.zeros((1, latent_dim)), jnp.zeros((1, actions.shape[-1])),
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.lr),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

    loss_mask = jnp.concatenate([
        jnp.zeros(config.warmup),
        jnp.ones(config.transitions - config.warmup),
    ])

    @jax.jit
    def train_step(state, z_seq, a_seq):
        def loss_fn(p):
            nll, _ = sequence_nll(model, p, z_seq, a_seq, loss_mask)
            return nll
        nll, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), nll

    @jax.jit
    def eval_nll(params, z_seq, a_seq):
        nll, _ = sequence_nll(model, params, z_seq, a_seq, loss_mask)
        return nll

    rng = np.random.default_rng(config.seed)
    val_rng = np.random.default_rng(config.seed + 1)
    val_z, val_a = sample_batch(val_rng, latents_norm, actions, val_split,
                                config.transitions, 256)

    for step in range(1, config.steps + 1):
        z_seq, a_seq = sample_batch(rng, latents_norm, actions, train_split,
                                    config.transitions, config.batch_size)
        state, nll = train_step(state, z_seq, a_seq)
        if step % config.log_every == 0:
            tracker.log(step, nll=nll)
        if step % config.val_every == 0:
            v = eval_nll(state.params, val_z, val_a)
            tracker.log(step, val_nll=v)
            print(f"step {step:6d}  nll {nll:8.3f}  val {v:8.3f}")

    (run_dir / "checkpoint.msgpack").write_bytes(
        serialization.to_bytes(state.params)
    )
    (run_dir / "standardizer.json").write_text(json.dumps(
        {k: [float(x) for x in v] for k, v in stats.items()}, indent=2
    ))

    mse, pred_frames, true_frames = drift_eval(
        model, state.params, vae_model, vae_params, dataset, latents_norm,
        stats, test_split, config.warmup, config.horizon,
    )
    drift = {
        "horizons": list(range(1, config.horizon + 1)),
        "pixel_mse": [float(v) for v in mse],
        "at": {str(k): float(mse[k - 1]) for k in (1, 5, 15, 30)},
    }
    tracker.log_json("drift", drift)
    tracker.log_figure("drift_curve", drift_figure(mse, config.warmup))
    tracker.log_figure("filmstrip", filmstrip_figure(pred_frames, true_frames,
                                                     episode=0))
    rollout_lib.side_by_side_gif(
        true_frames[:, 0, :, :, 0], pred_frames[:, 0, :, :, 0],
        run_dir / "rollout.gif",
    )

    probe_results = probe_hidden_state(
        model, state.params, latents_norm, dataset, test_split, config.warmup
    )
    tracker.log_json("probe", probe_results)
    tracker.finish()

    print("drift at 1/5/15/30:", drift["at"])
    print("probe results:")
    for k, v in probe_results.items():
        print(f"  {k}: {v}")
    return {"drift": drift, "probe": probe_results}


def main():
    parser = argparse.ArgumentParser(description="Train the Step 2 GRU")
    defaults = Config()
    parser.add_argument("--predict", choices=["direct", "residual"],
                        default=defaults.predict)
    parser.add_argument("--latent-source", choices=["mean", "sample"],
                        default=defaults.latent_source)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--vae-run", type=str, default=defaults.vae_run)
    parser.add_argument("--data", type=str, default=defaults.data_path)
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()
    config = Config(
        seed=args.seed,
        data_path=args.data,
        vae_run=args.vae_run,
        predict=args.predict,
        latent_source=args.latent_source,
        steps=args.steps,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
    )
    train(config)


if __name__ == "__main__":
    main()
