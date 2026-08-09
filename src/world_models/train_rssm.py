"""Train the Step 3 RSSM jointly on pixel sequences.

    uv run train-rssm --run-name base

Protocol pinned to Step 2: same dataset and splits, 32-transition
subsequences, warm-up 4 and horizon 30 for the open-loop drift, same
ridge probe. Artifacts land in runs/rssm/<run-name>/.
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
from world_models.models.rssm import RSSM, kl_balanced, kl_gauss
from world_models.plotstyle import AQUA, BASELINE, BLUE, MUTED, style
from world_models.tracking import Tracker, metrics_figure


@dataclass(frozen=True)
class Config:
    seed: int = 0
    data_path: str = "data/ball.npz"
    latent_dim: int = 16
    hidden: int = 128
    min_sigma: float = 0.1
    alpha: float = 0.8               # KL balancing weight toward the prior
    beta: float = 1.0
    beta_warmup_steps: int = 0       # 0 = constant beta
    transitions: int = 32
    warmup: int = 4                  # eval-only: rollout context length
    horizon: int = 30
    lr: float = 1e-3
    grad_clip: float = 1.0
    batch_size: int = 32
    steps: int = 20_000
    log_every: int = 100
    val_every: int = 500
    run_name: str = "base"
    run_root: str = "runs/rssm"
    wandb_project: str | None = None


def beta_at(config: Config, step) -> jnp.ndarray:
    beta = jnp.asarray(config.beta, jnp.float32)
    if config.beta_warmup_steps > 0:
        beta = beta * jnp.minimum(1.0, step / config.beta_warmup_steps)
    return beta


def sample_pixel_batch(rng, obs, actions, episodes, transitions, batch_size):
    """Random pixel subsequences -> time-major (T+1, B, H, W, 1) and (T+1, B, A)."""
    ep = rng.integers(episodes.start, episodes.stop, batch_size)
    t0 = rng.integers(0, obs.shape[1] - (transitions + 1), batch_size)
    t_idx = t0[:, None] + np.arange(transitions + 1)[None, :]
    o = obs[ep[:, None], t_idx]
    a = actions[ep[:, None], t_idx]
    return (jnp.asarray(o.transpose(1, 0, 2, 3, 4), jnp.float32) / 255.0,
            jnp.asarray(a.transpose(1, 0, 2)))


def make_losses(model: RSSM, config: Config):
    def sequence_loss(params, obs_seq, act_seq, key, beta):
        """obs_seq: (T+1, B, H, W, 1) time-major. Recon on every frame,
        KL on frames 1..T (frame 0 has no meaningful prior)."""
        t1, b = obs_seq.shape[0], obs_seq.shape[1]
        e = model.apply(params, obs_seq.reshape((-1,) + obs_seq.shape[2:]),
                        method=RSSM.encode).reshape(t1, b, -1)
        noise = jax.random.normal(key, (t1, b, config.latent_dim))

        h0 = model.initial_state(b)
        mu_q0, sig_q0 = model.apply(params, h0, e[0], method=RSSM.post_dist)
        z0 = mu_q0 + sig_q0 * noise[0]
        o_hat0 = model.apply(params, h0, z0, method=RSSM.decode)
        recon0 = ((o_hat0 - obs_seq[0]) ** 2).sum(axis=(1, 2, 3)).mean()

        def step(carry, xs):
            h, z = carry
            e_t, o_t, a_t, n_t = xs
            h = model.apply(params, h, z, a_t, method=RSSM.core_step)
            mu_p, sig_p = model.apply(params, h, method=RSSM.prior_dist)
            mu_q, sig_q = model.apply(params, h, e_t, method=RSSM.post_dist)
            z = mu_q + sig_q * n_t
            o_hat = model.apply(params, h, z, method=RSSM.decode)
            recon = ((o_hat - o_t) ** 2).sum(axis=(1, 2, 3)).mean()
            klb = kl_balanced(mu_q, sig_q, mu_p, sig_p, config.alpha).mean()
            klr = kl_gauss(mu_q, sig_q, mu_p, sig_p).mean()
            return (h, z), (recon, klb, klr)

        _, (recons, klbs, klrs) = jax.lax.scan(
            step, (h0, z0), (e[1:], obs_seq[1:], act_seq[1:], noise[1:])
        )
        recon = (recon0 + recons.sum()) / t1
        klb = klbs.mean()
        klr = klrs.mean()
        return recon + beta * klb, (recon, klr)

    return sequence_loss


def filter_episodes(model, params, obs, actions):
    """Posterior-mean filtering pass over full episodes.

    obs: (T+1, B, H, W, 1) time-major float. Returns h_seq and z_seq of
    shape (T+1, B, ·); h_seq[t] has seen frames 0..t-1 (prediction-time
    state), z_seq[t] has seen frame t.
    """
    t1, b = obs.shape[0], obs.shape[1]
    e = model.apply(params, obs.reshape((-1,) + obs.shape[2:]),
                    method=RSSM.encode).reshape(t1, b, -1)
    h0 = model.initial_state(b)
    z0, _ = model.apply(params, h0, e[0], method=RSSM.post_dist)

    def step(carry, xs):
        h, z = carry
        e_t, a_t = xs
        h = model.apply(params, h, z, a_t, method=RSSM.core_step)
        z, _ = model.apply(params, h, e_t, method=RSSM.post_dist)
        return (h, z), (h, z)

    _, (h_seq, z_seq) = jax.lax.scan(step, (h0, z0), (e[1:], actions[1:]))
    h_seq = jnp.concatenate([h0[None], h_seq])
    z_seq = jnp.concatenate([z0[None], z_seq])
    return h_seq, z_seq


def rollout_prior(model, params, h, z, a_future, noise=None):
    """Roll the prior open loop from a filtered state.

    a_future: (K, B, A), the actions of transitions warmup+1 .. warmup+K.
    noise None -> prior means; else (K, B, D) -> prior samples.
    Returns decoded frames (K, B, H, W, 1).
    """
    xs = (a_future, noise) if noise is not None else (
        a_future, jnp.zeros(a_future.shape[:2] + (model.latent_dim,)))

    def step(carry, x):
        h, z = carry
        a, n = x
        h = model.apply(params, h, z, a, method=RSSM.core_step)
        mu_p, sig_p = model.apply(params, h, method=RSSM.prior_dist)
        z = mu_p + (sig_p * n if noise is not None else 0.0)
        return (h, z), (h, z)

    _, (hs, zs) = jax.lax.scan(step, (h, z), xs)
    k, b = hs.shape[0], hs.shape[1]
    frames = model.apply(params, hs.reshape(-1, hs.shape[-1]),
                         zs.reshape(-1, zs.shape[-1]), method=RSSM.decode)
    return frames.reshape(k, b, *frames.shape[1:])


def drift_eval(model, params, dataset, test_split, config):
    obs = jnp.asarray(
        dataset["obs"][test_split].transpose(1, 0, 2, 3, 4), jnp.float32
    ) / 255.0                                             # (T+1, B, H, W, 1)
    act = jnp.asarray(dataset["action"][test_split].transpose(1, 0, 2))
    w, k = config.warmup, config.horizon

    h_seq, z_seq = filter_episodes(model, params, obs[:w], act[:w])
    h_w, z_w = h_seq[w - 1], z_seq[w - 1]
    a_future = act[w:w + k]

    curves = {}
    frames_mean = rollout_prior(model, params, h_w, z_w, a_future)
    true_frames = obs[w:w + k]
    curves["mean"] = np.asarray(rollout_lib.pixel_mse_per_horizon(
        frames_mean, true_frames))
    noise = jax.random.normal(jax.random.PRNGKey(config.seed + 7),
                              (k, obs.shape[1], config.latent_dim))
    frames_sample = rollout_prior(model, params, h_w, z_w, a_future, noise)
    curves["sample"] = np.asarray(rollout_lib.pixel_mse_per_horizon(
        frames_sample, true_frames))
    return curves, np.asarray(frames_mean), np.asarray(true_frames)


def probe_states(model, params, dataset, test_split, warmup):
    """Probes on h alone (capacity-matched vs the GRU) and on [h, z]."""
    obs = jnp.asarray(
        dataset["obs"][test_split].transpose(1, 0, 2, 3, 4), jnp.float32
    ) / 255.0
    act = jnp.asarray(dataset["action"][test_split].transpose(1, 0, 2))
    h_seq, z_seq = filter_episodes(model, params, obs, act)
    h = np.asarray(h_seq).transpose(1, 0, 2)              # (B, T+1, 128)
    z = np.asarray(z_seq).transpose(1, 0, 2)
    hz = np.concatenate([h, z], axis=-1)

    truth = {
        "position": np.stack([dataset["x"][test_split], dataset["y"][test_split]], -1),
        "velocity": np.stack([dataset["vx"][test_split], dataset["vy"][test_split]], -1),
    }
    n_ep = h.shape[0]
    half = n_ep // 2
    results = {}
    # h[t] has seen frames 0..t-1 -> align with truth[t-1];
    # hz[t] includes z_t, which saw frame t -> align with truth[t].
    for feat_name, feats, shift in (("h", h, 1), ("hz", hz, 0)):
        f = feats[:, warmup + shift:]
        for tgt_name, t in truth.items():
            tt = t[:, warmup:f.shape[1] + warmup]
            r2 = probe_lib.probe(
                jnp.asarray(f[:half].reshape(-1, f.shape[-1])),
                jnp.asarray(tt[:half].reshape(-1, tt.shape[-1])),
                jnp.asarray(f[half:].reshape(-1, f.shape[-1])),
                jnp.asarray(tt[half:].reshape(-1, tt.shape[-1])),
            )
            results[f"r2_{tgt_name}_{feat_name}"] = [float(v) for v in r2]
            results[f"r2_{tgt_name}_{feat_name}_mean"] = float(r2.mean())
    return results


def drift_figure(curves, warmup):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ks = np.arange(1, len(curves["mean"]) + 1)
    ax.plot(ks, curves["mean"], color=BLUE, lw=1.5, label="prior mean")
    ax.plot(ks, curves["sample"], color=AQUA, lw=1.5, label="prior sample")
    for level, label in ((0.104 ** 2, "mean image"),
                         (2 * 0.104 ** 2, "ball in the wrong place")):
        ax.axhline(level, ls="--", lw=1, color=BASELINE)
        ax.annotate(label, (1, level), fontsize=7, color=MUTED,
                    textcoords="offset points", xytext=(2, 3))
    ax.set_yscale("log")
    ax.set_xlabel(f"open-loop horizon (steps past {warmup} real frames)")
    ax.set_ylabel("pixel MSE (log)")
    ax.set_title("RSSM open-loop drift", fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor="#52514e")
    style(ax)
    fig.tight_layout()
    return fig


def filmstrip_figure(pred_frames, true_frames, episode: int = 0):
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
    return fig


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
    obs, actions = dataset["obs"], dataset["action"]

    model = RSSM(
        latent_dim=config.latent_dim,
        action_dim=actions.shape[-1],
        hidden=config.hidden,
        min_sigma=config.min_sigma,
    )
    root_key = jax.random.PRNGKey(config.seed)
    init_key, val_key = jax.random.split(root_key)
    params = model.init(
        init_key, jnp.zeros((1, 32, 32, 1)), model.initial_state(1),
        jnp.zeros((1, config.latent_dim)), jnp.zeros((1, actions.shape[-1])),
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.lr),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )
    sequence_loss = make_losses(model, config)

    @jax.jit
    def train_step(state, obs_seq, act_seq, key, beta):
        grad_fn = jax.value_and_grad(sequence_loss, has_aux=True)
        (loss, (recon, kl)), grads = grad_fn(state.params, obs_seq, act_seq,
                                             key, beta)
        return state.apply_gradients(grads=grads), loss, recon, kl

    @jax.jit
    def eval_loss(params, obs_seq, act_seq, beta):
        loss, (recon, kl) = sequence_loss(params, obs_seq, act_seq,
                                          val_key, beta)
        return recon, kl

    rng = np.random.default_rng(config.seed)
    val_rng = np.random.default_rng(config.seed + 1)
    val_obs, val_act = sample_pixel_batch(
        val_rng, obs, actions, val_split, config.transitions, 64
    )

    for step in range(1, config.steps + 1):
        obs_seq, act_seq = sample_pixel_batch(
            rng, obs, actions, train_split, config.transitions,
            config.batch_size,
        )
        step_key = jax.random.fold_in(root_key, step)
        beta = beta_at(config, step)
        state, loss, recon, kl = train_step(state, obs_seq, act_seq,
                                            step_key, beta)
        if step % config.log_every == 0:
            tracker.log(step, loss=loss, recon=recon, kl=kl)
        if step % config.val_every == 0:
            v_recon, v_kl = eval_loss(state.params, val_obs, val_act, beta)
            tracker.log(step, val_recon=v_recon, val_kl=v_kl)
            print(f"step {step:6d}  recon {recon:8.3f}  kl {kl:7.3f}  "
                  f"val recon {v_recon:8.3f}  val kl {v_kl:7.3f}")

    (run_dir / "checkpoint.msgpack").write_bytes(
        serialization.to_bytes(state.params)
    )

    curves, pred_frames, true_frames = drift_eval(
        model, state.params, dataset, test_split, config
    )
    drift = {
        "horizons": list(range(1, config.horizon + 1)),
        "pixel_mse": [float(v) for v in curves["mean"]],
        "pixel_mse_sampled": [float(v) for v in curves["sample"]],
        "at": {str(k): float(curves["mean"][k - 1]) for k in (1, 5, 15, 30)},
    }
    tracker.log_json("drift", drift)
    tracker.log_figure("loss_curves", metrics_figure(run_dir))
    tracker.log_figure("drift_curve", drift_figure(curves, config.warmup))
    tracker.log_figure("filmstrip", filmstrip_figure(pred_frames, true_frames))
    rollout_lib.side_by_side_gif(
        true_frames[:, 0, :, :, 0], pred_frames[:, 0, :, :, 0],
        run_dir / "rollout.gif",
    )

    probe_results = probe_states(model, state.params, dataset, test_split,
                                 config.warmup)
    tracker.log_json("probe", probe_results)
    tracker.finish()

    print("drift at 1/5/15/30:", drift["at"])
    print("probe results:")
    for k, v in probe_results.items():
        if k.endswith("_mean"):
            print(f"  {k}: {v}")
    return {"drift": drift, "probe": probe_results}


def main():
    parser = argparse.ArgumentParser(description="Train the Step 3 RSSM")
    defaults = Config()
    parser.add_argument("--run-name", type=str, default=defaults.run_name)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--beta", type=float, default=defaults.beta)
    parser.add_argument("--beta-warmup-steps", type=int,
                        default=defaults.beta_warmup_steps)
    parser.add_argument("--alpha", type=float, default=defaults.alpha)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--data", type=str, default=defaults.data_path)
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()
    config = Config(
        seed=args.seed,
        data_path=args.data,
        lr=args.lr,
        beta=args.beta,
        beta_warmup_steps=args.beta_warmup_steps,
        alpha=args.alpha,
        batch_size=args.batch_size,
        steps=args.steps,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
    )
    train(config)


if __name__ == "__main__":
    main()
