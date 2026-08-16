"""Post-hoc continue-head diagnostics for a finished train-doom run.

    uv run python scripts/doom_diag.py --run runs/doom/pretrain-v3

The question this answers (journal 0010, lessons 2 to 4): does the
death signal live somewhere imagination can reach, or only in posterior
states that dreams never visit? Four instruments, all over the buffer's
death-terminated episodes (timeouts carry no death label and are
excluded):

- one-step prior test: filter the posterior over the real frames to the
  terminal frame T, then read P(continue) on the posterior latent and
  on the prior latent of the very same h. Agreement means the cue is
  encoded somewhere the prior transitions into, which is the definition
  of an imagination-usable signal.
- anticipation: the posterior one frame before death.
- dream death traces: warm the posterior to T minus dream_start, roll
  the sampled prior forward on the recorded real actions, and read the
  head at every imagined step.
- calm-frame false positives: the posterior on frames at least 20 steps
  from any episode end, which should score alive.

One line per statistic on stdout with the v3 WSL reference in
parentheses (orientation, not a pass condition), full JSON to --out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from world_models.checkpoint import Checkpointer
from world_models.models.rssm import RSSM, load_rssm
from world_models.replay import ReplayBuffer
from world_models.train_rssm import filter_episodes

CHUNK = 32        # episodes per filtering pass, bounds the padded batch
CALM_MARGIN = 20  # a calm frame sits at least this far from episode end


def load_world_model(run_dir: Path):
    """Returns (model, params, source path relative to the run dir).

    checkpoint.msgpack is the finished-run artifact and load_rssm reads
    it directly. A run that stopped before writing it still has the
    Checkpointer rotation, where the wm params sit inside the train
    state tree, so I restore the raw state dict and pour the params
    slot into a fresh template.
    """
    cfg = json.loads((run_dir / "config.json").read_text())
    if not cfg.get("predict_continue", False):
        raise SystemExit(
            f"{run_dir} trained no continue head; nothing to diagnose")
    if (run_dir / "checkpoint.msgpack").exists():
        model, params = load_rssm(run_dir)
        return model, params, "checkpoint.msgpack"
    model = RSSM(
        latent_dim=cfg["latent_dim"], action_dim=cfg["action_dim"],
        hidden=cfg["hidden"], min_sigma=cfg["min_sigma"],
        obs_channels=cfg.get("obs_channels", 1), obs_size=cfg["obs_size"],
        predict_continue=True,
    )
    s, c = model.obs_size, model.obs_channels
    template = model.init(
        jax.random.PRNGKey(0), jnp.zeros((1, s, s, c)),
        model.initial_state(1), jnp.zeros((1, model.latent_dim)),
        jnp.zeros((1, model.action_dim)),
    )
    step = Checkpointer(run_dir).latest_step()
    if step is None:
        raise SystemExit(
            f"{run_dir} has neither checkpoint.msgpack nor checkpoints/")
    name = f"checkpoints/step_{step:08d}.msgpack"
    raw = serialization.msgpack_restore((run_dir / name).read_bytes())
    params = serialization.from_state_dict(template, raw["wm"]["params"])
    return model, params, name


def continue_p(model, params, h, z):
    logit = model.apply(params, h, z, method=RSSM.continue_logit)
    return jax.nn.sigmoid(logit)


def batch_episodes(chunk):
    """Pad a chunk of episodes into time-major (T+1, B, ...) batches.

    Zero padding past each episode's end is safe: filtering is causal,
    so no state at t <= T ever sees a padded frame, and nothing past T
    is ever read out.
    """
    tmax1 = max(ep["obs"].shape[0] for ep in chunk)
    b = len(chunk)
    obs = np.zeros((tmax1, b) + chunk[0]["obs"].shape[1:], np.float32)
    act = np.zeros((tmax1, b, chunk[0]["action"].shape[-1]), np.float32)
    for i, ep in enumerate(chunk):
        n = ep["obs"].shape[0]
        obs[:n, i] = ep["obs"].astype(np.float32) / 255.0
        act[:n, i] = ep["action"]
    return jnp.asarray(obs), jnp.asarray(act)


def dream_traces(model, params, h0, z0, actions, noise):
    """P(continue) along sampled-prior rollouts, (K, N) for (K, N, A)
    actions. Same step order as train_doom's imagine, with the recorded
    real actions in place of the actor."""
    def step(carry, xs):
        h, z = carry
        a, n = xs
        h = model.apply(params, h, z, a, method=RSSM.core_step)
        mu_p, sig_p = model.apply(params, h, method=RSSM.prior_dist)
        z = mu_p + sig_p * n
        return (h, z), continue_p(model, params, h, z)

    _, traces = jax.lax.scan(step, (h0, z0), (actions, noise))
    return traces


def analyze(model, params, episodes, dream_start, key):
    """All per-episode measurements. Returns a dict of 1d arrays plus
    the pooled calm-frame values."""
    post_T, post_Tm1, calm = [], [], []
    h_T, h_warm, z_warm, a_dream = [], [], [], []

    for start in range(0, len(episodes), CHUNK):
        chunk = episodes[start:start + CHUNK]
        obs, act = batch_episodes(chunk)
        h_seq, z_seq = filter_episodes(model, params, obs, act)
        flat = continue_p(model, params,
                          h_seq.reshape(-1, h_seq.shape[-1]),
                          z_seq.reshape(-1, z_seq.shape[-1]))
        c_grid = np.asarray(flat.reshape(h_seq.shape[0], h_seq.shape[1]))
        h_np, z_np = np.asarray(h_seq), np.asarray(z_seq)
        act_np = np.asarray(act)
        for i, ep in enumerate(chunk):
            t_term = ep["obs"].shape[0] - 1
            post_T.append(c_grid[t_term, i])
            post_Tm1.append(c_grid[t_term - 1, i])
            # frame 0 has no filtered history, so calm frames start at 1
            calm.extend(c_grid[1:t_term - CALM_MARGIN + 1, i])
            # filter_episodes already took the core step into T with the
            # real action, so h_seq[T] is the h the prior test needs:
            # posterior and prior latents get read on the identical h.
            h_T.append(h_np[t_term, i])
            t0 = t_term - dream_start
            h_warm.append(h_np[t0, i])
            z_warm.append(z_np[t0, i])
            a_dream.append(act_np[t0 + 1:t0 + 1 + dream_start, i])

    n = len(episodes)
    k_prior, k_dream = jax.random.split(key)

    h_T = jnp.asarray(np.stack(h_T))
    mu_p, sig_p = model.apply(params, h_T, method=RSSM.prior_dist)
    eps = jax.random.normal(k_prior, mu_p.shape)
    prior_mean = np.asarray(continue_p(model, params, h_T, mu_p))
    prior_sample = np.asarray(
        continue_p(model, params, h_T, mu_p + sig_p * eps))

    actions = jnp.asarray(np.stack(a_dream, axis=1))     # (K, N, A)
    noise = jax.random.normal(
        k_dream, (dream_start, n, model.latent_dim))
    traces = np.asarray(dream_traces(
        model, params, jnp.asarray(np.stack(h_warm)),
        jnp.asarray(np.stack(z_warm)), actions, noise))  # (K, N)

    return {
        "posterior_terminal": np.asarray(post_T),
        "anticipation": np.asarray(post_Tm1),
        "prior_mean": prior_mean,
        "prior_sample": prior_sample,
        "dream_min": traces.min(axis=0),
        "dream_final": traces[-1],
        "calm": np.asarray(calm),
    }


def stats(values):
    v = np.asarray(values, np.float64)
    if v.size == 0:
        return {"n": 0, "median": None, "frac_below_half": None}
    return {"n": int(v.size), "median": float(np.median(v)),
            "frac_below_half": float(np.mean(v < 0.5))}


def with_values(values):
    return {**stats(values), "values": [float(v) for v in values]}


def main():
    parser = argparse.ArgumentParser(
        description="Continue-head diagnostics on a finished train-doom run")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=200,
                        help="cap on death episodes analyzed")
    parser.add_argument("--dream-start", type=int, default=10,
                        help="dream starts this many steps before death")
    parser.add_argument("--out", type=Path, default=None,
                        help="default <run>/diag.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.dream_start < 1:
        raise SystemExit("--dream-start must be at least 1")
    out = args.out if args.out is not None else args.run / "diag.json"

    model, params, source = load_world_model(args.run)
    buffer = ReplayBuffer.load(args.run / "buffer.npz")
    # The buffer keeps its episodes behind a private deque and load() is
    # the only code that knows the npz layout, so I read through it
    # instead of reparsing the file.
    deaths = [ep for ep in buffer._episodes if ep["terminated"]]
    usable = [ep for ep in deaths
              if ep["obs"].shape[0] - 1 >= args.dream_start]
    if not usable:
        raise SystemExit(
            f"no death episode has {args.dream_start}+ steps; "
            f"{len(deaths)} deaths in the buffer")
    if len(usable) > args.windows:
        rng = np.random.default_rng([args.seed, 1])
        idx = np.sort(rng.choice(len(usable), args.windows, replace=False))
        usable = [usable[i] for i in idx]

    key = jax.random.PRNGKey(args.seed)
    m = analyze(model, params, usable, args.dream_start, key)
    dip = float(np.mean(m["dream_min"] < 0.5))

    result = {
        "run": str(args.run),
        "checkpoint": source,
        "seed": args.seed,
        "windows": args.windows,
        "dream_start": args.dream_start,
        "episodes": {
            "in_buffer": buffer.n_episodes,
            "deaths": len(deaths),
            "analyzed": len(usable),
        },
        "one_step_prior": {
            "posterior_terminal": with_values(m["posterior_terminal"]),
            "prior_mean": with_values(m["prior_mean"]),
            "prior_sample": with_values(m["prior_sample"]),
        },
        "anticipation": with_values(m["anticipation"]),
        "dream": {
            "frac_dip_below_half": dip,
            "final": with_values(m["dream_final"]),
            "min": with_values(m["dream_min"]),
        },
        "calm": stats(m["calm"]),
    }
    out.write_text(json.dumps(result, indent=2))

    def line(label, s, ref):
        print(f"  {label:34s} median {s['median']:.3f}  "
              f"{100 * s['frac_below_half']:5.1f}% below 0.5   ({ref})")

    print(f"run {args.run}  ({source})")
    print(f"death episodes: {len(deaths)} in buffer, {len(usable)} analyzed")
    print("one-step prior test at the terminal frame")
    line("posterior", result["one_step_prior"]["posterior_terminal"],
         "v3 median 0.146")
    line("one prior step, mean latent", result["one_step_prior"]["prior_mean"],
         "v3 median 0.201, 73% below 0.5")
    line("one prior step, sampled latent",
         result["one_step_prior"]["prior_sample"], "no v3 reference")
    print("anticipation")
    line("posterior at T-1", result["anticipation"],
         "v3 median 0.715, 30% below 0.5")
    print(f"dream death traces ({args.dream_start} sampled-prior steps, "
          "recorded actions)")
    print(f"  {'rollouts dipping below 0.5':34s} {100 * dip:5.1f}%"
          f"{'':22s}(v3 53.1%)")
    line("P(continue) at the death step", result["dream"]["final"],
         "v3 median 0.864")
    line("minimum P over the trace", result["dream"]["min"],
         "no v3 reference")
    print(f"calm frames ({CALM_MARGIN}+ steps from any episode end)")
    line("posterior", result["calm"], "should be near 0% below 0.5")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
