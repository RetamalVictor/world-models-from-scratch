"""Ball dataset: generation, loading, splits, batching.

One fixed dataset shared by Steps 1-3, regenerable byte-identically:

    uv run make-data

Splits are at the episode level (frames within an episode are near
duplicates, a frame-level split would leak between train and eval).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from world_models.envs import EnvParams, collect_trajectory, random_nudge_policy

DEFAULT_PATH = Path("data/ball.npz")


def generate(
    seed: int = 0,
    n_episodes: int = 500,
    n_steps: int = 200,
    nudge: float = 0.3,
    params: EnvParams = EnvParams(),
) -> dict:
    """Generate the dataset as host numpy arrays. Deterministic in seed.

    obs is stored as uint8 to keep the file small; everything else float32.
    Shapes have leading dims (n_episodes, n_steps + 1).
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), n_episodes)
    policy = random_nudge_policy(nudge)
    rollout = jax.vmap(
        lambda k: collect_trajectory(k, policy, n_steps, params)
    )(keys)
    rollout = jax.device_get(rollout)
    return {
        "obs": np.round(rollout["obs"] * 255.0).astype(np.uint8),
        "action": rollout["action"].astype(np.float32),
        "x": rollout["x"].astype(np.float32),
        "y": rollout["y"].astype(np.float32),
        "vx": rollout["vx"].astype(np.float32),
        "vy": rollout["vy"].astype(np.float32),
    }


def stats_report(dataset: dict, meta: dict) -> dict:
    """Summary statistics of a generated dataset.

    For the ball this is a sanity check. Once data comes from a real env
    on-policy (Doom), byte-identical regeneration stops being possible and
    comparing these reports across collection rounds becomes the way to
    verify two runs saw equivalent data.
    """
    obs = dataset["obs"].astype(np.float32) / 255.0
    speed = np.sqrt(dataset["vx"] ** 2 + dataset["vy"] ** 2)
    return {
        **meta,
        "pixel_mean": float(obs.mean()),
        "pixel_std": float(obs.std()),
        "action_mean": [float(m) for m in dataset["action"].mean(axis=(0, 1))],
        "action_std": [float(s) for s in dataset["action"].std(axis=(0, 1))],
        "vx_mean": float(dataset["vx"].mean()),
        "vx_std": float(dataset["vx"].std()),
        "vy_mean": float(dataset["vy"].mean()),
        "vy_std": float(dataset["vy"].std()),
        "speed_mean": float(speed.mean()),
    }


def load(path: str | Path = DEFAULT_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; generate it with `uv run make-data`"
        )
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def splits(n_episodes: int) -> tuple[slice, slice, slice]:
    """80/10/10 episode-level split as (train, val, test) slices."""
    n_train = int(0.8 * n_episodes)
    n_val = int(0.1 * n_episodes)
    return (
        slice(0, n_train),
        slice(n_train, n_train + n_val),
        slice(n_train + n_val, n_episodes),
    )


def frames_of(dataset: dict, split: slice) -> np.ndarray:
    """Flatten a split's episodes to a (N, H, W, 1) uint8 frame array."""
    obs = dataset["obs"][split]
    return obs.reshape((-1,) + obs.shape[2:])


def to_float(frames_uint8: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(frames_uint8, jnp.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description="Generate the ball dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--nudge", type=float, default=0.3)
    parser.add_argument("--out", type=str, default=str(DEFAULT_PATH))
    args = parser.parse_args()

    dataset = generate(args.seed, args.episodes, args.steps, args.nudge)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **dataset)

    meta = {
        "seed": args.seed,
        "n_episodes": args.episodes,
        "n_steps": args.steps,
        "nudge": args.nudge,
    }
    report = stats_report(dataset, meta)
    stats_path = out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(report, indent=2))

    size_mb = out.stat().st_size / 1e6
    print(f"saved {out} ({size_mb:.0f} MB) and {stats_path}")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
