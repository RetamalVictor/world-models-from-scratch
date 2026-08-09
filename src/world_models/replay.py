"""Episode-aware replay for online training.

Frames are stored uint8 — the same convention as the on-disk datasets —
and converted to float at batch time. Episodes are kept whole and the
oldest are evicted first, so a sampled subsequence can never straddle a
reset. Capacity is counted in frames because that is what RAM cares
about: a Doom episode and a ball episode differ in length, not width.

The buffer round-trips through save()/load() so a resumed run continues
with exactly the data the interrupted run had.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import jax.numpy as jnp
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._episodes: deque[dict] = deque()
        self.frames = 0
        self.frames_added = 0  # lifetime count: the data-budget number

    def __len__(self) -> int:
        return self.frames

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    def add_episode(self, obs: np.ndarray, action: np.ndarray,
                    reward: np.ndarray):
        """obs uint8 (T+1, H, W, C); action (T+1, A); reward (T+1,)."""
        obs = np.asarray(obs)
        if obs.dtype != np.uint8:
            raise ValueError(
                f"store uint8 frames, got {obs.dtype}; convert at batch time"
            )
        self._episodes.append({
            "obs": obs,
            "action": np.asarray(action, np.float32),
            "reward": np.asarray(reward, np.float32),
        })
        self.frames += obs.shape[0]
        self.frames_added += obs.shape[0]
        while self.frames > self.capacity and len(self._episodes) > 1:
            gone = self._episodes.popleft()
            self.frames -= gone["obs"].shape[0]

    def sample_sequences(self, rng: np.random.Generator, batch_size: int,
                         transitions: int):
        """Uniform over every valid (episode, start) pair.

        Returns time-major float batches matching sample_pixel_batch:
        obs (T+1, B, H, W, C) in [0, 1], actions (T+1, B, A),
        rewards (T+1, B).
        """
        episodes = list(self._episodes)
        counts = np.array(
            [ep["obs"].shape[0] - transitions for ep in episodes]
        ).clip(min=0)
        total = int(counts.sum())
        if total <= 0:
            raise ValueError(
                f"no stored episode has {transitions + 1} frames to sample"
            )
        cum = np.cumsum(counts)
        u = rng.integers(0, total, batch_size)
        ep_idx = np.searchsorted(cum, u, side="right")
        t0 = u - (cum[ep_idx] - counts[ep_idx])

        obs = np.stack([episodes[i]["obs"][t:t + transitions + 1]
                        for i, t in zip(ep_idx, t0)])
        act = np.stack([episodes[i]["action"][t:t + transitions + 1]
                        for i, t in zip(ep_idx, t0)])
        rew = np.stack([episodes[i]["reward"][t:t + transitions + 1]
                        for i, t in zip(ep_idx, t0)])
        return (
            jnp.asarray(obs.transpose(1, 0, 2, 3, 4), jnp.float32) / 255.0,
            jnp.asarray(act.transpose(1, 0, 2)),
            jnp.asarray(rew.transpose(1, 0)),
        )

    def save(self, path: str | Path):
        path = Path(path)
        episodes = list(self._episodes)
        tmp = path.with_name(path.stem + ".incoming.npz")
        np.savez_compressed(
            tmp,
            obs=np.concatenate([ep["obs"] for ep in episodes]),
            action=np.concatenate([ep["action"] for ep in episodes]),
            reward=np.concatenate([ep["reward"] for ep in episodes]),
            lengths=np.array([ep["obs"].shape[0] for ep in episodes]),
            capacity=self.capacity,
            frames_added=self.frames_added,
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        with np.load(path) as f:
            buffer = cls(int(f["capacity"]))
            lengths = f["lengths"]
            obs, act, rew = f["obs"], f["action"], f["reward"]
            for start, length in zip(
                np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths
            ):
                buffer.add_episode(obs[start:start + length],
                                   act[start:start + length],
                                   rew[start:start + length])
            buffer.frames_added = int(f["frames_added"])
        return buffer
