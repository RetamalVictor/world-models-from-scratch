"""Episode-aware replay for online training.

Frames are stored uint8 — the same convention as the on-disk datasets —
and converted to float at batch time. Episodes are kept whole and the
oldest are evicted first, so a sampled subsequence can never straddle a
reset. Capacity is counted in frames because that is what RAM cares
about: a Doom episode and a ball episode differ in length, not width.

The buffer round-trips through save()/load() so a resumed run continues
with exactly the data the interrupted run had.

Episodes carry a terminated flag: True means the agent died, False means
the collector simply stopped (a time limit). Only the first kind ends
the world, and only the first kind produces a zero continue target.

Campaign episodes carry two more things, both optional and both
inert as far as training is concerned: the game variables recorded
beside every frame, and the flavor the episode ended with. They ride
along with their episode through saving, loading and eviction, and no
sampler ever puts them in a batch. They exist for dataset reports and
for probes, which need ground truth behind the pixels.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def _continues(episode: dict) -> np.ndarray:
    """1.0 while the episode continues past frame t.

    Only the last frame of a death gets a 0.0. A timeout episode stays
    all ones: a time limit is the collector stopping, not the world
    ending, and training the continue head on it would teach the model
    to expect death at a fixed clock.
    """
    c = np.ones(episode["obs"].shape[0], np.float32)
    if episode["terminated"]:
        c[-1] = 0.0
    return c


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
                    reward: np.ndarray, terminated: bool = False,
                    variables: np.ndarray | None = None,
                    reason: str | None = None):
        """obs uint8 (T+1, H, W, C); action (T+1, A); reward (T+1,).

        terminated: the episode ended in death rather than a time limit.

        variables (T+1, n_vars) float32 and reason are the campaign
        collector's extras and default to absent, so every existing
        caller stores exactly what it stored before. A buffer either
        carries variables for every episode or for none, because save()
        concatenates them into one array; mixing is a caller bug and is
        rejected here rather than at the next checkpoint, an hour into
        a run.
        """
        obs = np.asarray(obs)
        if obs.dtype != np.uint8:
            raise ValueError(
                f"store uint8 frames, got {obs.dtype}; convert at batch time"
            )
        if variables is not None:
            variables = np.asarray(variables, np.float32)
            if variables.shape[0] != obs.shape[0]:
                raise ValueError(
                    f"variables has {variables.shape[0]} rows for "
                    f"{obs.shape[0]} frames; one row per frame"
                )
        if self._episodes:
            stored = self._episodes[0]["variables"]
            if (stored is None) != (variables is None):
                raise ValueError(
                    "either every episode in a buffer carries variables "
                    "or none does"
                )
            if stored is not None and stored.shape[1] != variables.shape[1]:
                raise ValueError(
                    f"variables has {variables.shape[1]} columns, the "
                    f"buffer stores {stored.shape[1]}"
                )
        self._episodes.append({
            "obs": obs,
            "action": np.asarray(action, np.float32),
            "reward": np.asarray(reward, np.float32),
            "terminated": bool(terminated),
            "variables": variables,
            "reason": None if reason is None else str(reason),
        })
        self.frames += obs.shape[0]
        self.frames_added += obs.shape[0]
        while self.frames > self.capacity and len(self._episodes) > 1:
            gone = self._episodes.popleft()
            self.frames -= gone["obs"].shape[0]

    def _draw(self, rng: np.random.Generator, batch_size: int,
              transitions: int):
        """Uniform over every valid (episode, start) pair."""
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
        return episodes, ep_idx, t0

    def _gather(self, episodes, ep_idx, t0, transitions):
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

    def longest_episode(self) -> dict:
        """The stored episode with the most frames — filmstrip material."""
        return max(self._episodes, key=lambda ep: ep["obs"].shape[0])

    def sample_sequences(self, rng: np.random.Generator, batch_size: int,
                         transitions: int):
        """Uniform over every valid (episode, start) pair.

        Returns time-major float batches matching sample_pixel_batch:
        obs (T+1, B, H, W, C) in [0, 1], actions (T+1, B, A),
        rewards (T+1, B).
        """
        draw = self._draw(rng, batch_size, transitions)
        return self._gather(*draw, transitions)

    def sample_sequences_with_continues(self, rng: np.random.Generator,
                                        batch_size: int, transitions: int):
        """sample_sequences plus continue targets (T+1, B).

        Same draw from the same rng, so the extra array is the only
        difference from sample_sequences on an identically seeded call.
        """
        episodes, ep_idx, t0 = self._draw(rng, batch_size, transitions)
        con = np.stack([_continues(episodes[i])[t:t + transitions + 1]
                        for i, t in zip(ep_idx, t0)])
        return self._gather(episodes, ep_idx, t0, transitions) + (
            jnp.asarray(con.transpose(1, 0)),
        )

    def save(self, path: str | Path):
        # Uncompressed: a full buffer is a few hundred MB of frames and
        # zlib spends more time on it than the checkpoint interval can
        # afford. np.load reads either format, so older compressed
        # buffers still restore.
        path = Path(path)
        episodes = list(self._episodes)
        tmp = path.with_name(path.stem + ".incoming.npz")
        fields = dict(
            obs=np.concatenate([ep["obs"] for ep in episodes]),
            action=np.concatenate([ep["action"] for ep in episodes]),
            reward=np.concatenate([ep["reward"] for ep in episodes]),
            lengths=np.array([ep["obs"].shape[0] for ep in episodes]),
            terminated=np.array([ep["terminated"] for ep in episodes], bool),
            capacity=self.capacity,
            frames_added=self.frames_added,
        )
        # The campaign fields are written only when they exist, so a
        # buffer that never had them saves the file it always did.
        if episodes and episodes[0]["variables"] is not None:
            fields["variables"] = np.concatenate(
                [ep["variables"] for ep in episodes])
        if any(ep["reason"] is not None for ep in episodes):
            # The empty string is "not recorded": npz has no None.
            fields["reason"] = np.array(
                [ep["reason"] or "" for ep in episodes])
        np.savez(tmp, **fields)
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        with np.load(path) as f:
            buffer = cls(int(f["capacity"]))
            lengths = f["lengths"]
            obs, act, rew = f["obs"], f["action"], f["reward"]
            # Files written before the flag existed came from the ball,
            # which only ever times out.
            terminated = (f["terminated"] if "terminated" in f
                          else np.zeros(len(lengths), bool))
            # Both campaign fields are absent from every dataset written
            # before the campaign env existed, and from every take_cover
            # one written after it.
            variables = f["variables"] if "variables" in f else None
            reasons = f["reason"] if "reason" in f else None
            for i, (start, length, term) in enumerate(zip(
                np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths,
                terminated,
            )):
                reason = None if reasons is None else str(reasons[i]) or None
                buffer.add_episode(
                    obs[start:start + length],
                    act[start:start + length],
                    rew[start:start + length], bool(term),
                    variables=(None if variables is None
                               else variables[start:start + length]),
                    reason=reason)
            buffer.frames_added = int(f["frames_added"])
        return buffer
