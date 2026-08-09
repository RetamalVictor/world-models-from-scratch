"""Checkpoint/resume for long runs.

A checkpoint is one msgpack file holding an arbitrary pytree — train
states, counters, anything flax can serialize. Files live under
<run_dir>/checkpoints/ as step_00000123.msgpack, pruned to the last K,
plus an optional best-by-metric copy that pruning never touches.

Restoring needs a template tree with the right structure (flax
deserialization is structural), which every trainer already has for
free: build the initial states exactly as a fresh run would, then
restore into them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from flax import serialization


class Checkpointer:
    def __init__(self, run_dir: str | Path, keep: int = 3):
        self.dir = Path(run_dir) / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep = keep

    def _path(self, step: int) -> Path:
        return self.dir / f"step_{step:08d}.msgpack"

    def steps(self) -> list[int]:
        found = []
        for p in self.dir.glob("step_*.msgpack"):
            m = re.fullmatch(r"step_(\d+)\.msgpack", p.name)
            if m:
                found.append(int(m.group(1)))
        return sorted(found)

    def latest_step(self) -> int | None:
        steps = self.steps()
        return steps[-1] if steps else None

    def save(self, step: int, tree, metric: float | None = None):
        """Write a checkpoint and prune to the last K.

        metric, when given, is higher-is-better; the best checkpoint so
        far is kept as best.msgpack outside the pruning rotation.
        """
        data = serialization.to_bytes(tree)
        tmp = self.dir / "incoming.msgpack"
        tmp.write_bytes(data)
        tmp.replace(self._path(step))
        for old in self.steps()[: -self.keep]:
            self._path(old).unlink()
        if metric is not None:
            best = self.best_meta()
            if best is None or float(metric) > best["metric"]:
                (self.dir / "best.msgpack").write_bytes(data)
                (self.dir / "best.json").write_text(
                    json.dumps({"step": step, "metric": float(metric)})
                )

    def best_meta(self) -> dict | None:
        path = self.dir / "best.json"
        return json.loads(path.read_text()) if path.exists() else None

    def restore(self, template, step: int | None = None):
        """Restore (step, tree); step None means latest."""
        if step is None:
            step = self.latest_step()
            if step is None:
                raise FileNotFoundError(f"no checkpoints in {self.dir}")
        tree = serialization.from_bytes(template, self._path(step).read_bytes())
        return step, tree

    def restore_best(self, template):
        meta = self.best_meta()
        if meta is None:
            raise FileNotFoundError(f"no best checkpoint in {self.dir}")
        tree = serialization.from_bytes(
            template, (self.dir / "best.msgpack").read_bytes()
        )
        return meta["step"], tree
