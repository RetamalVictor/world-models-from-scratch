"""Local-first run tracking.

The run directory is the source of truth: config.json, metrics.jsonl, and
pngs, all readable without any tool. wandb, when enabled, is a mirror of
those same files, never the only copy. Swapping the tracker later touches
this file and nothing else.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def git_state() -> str:
    """Current commit hash, with a -dirty suffix if the tree has changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


class Tracker:
    def __init__(
        self,
        run_dir: str | Path,
        config: dict,
        wandb_project: str | None = None,
        run_name: str | None = None,
    ):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        config = {**config, "git_commit": git_state()}
        (self.dir / "config.json").write_text(json.dumps(config, indent=2))
        self._metrics = open(self.dir / "metrics.jsonl", "a")
        self._wandb = None
        if wandb_project:
            import wandb  # optional dependency: uv sync --extra wandb
            self._wandb = wandb.init(
                project=wandb_project,
                name=run_name,
                config=config,
                dir=str(self.dir),
            )

    def log(self, step: int, **metrics):
        row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        self._metrics.write(json.dumps(row) + "\n")
        self._metrics.flush()
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def log_figure(self, name: str, fig):
        """Save a matplotlib figure into the run dir (and mirror to wandb)."""
        path = self.dir / f"{name}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        if self._wandb is not None:
            import wandb
            self._wandb.log({name: wandb.Image(str(path))})

    def log_json(self, name: str, payload: dict):
        (self.dir / f"{name}.json").write_text(json.dumps(payload, indent=2))

    def finish(self):
        self._metrics.close()
        if self._wandb is not None:
            self._wandb.finish()
