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


def metrics_figure(run_dir: str | Path):
    """Training curves from a run's metrics.jsonl: one panel per metric,
    with its val_* series overlaid when present."""
    import matplotlib.pyplot as plt

    from world_models.plotstyle import BLUE, ORANGE, style

    rows = [
        json.loads(line)
        for line in (Path(run_dir) / "metrics.jsonl").read_text().splitlines()
    ]
    base_keys = list(dict.fromkeys(
        k for r in rows for k in r
        if k != "step" and not k.startswith("val_")
    ))
    fig, axes = plt.subplots(1, len(base_keys),
                             figsize=(4.5 * len(base_keys), 3.2))
    axes = axes if len(base_keys) > 1 else [axes]
    for ax, key in zip(axes, base_keys):
        train = [(r["step"], r[key]) for r in rows if key in r]
        ax.plot(*zip(*train), color=BLUE, lw=1.2, label=key)
        val = [(r["step"], r["val_" + key]) for r in rows if "val_" + key in r]
        if val:
            ax.plot(*zip(*val), color=ORANGE, lw=1.5, label=f"val {key}")
            ax.legend(frameon=False, fontsize=8, labelcolor="#52514e")
        ax.set_xlabel("step")
        ax.set_title(key, fontsize=10)
        style(ax)
    fig.tight_layout()
    return fig


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
