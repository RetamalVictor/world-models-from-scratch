"""Backfill or regenerate loss_curves.png for existing runs.

    uv run python scripts/plot_metrics.py runs/gru/direct-mean [more...]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from world_models.tracking import metrics_figure


def main():
    for arg in sys.argv[1:]:
        run_dir = Path(arg)
        fig = metrics_figure(run_dir)
        out = run_dir / "loss_curves.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
