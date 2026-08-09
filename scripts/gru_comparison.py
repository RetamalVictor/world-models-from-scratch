"""Comparison figure for the Step 2 GRU 2x2 ablation grid.

Run in the clone that has runs/ (the WSL clone):

    uv run python scripts/gru_comparison.py

Reads drift.json and probe.json from the four runs and writes
docs/media/gru-drift-comparison.png.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world_models.plotstyle import AQUA, BASELINE, BLUE, INK2, MUTED, ORANGE, SURFACE, style

RUNS = Path("runs/gru")
MEDIA = Path("docs/media")
YELLOW = "#eda100"  # categorical slot 4

# (name, color, label y-offset in points — staggered because three of the
# four curves plateau at nearly the same value)
GRID = [
    ("direct-mean", BLUE, 8),
    ("residual-mean", ORANGE, 0),
    ("direct-sample", AQUA, -2),
    ("residual-sample", YELLOW, -12),
]

# Reference levels for a 32x32 ball frame (pixel std 0.104):
# predicting the dataset mean image, and drawing the ball in an
# uncorrelated position (~2x the per-pixel variance).
MEAN_IMAGE = 0.104 ** 2
WRONG_PLACE = 2 * MEAN_IMAGE


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))

    for name, color, dy in GRID:
        drift = json.loads((RUNS / name / "drift.json").read_text())
        mse = drift["pixel_mse"]
        ax.plot(drift["horizons"], mse, color=color, lw=1.5)
        ax.annotate(name, (drift["horizons"][-1], mse[-1]),
                    textcoords="offset points", xytext=(4, dy),
                    fontsize=8, color=color)
    for level, label in ((MEAN_IMAGE, "mean image"),
                         (WRONG_PLACE, "ball in the wrong place")):
        ax.axhline(level, ls="--", lw=1, color=BASELINE)
        ax.annotate(label, (1, level), fontsize=7, color=MUTED,
                    textcoords="offset points", xytext=(2, 3))
    ax.set_yscale("log")
    ax.set_xlim(1, 44)
    ax.set_xlabel("open-loop horizon (steps past 4 real transitions)")
    ax.set_ylabel("pixel MSE vs real continuation (log)")
    ax.set_title("open-loop drift", fontsize=10)
    style(ax)

    names = [n for n, _, _ in GRID]
    vels = []
    for name in names:
        probe = json.loads((RUNS / name / "probe.json").read_text())
        vels.append(probe["r2_velocity_mean"])
    x = np.arange(len(names))
    ax2.bar(x, vels, width=0.55,
            color=[c for _, c, _ in GRID], edgecolor=SURFACE)
    ax2.axhline(0.0, lw=1, color=BASELINE)
    for xi, v in zip(x, vels):
        ax2.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom",
                     fontsize=8, color=INK2)
    ax2.annotate("frozen VAE latent probes at 0.00", (0.5, 0.93),
                 xycoords="axes fraction", ha="center",
                 fontsize=8, color=MUTED)
    ax2.set_xticks(x, [n.replace("-", "\n") for n in names], fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("velocity probe R² on h")
    ax2.set_title("velocity finally leaves zero", fontsize=10)
    style(ax2)
    ax2.grid(axis="x", visible=False)

    fig.tight_layout()
    out = MEDIA / "gru-drift-comparison.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
