"""Figures for the Step 3+4 post. Run in the clone that has runs/ (WSL):

    uv run python scripts/step34_figures.py

Writes into docs/media/: the three-model comparison, the actor-critic
training/eval figure, and the rollout frame strips.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world_models.plotstyle import (AQUA, BASELINE, BLUE, INK2, MUTED, ORANGE,
                                    SURFACE, style)

RUNS = Path("runs")
MEDIA = Path("docs/media")

MEAN_IMAGE = 0.104 ** 2
WRONG_PLACE = 2 * MEAN_IMAGE


def _drift(path):
    return json.loads((path / "drift.json").read_text())


def _probe(path, key):
    return json.loads((path / "probe.json").read_text())[key]


def three_model_comparison():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))

    curves = [
        ("frozen GRU (direct-mean)", RUNS / "gru/direct-mean", ORANGE, -2),
        ("frozen GRU (residual-sample)", RUNS / "gru/residual-sample", AQUA, -12),
        ("joint RSSM", RUNS / "rssm/base", BLUE, 6),
    ]
    for label, path, color, dy in curves:
        d = _drift(path)
        ax.plot(d["horizons"], d["pixel_mse"], color=color, lw=1.5)
        ax.annotate(label, (d["horizons"][-1], d["pixel_mse"][-1]),
                    textcoords="offset points", xytext=(4, dy),
                    fontsize=8, color=color)
    for level, label in ((MEAN_IMAGE, "mean image"),
                         (WRONG_PLACE, "ball in the wrong place")):
        ax.axhline(level, ls="--", lw=1, color=BASELINE)
        ax.annotate(label, (1, level), fontsize=7, color=MUTED,
                    textcoords="offset points", xytext=(2, 3))
    ax.set_yscale("log")
    ax.set_xlim(1, 52)
    ax.set_xlabel("open-loop horizon (steps past 4 real transitions)")
    ax.set_ylabel("pixel MSE vs real continuation (log)")
    ax.set_title("open-loop drift, frozen vs joint", fontsize=10)
    style(ax)

    bars = [
        ("VAE\n(single frame)",
         _probe(RUNS / "vae/warmup5k", "r2_velocity_mean"), MUTED),
        ("frozen GRU\n(h)",
         _probe(RUNS / "gru/direct-mean", "r2_velocity_mean"), ORANGE),
        ("joint RSSM\n(h)",
         _probe(RUNS / "rssm/base", "r2_velocity_h_mean"), BLUE),
    ]
    x = np.arange(len(bars))
    vals = [max(0.0, v) for _, v, _ in bars]
    ax2.bar(x, vals, width=0.55, color=[c for _, _, c in bars],
            edgecolor=SURFACE)
    for xi, v in zip(x, vals):
        ax2.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom",
                     fontsize=9, color=INK2)
    ax2.set_xticks(x, [n for n, _, _ in bars], fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("velocity probe R²")
    ax2.set_title("does the state know velocity?", fontsize=10)
    style(ax2)
    ax2.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(MEDIA / "three-model-comparison.png", dpi=150,
                facecolor=SURFACE, bbox_inches="tight")


def ac_training_eval():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))

    for task, color in (("hover", BLUE), ("follow", ORANGE)):
        rows = [json.loads(l) for l in
                (RUNS / f"ac/{task}/metrics.jsonl").read_text().splitlines()]
        imag = [(r["step"], r["imag_reward"]) for r in rows if "imag_reward" in r]
        real = [(r["step"], r["real_reward"]) for r in rows if "real_reward" in r]
        ax.plot(*zip(*imag), color=color, lw=1.0, alpha=0.4)
        ax.plot(*zip(*real), color=color, lw=1.8, marker="o", ms=4)
        ax.annotate(f"{task} (real)", real[-1], textcoords="offset points",
                    xytext=(-8, 8), fontsize=8, color=color, ha="right")
    ax.annotate("thin lines: imagined reward", (0.03, 0.55),
                xycoords="axes fraction", fontsize=8, color=MUTED)
    ax.set_xlabel("actor-critic training step")
    ax.set_ylabel("mean per-step reward")
    ax.set_ylim(0, 1)
    ax.set_title("imagination trains, reality improves", fontsize=10)
    style(ax)

    tasks = ("hover", "follow")
    policies = ("actor", "zero", "random")
    colors = {"actor": BLUE, "zero": BASELINE, "random": MUTED}
    width = 0.25
    for i, policy in enumerate(policies):
        vals, errs = [], []
        for task in tasks:
            e = json.loads((RUNS / f"ac/{task}/eval.json").read_text())
            vals.append(e[policy]["mean_reward"])
            errs.append(e[policy]["std"])
        xs = np.arange(len(tasks)) + (i - 1) * width
        ax2.bar(xs, vals, width=width * 0.9, color=colors[policy],
                edgecolor=SURFACE, label=policy, yerr=errs, capsize=3,
                error_kw={"ecolor": INK2, "elinewidth": 1})
        for xi, v in zip(xs, vals):
            ax2.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom",
                         fontsize=8, color=INK2)
    ax2.set_xticks(np.arange(len(tasks)), tasks)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("mean per-step reward (100 episodes)")
    ax2.set_title("final eval vs baselines", fontsize=10)
    ax2.legend(frameon=False, fontsize=8, labelcolor=INK2)
    style(ax2)
    ax2.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(MEDIA / "ac-training-eval.png", dpi=150,
                facecolor=SURFACE, bbox_inches="tight")


def rollout_strips():
    rows = []
    for task in ("hover", "follow"):
        gif = Image.open(RUNS / f"ac/{task}/real_rollout.gif")
        frames = []
        for i in (0, 15, 30, 50, 75, 100, 125, 149):
            gif.seek(min(i, gif.n_frames - 1))
            frames.append(np.asarray(gif.convert("RGB")))
        gap = np.full((frames[0].shape[0], 6, 3), 252, np.uint8)
        row = []
        for f in frames:
            row += [f, gap]
        rows.append(np.concatenate(row[:-1], axis=1))
    vgap = np.full((10, rows[0].shape[1], 3), 252, np.uint8)
    strip = np.concatenate([rows[0], vgap, rows[1]], axis=0)
    img = Image.fromarray(strip)
    img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
    img.save(MEDIA / "ac-rollout-strips.png")


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    three_model_comparison()
    ac_training_eval()
    rollout_strips()
    print(f"wrote figures to {MEDIA}/")


if __name__ == "__main__":
    main()
