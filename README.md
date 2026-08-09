# world models, from scratch

World models from scratch in JAX, built up step by step: the
Ha & Schmidhuber (2018) recipe first, then Dreamer's RSSM, then an agent
that learns to act entirely inside its own imagination. The write-ups that
go with each step land on [victor-retamal.com](https://victor-retamal.com);
this repo is the code, and every number in the posts reproduces from here.

The core idea behind the setup: before trusting a world model on a real
game, train it on a bouncing ball where I control the simulator and know
the true state (position, velocity) behind every frame. That makes the
interesting question measurable. Instead of eyeballing reconstructions, I
fit a linear probe from the latent to the true velocity and get a number
that says whether the model actually learned the dynamics. Real benchmarks
hide that ground truth from you.

![bouncing ball rollout](docs/media/bouncing_ball.gif)

## What's inside

- Step 0 — bouncing ball environment (gymnax-style, pure JAX)
- Step 1 — VAE + the velocity probe baseline
- Step 2 — frozen encoder + GRU dynamics (Ha-style)
- Step 3 — joint RSSM (Dreamer-style)
- Step 4 — actor-critic in imagination (goal env, reward from pixels)
- The online loop: collect -> train, checkpoint/resume, replay buffer
- A ViZDoom take_cover wrapper, 64x64 model scaling, and the continue
  head a terminating environment needs

The measurement that threads the steps together — a ridge probe from the
model state to the true ball velocity, identical protocol everywhere:

| model                     | velocity probe R² |
|---------------------------|-------------------|
| VAE latent (Step 1)       | 0.00              |
| frozen VAE + GRU (Step 2) | 0.81              |
| joint RSSM (Step 3)       | 0.96              |

And the control scores (mean per-step reward, 100 episodes, baselines
score ~0.05): offline actor-critic on the follow task 0.879; the online
loop's co-trained actor 0.801; a fresh actor re-fit against the frozen
online world model 0.911.

## Setup

The project uses [uv](https://docs.astral.sh/uv/); any platform works for
the CPU install:

```
uv sync
uv run ball-demo --gif ball.gif   # render a random rollout of the env
uv run pytest                     # 69 tests; doom tests skip without the extra
```

Extras, all optional:

```
uv sync --extra cuda    # GPU training (Linux only; on Windows use WSL2)
uv sync --extra wandb   # mirror metrics to wandb; local jsonl stays canonical
uv sync --extra doom    # ViZDoom -- ships the scenario and a free WAD
```

I train on an RTX 5070 Ti through WSL2 (JAX has no native Windows CUDA
support): clone the repo inside WSL, `uv sync --extra cuda`, and the same
commands below run on the GPU. Numbers here came from that GPU; CPU runs
match to a few decimals (backends differ in the last bits of float32).

## Reproduce

Everything is seeded; each command writes config, metrics (jsonl), figures
and gifs into its own `runs/<family>/<name>/` directory.

```
uv run make-data                        # Steps 1-3 dataset (plain ball)
uv run make-data --task follow          # Step 4 dataset (moving red goal)

uv run train-vae  --run-name base       # Step 1  (see --beta-warmup-steps)
uv run train-gru  --run-name base       # Step 2  (see --predict, --latent-source)
uv run train-rssm --run-name base       # Step 3
uv run train-rssm --data data/ball_follow.npz --run-name follow
uv run train-ac   --wm-run runs/rssm/follow --data data/ball_follow.npz \
                  --goal-speed 1.0 --run-name follow            # Step 4

uv run train-online --run-name follow --no-wandb   # the online loop
```

The online run collects its own data (no dataset needed), checkpoints
every 25 rounds, and `--resume` continues a killed run exactly — checkpoints
are byte-identical to an uninterrupted run, and the test suite enforces
that. It ends with a fresh actor-critic re-fit against the final frozen
world model (`--final-ac-steps`), which is where the 0.911 comes from.

## Layout

```
src/world_models/    envs, models, training entry points
scripts/             figure scripts for the write-ups
tests/               pytest suite
docs/media/          gifs and figures referenced by the posts
```

## License

MIT — see [LICENSE](LICENSE).
