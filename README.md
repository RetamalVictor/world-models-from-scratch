# world models

JAX implementations of three latent world models (VAE, frozen-encoder
GRU, RSSM), an actor-critic that trains inside them, an online
collect/train loop, and two environments.

The same models run on both environments. The bouncing ball is the
validation harness: its simulator reports the true `(x, y, vx, vy)`
behind every frame, so a linear probe can measure whether a latent
actually encodes the dynamics. ViZDoom `take_cover` is the target, where
no such ground truth exists.

Write-ups live on [victor-retamal.com](https://victor-retamal.com). This
repo is the code and the numbers.

## What is here

| component | file | notes |
|---|---|---|
| Bouncing ball env | `envs/bouncing_ball.py` | pure JAX, gymnax-style, reports true `(x, y, vx, vy)` per frame |
| Goal env (RGB, reward) | `envs/bouncing_ball_goal.py` | second ball as target, static or moving |
| ViZDoom `take_cover` | `envs/doom.py` | stateful CPU wrapper, not JAX; 64x64 grayscale out |
| Conv VAE | `models/vae.py` | 32x32 in, 16-dim diagonal Gaussian latent |
| Ridge probe | `probe.py` | closed form, no sklearn |
| GRU dynamics | `models/gru_dynamics.py` | frozen-encoder, direct or residual prediction |
| RSSM | `models/rssm.py` | GRU core, prior/posterior heads, KL balancing, reward and continue heads |
| Actor-critic | `models/actor_critic.py` | TD(lambda) returns, value gradients through the dynamics |
| Open-loop rollout | `rollout.py` | shared drift protocol for every dynamics model |
| Replay buffer | `replay.py` | uint8, episode-aware eviction, save/load |
| Checkpointing | `checkpoint.py` | whole training pytree, keep-last-K plus best-by-metric |

## Install

Requires [uv](https://docs.astral.sh/uv/). CPU install works anywhere:

```
uv sync
uv run pytest          # 86 tests; doom tests skip without the extra
```

Optional extras:

```
uv sync --extra cuda    # GPU training (Linux only; on Windows use WSL2)
uv sync --extra wandb   # mirror metrics to wandb; local jsonl stays canonical
uv sync --extra doom    # ViZDoom, ships the scenario and a free WAD
```

JAX has no native Windows CUDA support, so on Windows the GPU path is a
WSL2 clone: install the Ubuntu distro, clone there, `uv sync --extra
cuda`, and check `jax.default_backend()` reports `gpu`. Every run records
its backend and commit hash in `config.json`.

## Usage

Every command is seeded and writes config, `metrics.jsonl`, figures and
gifs into `runs/<family>/<name>/`.

```
uv run ball-demo --gif ball.gif           # render a random rollout

uv run make-data                          # ball dataset, 500 x 200 steps
uv run make-data --task follow            # moving-goal dataset

uv run train-vae  --run-name warmup5k --beta 1.0 --beta-warmup-steps 5000
uv run train-gru  --predict direct --latent-source mean
uv run train-rssm --run-name base
uv run train-ac   --wm-run runs/rssm/follow --data data/ball_follow.npz \
                  --goal-speed 1.0 --run-name follow

uv run train-online --run-name follow     # collect, train WM, train AC
uv run --extra doom train-doom --help     # ViZDoom pretraining
```

`train-gru` takes `--predict {direct,residual}` and `--latent-source
{mean,sample}`. `train-online` supports `--resume`, which continues a
killed run to byte-identical checkpoints, and `--final-ac-steps`, which
re-fits a fresh actor against the finished world model.

## What to expect

Ridge probe from model state to true ball velocity. Same dataset, same
episode-level split, same 128-dim state where sizes would otherwise
differ:

| model | velocity R² | position R² |
|---|---|---|
| VAE latent (single frame) | 0.00 | 0.75 |
| frozen VAE + GRU | 0.81 | 0.98 |
| joint RSSM | 0.96 | 0.998 |

Open-loop drift, pixel MSE against the real continuation after 4 warm-up
transitions. Reference levels for this dataset: 0.011 is the mean image,
0.022 is a correct ball in an uncorrelated place.

| model | k=1 | k=5 | k=15 | k=30 |
|---|---|---|---|---|
| frozen VAE + GRU (best per column) | 0.00035 | 0.0053 | 0.0162 | 0.0177 |
| joint RSSM | 0.000031 | 0.00039 | 0.0036 | 0.0129 |

Control, mean per-step reward over 100 episodes. Zero-action and
random-nudge baselines both score about 0.05.

| setup | reward |
|---|---|
| offline actor-critic, hover | 0.652 |
| offline actor-critic, follow | 0.879 |
| online loop, co-trained actor | 0.801 |
| online loop, actor re-fit on the frozen model | 0.911 |

Runtimes on an RTX 5070 Ti: VAE about 2 minutes, GRU about 3 minutes,
RSSM about 20 minutes, actor-critic about 5 minutes. CPU runs match to a
few decimals; float32 reduction order differs across backends.

Doom status: the world model is pretrained on 232k random `take_cover`
frames and the continue head predicts death inside imagination. There is
no trained Doom agent in this repo yet.

## Layout

```
src/world_models/    envs, models, training entry points
scripts/             scripts that rebuild the write-up figures from runs/
tests/               pytest suite
```

## License

MIT, see [LICENSE](LICENSE).
