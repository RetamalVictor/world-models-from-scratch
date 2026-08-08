# world models, from scratch

I'm building world models step by step in JAX: the Ha & Schmidhuber (2018)
recipe first, then Dreamer's RSSM, and eventually an agent that plays DOOM.
This repo doubles as material for a tutorial I'm writing along the way.

The core idea behind the setup: before trusting a world model on a real game,
train it on a bouncing ball where I control the simulator and know the true
state (position, velocity) behind every frame. That makes the interesting
question measurable. Instead of eyeballing reconstructions, I can fit a linear
probe from the latent to the true velocity and get a number that says whether
the model actually learned the dynamics. Real benchmarks hide that ground
truth from you.

![bouncing ball rollout](docs/media/bouncing_ball.gif)

## Status

- [x] Step 0: bouncing ball environment (gymnax-style, pure JAX)
- [ ] Step 1: VAE + velocity probe baseline
- [ ] Step 2: frozen encoder + GRU dynamics (Ha-style)
- [ ] Step 3: joint RSSM (Dreamer-style)
- [ ] Step 4: actor-critic in imagination
- [ ] Step 5: comparison writeup
- [ ] Step 6: DOOM (ViZDoom take_cover)

The full plan with per-step checkpoints is in
[docs/design/roadmap.md](docs/design/roadmap.md).

## Setup

The project uses [uv](https://docs.astral.sh/uv/):

```
uv sync
```

Render a random rollout of the ball env:

```
uv run ball-demo --gif ball.gif
```

Run the tests:

```
uv run pytest
```

Training runs on the GPU (an RTX 5070 Ti) through WSL2, since JAX has no
native Windows CUDA support; the Windows side is for development and the
Ubuntu side for runs. Setup and per-session flow are in
[docs/gpu-setup.md](docs/gpu-setup.md).

## Layout

```
src/world_models/    the package (envs now, models as the steps land)
scripts/             analysis and figure scripts
tests/               pytest suite
docs/design/         design notes, one file per component
docs/journal/        dev log, one entry per work session
docs/posts/          tutorial drafts, written from the journal
```

The journal is the raw material for the tutorial: what was built, what broke,
and why the design ended up the way it did.
