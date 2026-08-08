# VAE implementation proposal (Step 1)

Status: proposal, up for review before any code. Concept-level design is in
[world-model-architectures.md](world-model-architectures.md); this is the
concrete plan: files, shapes, hyperparameters, and what "done" means.

## New files

```
src/world_models/data.py             dataset generation, loading, batching
src/world_models/models/__init__.py
src/world_models/models/vae.py       flax modules: Encoder, Decoder, VAE
src/world_models/probe.py            ridge-regression probe (closed form, jnp)
src/world_models/train_vae.py        config, training loop, artifacts
tests/test_vae.py
tests/test_probe.py
```

Two new entry points in pyproject: `make-data` and `train-vae`.

## Dataset

One fixed dataset, generated once and shared by Steps 1-3 so every model
sees identical data:

- 500 episodes x 200 steps, `random_nudge_policy(0.3)`, master seed 0,
  `vmap(collect_trajectory)` over per-episode keys.
- Stored as a single `.npz` in `data/` (gitignored): obs as uint8 (frames
  times 255), actions and true state as float32. About 100 MB. `uv run
  make-data` regenerates it byte-identically from the seed.
- Fixed episode-level split: 0-399 train, 400-449 val, 450-499 test.
  Frames within an episode are near-duplicates, so a frame-level split
  would leak between train and eval.

The VAE only needs frames: batches are random frames from train episodes,
converted to float32 in [0, 1] at batch time. Val monitors training; test
is touched only by the probe.

The alternative would be generating data on the fly each run (the env is
fast and jittable, no files needed). I prefer the cached file: Steps 2-3
must train on literally the same episodes for the comparison to be clean,
and a file makes that a fact rather than a discipline.

## Model

Latent dim 16. The ball has 4 true degrees of freedom; 16 leaves headroom
without making the probe flatter the model. Encoder:

```
32x32x1 -> Conv 32, 4x4, stride 2 -> 16x16
        -> Conv 64, 4x4, stride 2 ->  8x8
        -> Conv 128, 4x4, stride 2 ->  4x4
        -> flatten -> Dense 256 -> (mu, logvar), 16 each
```

SiLU activations. Decoder mirrors it with a Dense back to 4x4x128 and three
ConvTranspose layers, final layer linear (no output activation). The output
is read as the mean of a fixed-variance Gaussian, so the reconstruction
loss is MSE summed over pixels; distrax handles q(z|o), the reparameterized
sample, and the analytic KL against N(0, I).

Loss: `recon + beta * KL`, beta fixed at 1.0 to start. If the KL collapses
(posterior equals prior, reconstructions turn into the mean image), the
first lever is lowering beta, not free bits; either way it gets a journal
entry.

## Training

- adam, lr 1e-3, global-norm gradient clip at 1.0.
- Batch 128, 20k steps (~34 epochs over 80k train frames). CPU-sized on
  purpose; if a run takes more than ~15 min the model is too big for the
  step.
- Full-jit train step, seeded end to end: one PRNG key in the config
  determines init, batch order, and sampling.
- Config as a frozen dataclass, dumped to `runs/vae/<run-name>/config.json`.
- Tracking through a small `tracking.py` module: the run directory
  (config.json, metrics.jsonl, pngs) is the source of truth and needs no
  tools to read; wandb is an optional mirror, enabled with
  `--wandb-project` after `uv sync --extra wandb`. Only tracking.py knows
  wandb exists, so swapping trackers later is a one-file change.
- Checkpoint via `flax.serialization` to a single msgpack file. Orbax is
  the grown-up answer but is overkill at this scale.
- End-of-run artifacts, saved automatically: a held-out reconstruction
  grid, a grid of prior samples decoded, the loss curves as png, and the
  probe numbers.

## Probe

Closed-form ridge regression in jnp (no sklearn dependency):
`w = (X^T X + lam I)^-1 X^T y` on latent means mu, lam = 1e-3. Protocol,
same for every future model: encode the 50 test episodes, fit on episodes
450-474, report R^2 on 475-499, separately for targets (x, y) and
(vx, vy).

## Step 1 checkpoint (definition of done)

- Held-out reconstructions visually clean, val recon MSE recorded.
- Prior samples decode to single round blobs somewhere in the box.
- Position probe R^2 > 0.9 — the latent should trivially know where the
  ball is.
- Velocity probe R^2 near 0 — expected, and it is the baseline number the
  dynamics models get compared against.
- Journal entry with the numbers and images.

## Outcome (2026-08-09)

beta = 1 from a cold start collapses the posterior (KL = 0, mean-image
decoder). The fix that keeps the beta = 1 target is the warmup knob:
ramp over 5k steps, after which KL settles at ~5.4 nats with no
re-collapse. Position probe lands at 0.75 (not the > 0.9 hoped for —
the code is only partly linear; see journal 0004 for why that and the
collapsed run's misleading probe score are lessons, not failures), and
velocity R^2 is 0.00, which is the baseline this step existed to
produce. Canonical run:

    uv run train-vae --run-name warmup5k --beta 1.0 --beta-warmup-steps 5000

## Tests

- Encoder/decoder/VAE output shapes and finite losses on random input.
- KL is non-negative and zero when mu=0, logvar=0.
- Overfit smoke test: 200 steps on a single batch cuts the loss to well
  under its initial value (catches broken gradients without needing a GPU
  or patience).
- Probe: R^2 ~ 1 on synthetic data with a known linear map, ~0 on pure
  noise targets.

## Decisions (reviewed 2026-08-08)

1. Cached dataset for the ball. Doom will be different: on-policy data
   from a stateful engine can't be byte-identical, so reproducibility
   there shifts to statistical verification — see the reproducibility
   note in [doom-plan.md](doom-plan.md). `make-data` already writes a
   stats report next to the npz for exactly that reason.
2. Latent stays 16.
3. beta = 1, fixed. The config carries a `beta_warmup_steps` knob
   (default 0, meaning constant) because Doom-scale models will need KL
   warmup; the structure exists now, the ball doesn't use it.
4. Tracking is local-first with wandb as the optional mirror (over
   mlflow: better media logging for recon grids and imagined rollouts,
   standard in the RL/world-models community, free hosted tier).
