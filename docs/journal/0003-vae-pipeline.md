# 0003 - VAE pipeline built, first surprise already logged

2026-08-08, continued from 0002.

The Step 1 proposal (docs/design/vae.md) survived review with small
amendments, now recorded in its decisions section: cached dataset for the
ball (Doom will need on-the-fly collection verified statistically), latent
stays 16, beta fixed at 1, and tracking is local-first files with wandb as
an optional mirror — wandb over mlflow because image/video logging is the
part I'll actually use (recon grids now, imagined rollouts at Step 3).

Implemented: `data.py` (make-data, stats report), `probe.py` (closed-form
ridge, no sklearn), `models/vae.py`, `tracking.py`, `train_vae.py`, and
tests for all of it. Even though beta is fixed, the config carries a
`beta_warmup_steps` knob (default 0 = constant): Doom-scale models will
want KL warmup, so the structure exists now and turning it on later is a
config change, not a code change.

Two finds before the first real run:

1. `jax.random.fold_in(key, -1)` crashes — fold_in data must be
   non-negative. I was abusing -1 as a "special" step for the artifact
   key. Proper fix, and consistent with the 0002 lesson: split a
   dedicated artifact key from the root at startup.

2. More interesting: a 1000-step CPU smoke run showed the KL sitting at
   ~0.001 with recon stuck at ~11.2. That recon number is exactly the
   dataset's pixel variance (0.104^2 x 1024 pixels), which means the
   decoder is painting the mean image and the latent is unused: posterior
   collapse, live and on schedule. It may just be too few steps — the
   overfit test passes, so gradients flow — but if it persists in the
   full run, the levers are lower beta or that warmup knob I just added
   "for Doom". Funny how that goes.

Training moves to the GPU (RTX 5070 Ti) via a WSL2 Ubuntu clone; runbook
in docs/gpu-setup.md. Every run now records its JAX backend in
config.json, so CPU and GPU results can't get mixed up silently.

Next: first full run on the GPU, then the Step 1 checkpoint numbers.
