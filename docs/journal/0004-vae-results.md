# 0004 - First GPU runs: collapse, recovery, and a surprise about probes

2026-08-09, small hours.

GPU first light: the 5070 Ti works through WSL2, `jax.default_backend()`
says gpu, and a 20k-step VAE run takes a couple of minutes. Every run
records its backend and commit hash in config.json, which already paid off
below.

One determinism caveat found immediately: `make-data` on the GPU produces a
dataset that differs from the CPU one in the 8th decimal of the stats
(float reduction order differs across backends, so a few pixels at the
uint8 rounding boundary flip). "Byte-identical from seed" holds per
backend, not across backends. Fine for us — Steps 1-3 all train on the one
file generated in the WSL clone — but worth remembering before ever
claiming bitwise reproducibility across machines.

## The three runs

| run      | beta schedule    | KL (nats) | val recon | pos R^2 | vel R^2 |
|----------|------------------|-----------|-----------|---------|---------|
| base     | 1.0 from step 0  | 0.00      | 11.17     | 0.74 *  | -0.005  |
| beta01   | 0.1 fixed        | ~9.6      | 0.06      | 0.35    | -0.013  |
| warmup5k | 1.0, 5k warmup   | ~5.4      | 0.28      | 0.75    | -0.013  |

**base** is the posterior collapse from 0003, confirmed at full length: KL
pinned at zero for 20k steps and val recon parked at 11.17, which is
exactly the dataset's pixel variance — the decoder paints the mean image
and the latent does nothing. The recon grid is a row of black frames
(docs/media/vae-recon-collapsed.png). With most of the image black, beta=1
from step 0 makes "ignore the latent" a local optimum the model never
leaves.

**beta01** escapes it: KL ~9.6 nats, essentially perfect reconstructions
(docs/media/vae-recon-beta01.png). But the position probe reads only 0.35.

**warmup5k** — beta ramped 0 to 1 over 5k steps, then held — is the
interesting one. KL descends smoothly to ~5.4 and *stays* there after the
ramp ends: no re-collapse once the decoder is wired up. Reconstructions
good, prior samples are single round balls with the occasional double
(docs/media/vae-recon-warmup.png, vae-prior-samples.png), and position
R^2 is 0.75, double the beta=0.1 run.

The knob I added "for Doom" got used on day one.

## Two lessons I didn't expect to learn at Step 1

1. *Reconstruction proves information presence, not linear accessibility.*
   beta01 reconstructs the ball's position pixel-perfectly, so (x, y) is
   in the latent beyond doubt — yet a linear probe only recovers R^2 0.35.
   The code is entangled. More KL pressure (warmup5k) buys a more linearly
   readable code, which is the beta-VAE disentanglement story showing up
   uninvited. I expected position R^2 > 0.9 in the checkpoint; the honest
   revision is that ~0.75 is what beta=1 buys at this latent size, and the
   criterion that actually matters — velocity vs position gap — is intact.

2. *A probe can flatter a dead latent.* The collapsed base run scores 0.74
   on position — same as the healthy warmup run! With KL ~0 the posterior
   means barely move, but ridge regression rescales freely, so microscopic
   input-correlated wiggles read as signal. A probe number means nothing
   without KL and reconstruction next to it. This goes straight into the
   Step 5 writeup: report probe R^2, KL, and recon together, always.

## The baseline that Step 1 existed to produce

Velocity R^2 is ~0.00 in every run, healthy or collapsed. A single frame
contains no motion, so no single-frame encoder can know velocity, however
good its reconstructions. That zero is the number Steps 2 and 3 have to
beat — the whole point of the dynamics models.

Checkpoint verdict: passed, with the position expectation revised as
above. Canonical Step 1 model is the warmup5k run:

    uv run train-vae --run-name warmup5k --beta 1.0 --beta-warmup-steps 5000

Its checkpoint (runs/vae/warmup5k/ in the WSL clone) is the frozen encoder
Step 2 will build on.

Next: Step 2, the frozen-encoder GRU.
