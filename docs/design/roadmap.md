# Roadmap

Goal: build the two classic latent world models side by side, compare them
quantitatively on an env where the ground truth is known, then graduate to
DOOM. Six steps, each with a checkpoint. The rule is: don't move to the next
step until the checkpoint passes.

Why a toy env instead of a benchmark: the entire comparison hinges on being
able to ask "did the latent learn the dynamics?" With a custom simulator I
know the true (x, y, vx, vy) behind every frame, so I can fit a linear probe
from latents to velocity and answer that with an R^2 instead of a feeling.

## Step 0 - toy environment (done)

A ball in a box, position and velocity, bounces off walls, rendered to a
32x32 grayscale image. Optional action nudges the velocity. API mirrors
gymnax (`reset(key, params)`, `step(key, state, action, params)`) so a real
env can be swapped in later with minimal changes.

Checkpoint: animate a random rollout, confirm it looks like a bouncing ball.
Keep the true (vx, vy) around as the probe target.

## Step 1 - the VAE (the "V", shared by both models)

Standard conv encoder to a Gaussian latent, deconv decoder, ELBO loss.

Checkpoint: reconstructions of held-out frames look right, and samples from
the prior decode to plausible ball images. Then the baseline probe: freeze
the encoder, fit a linear regressor from z to the true (vx, vy), record R^2.
A single-frame VAE cannot know velocity (one frame has no motion), so this
number should be near zero. That failure is the baseline the dynamics models
get measured against, and the first lesson of the tutorial: reconstruction
alone does not buy you dynamics.

## Step 2 - Ha-style: frozen V + separate dynamics

Freeze the Step 1 encoder. Encode trajectories to latent sequences, then
train a GRU with a Gaussian head to predict the next latent from (z, a).
Two separate training phases, the encoder never sees the dynamics loss.
This is the World Models (2018) recipe minus the CMA-ES controller, which
I skip: the comparison is between world models, not controllers.

Checkpoint: open-loop rollout. Encode one real frame, run the GRU forward K
steps on its own predictions, decode each predicted latent. Watch it drift.

## Step 3 - Dreamer-style: joint RSSM

One model trained end to end. Deterministic recurrent state h, stochastic
latent z, and the predict/correct split: a prior p(z_t | h_t) that must
predict without seeing the frame, and a posterior q(z_t | h_t, o_t) that
gets to look. The KL between them ties prediction to correction and is
where the dynamics get learned. Encoder, recurrence, prior, posterior and
decoder all train together.

Implementation notes that save hours: the recurrence is a `jax.lax.scan`
with carry (h, z), and KL balancing (~0.8 toward the prior) is what keeps
training stable. Skip symlog / two-hot / free bits until something breaks.

Checkpoint: the same open-loop rollout as Step 2, and the same velocity
probe. Comparing probe R^2 between the frozen latent and the RSSM latent is
the headline number of the whole tutorial.

## Step 4 - imagination + actor-critic (optional)

Actor and critic trained purely on imagined rollouts: start from real
posterior states, run the prior open loop, never touch pixels. Only needed
if I want control; Steps 0-3 settle the conceptual comparison on their own.

Checkpoint: the actor improves task reward while training only in
imagination.

## Step 5 - comparison writeup

Three plots, each tied to a concept: open-loop drift vs horizon (Ha vs
Dreamer), the probe R^2 bar chart, and the KL(post || prior) curve over
training. Plus sample efficiency if Step 4 happened.

## Step 6 - DOOM

Swap the toy env for ViZDoom's take_cover scenario, which is literally the
task from the 2018 paper. Details in [doom-plan.md](doom-plan.md).
