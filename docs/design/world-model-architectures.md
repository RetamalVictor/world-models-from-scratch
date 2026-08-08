# World model architectures (Steps 1-3)

Design notes for the three models. The point of the exercise is the
comparison in the last section, so the three share data, encoder size, and
evaluation protocol wherever possible.

## Shared conventions

- Data: trajectories from `collect_trajectory` with `random_nudge_policy`,
  so actions carry information. Something like 1000 episodes x 200 steps at
  32x32 is a few hundred MB as float32 and fits in memory; store as stacked
  arrays, sample subsequences for training.
- Held-out split at the episode level, not the frame level, or the probe
  numbers are contaminated by near-duplicate frames.
- Everything in Flax linen + optax + distrax. The recurrences are
  `jax.lax.scan`; imagination later is the same scan with the posterior
  swapped for the prior.
- Fixed seeds everywhere, and every experiment records its config. Boring,
  but this is the part of "good practices" that actually matters for a
  tutorial: every number in the writeup should be reproducible from a seed
  and a config.

## Step 1: VAE

Small conv encoder to a diagonal Gaussian q(z|o), deconv decoder back to the
image. For a 32x32x1 input, 3-4 conv layers into a latent of ~16 dims is
plenty; the ball has 4 true degrees of freedom, and an over-wide latent just
makes the probe look better than the model deserves.

Loss is the ELBO:

    L = recon(o_hat, o) + beta * KL(q(z|o) || N(0, I))

Recon as Gaussian NLL with fixed variance (i.e. MSE) is fine for this env.
Start with beta = 1; if the KL collapses to zero (posterior equals prior,
latent unused), lower beta before reaching for free bits.

Checkpoint: good held-out reconstructions, plausible samples from the prior.
Then the baseline probe (protocol below). Expected result: position R^2 high,
velocity R^2 near zero, because one frame contains no motion. This number is
the punchline the later models get compared against.

## Step 2: Ha-style (frozen V + MDN-RNN, minus the MDN at first)

Two phases, deliberately separate:

1. Train the Step 1 VAE, freeze it.
2. Encode all trajectories to latent sequences z_{1:T} (use the posterior
   mean, not samples, for the dataset). Train a GRU that takes (z_{t-1},
   a_{t-1}) and predicts a Gaussian over z_t:

       h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
       mu_t, sigma_t = head(h_t)
       L = -log N(z_t | mu_t, sigma_t)

Start with a single Gaussian head. The full MDN (mixture) only matters when
the next-latent distribution is multimodal, which for a ball in a box it
mostly isn't; add mixture components later as an ablation if the single
Gaussian visibly underfits at wall bounces (the one place multimodality
could show up).

The encoder is never updated in phase 2. That is the defining property of
this model and the source of its weakness: the latent was shaped by
reconstruction only, and the dynamics model has to make do with whatever
information reconstruction happened to preserve.

Checkpoint: open-loop rollout. Encode one real frame, roll the GRU forward
K steps feeding its own mean predictions, decode each through the frozen
decoder. Log per-step pixel error vs a real continuation, and save the
decoded filmstrip. Expect visible drift and blur.

## Step 3: Dreamer-style RSSM

One model, trained jointly. Recurrent deterministic state h_t plus
stochastic latent z_t, with the predict/correct pair:

    h_t   = f(h_{t-1}, z_{t-1}, a_{t-1})        # deterministic core
    prior = p(z_t | h_t)                        # predict, no observation
    post  = q(z_t | h_t, enc(o_t))              # correct, observation enters
    z_t   ~ post                                # posterior sample carries forward
    o_hat = dec(h_t, z_t)

    L = recon(o_hat, o_t) + beta * KL(post || prior)

summed over the sequence, all parameters trained together. The KL is the
mechanism: the prior is forced to predict what the posterior will conclude
after seeing the frame, so dynamics information gets pushed into h and z.

Implementation notes:

- The whole per-sequence computation is one `lax.scan` with carry (h, z).
- KL balancing: apply the KL gradient asymmetrically, ~0.8 toward the prior
  and ~0.2 toward the posterior (stop-gradient trick on each side). Without
  it the posterior collapses toward a weak prior and training is unstable.
  This is the single most load-bearing trick in the whole build.
- Continuous (Gaussian) latents first. Categorical latents like DreamerV2
  are an ablation for later, not a starting point.
- Sizes: keep enc/dec identical to Step 1, h around 128, z around 16, so
  differences in the probe are attributable to training, not capacity.
- Skip symlog, two-hot, and free bits until a concrete failure calls for
  them; each one added should get a journal entry explaining what broke.

Checkpoint: same open-loop rollout as Step 2 (roll the prior, decode), same
probe protocol. Also log KL(post || prior) over training; it should fall
and stabilize, and its trajectory is one of the three headline plots.

## Evaluation protocol (shared)

The probe:

1. Freeze the model. Encode held-out episodes; collect latents per frame
   (Step 1: z. Step 2: z and optionally GRU h. Step 3: [h, z]).
2. Fit ridge regression from latent to true (vx, vy) on half the held-out
   episodes, report R^2 on the other half.
3. Same split and seeds for every model.

Comparison table at the end of Step 3, one row per model: velocity probe
R^2, position probe R^2, open-loop pixel error at K in {1, 5, 15, 30}.

Expected story: Step 1 knows position but not velocity. Step 2 recovers
some velocity in h (the GRU integrates motion) but is capped by what the
frozen latent kept. Step 3 wins on both velocity R^2 and drift, because
joint training shapes the latent for prediction, not just reconstruction.
If Step 3 does not win, something is wrong, and finding out what will teach
more than the success would have.
