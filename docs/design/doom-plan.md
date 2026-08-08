# DOOM plan (Step 6)

The final rung reproduces the actual experiment from Ha & Schmidhuber's
World Models paper: their Doom agent trained on the take_cover scenario of
ViZDoom. Same task, our models.

## Why ViZDoom

It is maintained by the Farama Foundation, ships Gymnasium wrappers on all
platforms, exposes raw screen pixels (plus depth and object labels if ever
needed), and runs at thousands of fps on a single CPU thread. Install is
`pip install vizdoom`. The built-in scenario menu includes take_cover, which
is the canonical target here, with basic / defend_the_center /
health_gathering as easier warm-ups.

## The CPU/JAX boundary

There is no JAX-native Doom. ViZDoom is a C++ engine with a Python API, so
env steps run on CPU outside jit-land, and the vmap-thousands-of-envs trick
does not apply. This costs less than it sounds like: a world-model agent
touches the real env only to fill the replay buffer, and does its thousands
of updates in imagination. A slow env hurts model-free PPO far more than it
hurts Dreamer.

The boundary is explicit and small: collect rollouts through the Gymnasium
API into a numpy buffer, `jnp.asarray` at the buffer edge, everything from
there on is the same JAX training code as the ball. Because the ball env
already mimics the gymnax signature, the swap is a thin wrapper plus that
one array hop.

## Keeping the model unchanged

- Downscale the screen buffer to 64x64 grayscale, which is effectively what
  the 2018 paper did. The VAE/RSSM stay the same architecture with one more
  conv stage for the larger input.
- take_cover has two discrete actions (move left / move right), so the
  action channel stays trivial.
- Reward is survival time; the reward head can be deferred until control
  (Step 4 machinery) is ported over.

## Reproducibility when the data is on-policy

The ball dataset is a fixed file, byte-identical from a seed. Doom data
can't be: the replay buffer fills from a stateful C++ engine while the
policy is changing, so "same bytes" stops being a meaningful target.
The target becomes "same generating process, verified statistically":
seed everything that can be seeded, log the collection config, and write
a dataset report per collection round (episode length distribution,
action histogram, pixel stats) so two supposedly identical runs can be
compared. The stats report `make-data` writes for the ball is the same
idea in embryo.

## Order of operations

1. Optional middle rung if the jump feels big: gymnax MinAtar Breakout,
   still pure JAX, a real game, no CPU boundary yet.
2. ViZDoom wrapper with the ball env's API + replay buffer.
3. Retrain VAE and RSSM on take_cover frames; the probes are gone (no
   ground truth), which is exactly why the ball came first: hyperparameters
   and code arrive here already trusted.
4. Imagination training for the policy, then compare survival time against
   the numbers in the 2018 paper.
