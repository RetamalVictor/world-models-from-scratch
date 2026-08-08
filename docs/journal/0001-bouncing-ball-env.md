# 0001 - Bouncing ball environment

2026-06-05

Step 0 of the plan: the toy env. A ball bouncing in a box, rendered to
32x32 grayscale, with the true (x, y, vx, vy) exposed for the probes that
come later. Wrote it gymnax-style (pure functions, params NamedTuple, PRNG
keys threaded everywhere) so the model code won't care when a real env
replaces it.

Decisions that took a moment:

- Gaussian blob instead of a hard circle, so position is sub-pixel and the
  future reconstruction loss gets smooth gradients.
- Reflection at the walls rather than clamping, so velocity actually flips
  and the dynamics are worth modelling. Two reflections per axis per step
  is enough given the speed clamp.
- `collect_trajectory` as a `lax.scan` that returns the whole rollout as
  stacked arrays, ground truth included.

Checkpoint passed: animated a rollout, it looks like a bouncing ball, and
the velocity traces show clean sign flips at each bounce.

Next: the VAE.
