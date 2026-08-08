# Bouncing ball environment

`src/world_models/envs/bouncing_ball.py`

## What it is

A ball in a box. State is (x, y, vx, vy) in continuous coordinates, the
action is a small (ax, ay) nudge added to the velocity, and the observation
is a 32x32 grayscale image with a Gaussian blob centred on the ball. Physics
is: integrate position, reflect position and velocity at the walls, clamp
speed.

## Design decisions

**gymnax-shaped API.** `reset(key, params) -> (obs, state)` and
`step(key, state, action, params) -> (obs, state, reward, done, info)`, all
pure functions, params as a static NamedTuple. This means the training code
never knows it is talking to a toy: swapping in gymnax MinAtar later (or a
thin wrapper around ViZDoom) does not touch the model code.

**Ground truth in `info`.** Every step returns the true (x, y, vx, vy).
This is the env's reason to exist: latents get probed against these values,
so they ride along with every trajectory instead of being reconstructed
after the fact.

**Gaussian blob rendering, not a hard circle.** The blob gives smooth
pixel gradients, which makes the reconstruction loss well behaved, and it
encodes sub-pixel position: a half-pixel move visibly shifts intensity.
With a hard-edged sprite, position information would be quantized to the
pixel grid.

**Reflection instead of clamping.** Velocity flips sign at the wall and the
overshoot reflects back, so energy is conserved and the dynamics stay
interesting. One floor plus one ceiling reflection per axis is enough
because max_speed is far below the box size; that assumption is documented
in the code and would need revisiting if the speed cap ever gets close to
the box dimensions.

**Speed clamp.** Actions keep injecting energy, so without a cap the ball
accelerates forever. The clamp rescales the velocity vector (preserving
direction) rather than clipping components.

## Data collection

`collect_trajectory(key, policy_fn, n_steps, params)` rolls the env with
`lax.scan` and returns stacked arrays: obs, action, and the true state per
frame, with the reset frame prepended (zero action). `policy_fn` is a
`(key, state) -> action` callable; `random_nudge_policy(scale)` is the one
to use for training data, since dynamics data collected with zero actions
would give the action-conditioned part of the model nothing to learn.

The policy is a static jit argument, so build it once and reuse it; a fresh
closure per call means a recompile per call.

## Knobs

All in `EnvParams`: image size, blob sigma, max speed, action clamp, dt,
episode length. Defaults (32x32, sigma 2, max_speed 2) are tuned so the
ball crosses the box in ~16 steps and a typical episode contains a dozen
bounces, which keeps sequence models honest at modest sequence lengths.

## Known limitations

- No reward. Fine until Step 4; the reward head can train on a synthetic
  target (e.g. distance to a goal region) when the time comes.
- `done` only fires on the step cap, there is no failure state.
- Single ball, static background. That is intentional for now; distractor
  backgrounds are a later experiment about reconstruction-free models.
