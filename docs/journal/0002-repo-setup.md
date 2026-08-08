# 0002 - Repo setup, two env bugs

2026-08-08

Two months since the last entry, so before touching models: turn the pile
of loose files into an actual project.

What changed:

- Proper uv project with a src layout: the env now lives in
  `src/world_models/envs/`, the render script became a `ball-demo` entry
  point, and there's a pytest suite. `uv sync` and everything works.
- `docs/design/` for design notes (one per component) and `docs/journal/`
  for this log. The journal is deliberately the raw material for the
  tutorial: decisions and failures get written down when they happen, not
  reconstructed later.
- First real commits. The plan is one commit per coherent chunk so the
  history itself tells the story of the build.

Moving the env into the package, I found two bugs:

1. `reset` drew x and y with the *same* PRNG key. Same key, same uniform
   sample, so with a square image every single episode spawned on the
   diagonal x == y. Velocity was random, so trajectories decorrelated after
   a bounce or two, but the spawn distribution was a line instead of a
   square, and that's exactly the kind of skew a latent probe would quietly
   absorb. Classic JAX lesson: every random draw gets its own split, no
   exceptions. There's a regression test now that vmaps reset over 64 keys
   and checks x and y aren't glued together.

2. `collect_trajectory` accepted a `policy_fn` argument and then ignored
   it, always rolling out zero actions. Harmless for the VAE, fatal for
   Steps 2-3: dynamics data with constant actions means the
   action-conditioned part of the model has literally nothing to learn.
   Fixed, added `random_nudge_policy`, and a test asserts the recorded
   actions are actually nonzero.

Both bugs are going in the tutorial. "The env is the easy part" is exactly
when this stuff sneaks in, and both would have silently poisoned the
comparison the whole project is built around.

Next: Step 1, the VAE and the velocity-probe baseline.
