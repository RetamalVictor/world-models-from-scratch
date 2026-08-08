"""Open-loop rollout: the shared evaluation protocol for Steps 2 and 3.

A dynamics model is handed `W` real transitions to warm up, then must run
on its own mean predictions. Every model implements one
`step_fn(carry, z_in, action) -> (carry, mu)` and this module does the
rest, so drift numbers stay comparable across architectures by
construction.

Index conventions (matching the dataset layout, where action[t] is the
action of the transition t-1 -> t):

    step i of the warm-up consumes (z_i, a_{i+1}) and predicts frame i+1,
    so after consuming (z_{W-1}, a_W) the model has already produced its
    horizon-1 prediction: frame W. The free-running phase continues from
    there. Predictions returned cover frames W .. W+K-1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


def open_loop_predict(step_fn, carry, z_context, a_context, a_future):
    """Warm up teacher-forced, then roll on own mean predictions.

    z_context: (W, B, D) true latents for frames 0 .. W-1
    a_context: (W, B, A) actions 1 .. W (one per warm-up transition)
    a_future:  (K-1, B, A) actions W+1 .. W+K-1
    Returns (K, B, D): predicted latents for frames W .. W+K-1, where
    the first entry is the one-step prediction made from real history.
    """
    def forced(carry_and_z, inp):
        carry, _ = carry_and_z
        z_in, a = inp
        carry, mu = step_fn(carry, z_in, a)
        return (carry, mu), None

    (carry, z_first), _ = jax.lax.scan(
        forced, (carry, z_context[0]), (z_context, a_context)
    )

    def free(carry_and_z, a):
        carry, z_in = carry_and_z
        carry, mu = step_fn(carry, z_in, a)
        return (carry, mu), mu

    _, rest = jax.lax.scan(free, (carry, z_first), a_future)
    return jnp.concatenate([z_first[None], rest], axis=0)


def pixel_mse_per_horizon(pred_frames, true_frames):
    """Mean squared error per rollout step. Inputs (K, B, H, W, C) -> (K,)."""
    return jnp.mean((pred_frames - true_frames) ** 2, axis=(1, 2, 3, 4))


def side_by_side_gif(frames_a, frames_b, out_path, scale: int = 4,
                     duration_ms: int = 50):
    """Two aligned (T, H, W) float arrays in [0,1] -> one gif, A | B."""
    divider = np.full((frames_a.shape[1], 2), 1.0)
    imgs = []
    for t in range(frames_a.shape[0]):
        strip = np.concatenate([frames_a[t], divider, frames_b[t]], axis=1)
        img = Image.fromarray((np.clip(strip, 0, 1) * 255).astype(np.uint8),
                              mode="L")
        img = img.resize((img.width * scale, img.height * scale),
                         Image.NEAREST)
        imgs.append(img)
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=duration_ms, loop=0)
