"""Render a random rollout of the bouncing ball env.

    uv run ball-demo --gif docs/media/bouncing_ball.gif
    uv run ball-demo --show --random-actions
"""

import argparse

import jax
import matplotlib

from world_models.envs import EnvParams, collect_trajectory, random_nudge_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=199)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gif", type=str, default=None, help="save the animation here")
    parser.add_argument("--show", action="store_true", help="open a matplotlib window")
    parser.add_argument("--random-actions", action="store_true",
                        help="apply random nudges instead of zero actions")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    params = EnvParams()
    policy = random_nudge_policy() if args.random_actions else None
    traj = collect_trajectory(jax.random.PRNGKey(args.seed), policy, args.steps, params)

    frames = traj["obs"]
    T = frames.shape[0]

    fig, (ax_img, ax_vel) = plt.subplots(1, 2, figsize=(8, 4))
    im = ax_img.imshow(frames[0, :, :, 0], cmap="gray", vmin=0, vmax=1)
    ax_img.set_title(f"Observation ({params.img_h}x{params.img_w})")
    ax_img.axis("off")

    ts = range(T)
    ax_vel.plot(ts, traj["vx"], label="vx", alpha=0.8)
    ax_vel.plot(ts, traj["vy"], label="vy", alpha=0.8)
    vline = ax_vel.axvline(0, color="red", lw=1)
    ax_vel.set_xlabel("step")
    ax_vel.set_ylabel("velocity")
    ax_vel.legend(loc="upper right")
    ax_vel.set_title("Ground-truth velocity")

    def update(i):
        im.set_data(frames[i, :, :, 0])
        vline.set_xdata([i, i])
        return im, vline

    ani = animation.FuncAnimation(fig, update, frames=T, interval=50, blit=True)
    plt.tight_layout()

    out = args.gif
    if out is None and not args.show:
        out = "bouncing_ball.gif"
    if out:
        ani.save(out, writer="pillow", fps=20)
        print(f"saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
