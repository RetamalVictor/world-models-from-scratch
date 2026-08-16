"""Big-sample evaluation of a finished train-doom run.

    uv run python scripts/doom_eval.py --run runs/doom/take-cover \
        --episodes 30

The training loop's real_eval quotes 3 episodes, and take_cover
survival is noisy enough that 3-episode means swing by more than the
gap between policies. This replays the same greedy evaluation path,
same env construction, same RSSM filter, same argmax actor, at a
sample size worth quoting, adds a matched random baseline, and keeps
every per-episode number in the JSON so the claim can be rechecked
without rerunning anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from world_models.collect import RandomPolicy, RSSMPolicy, collect_episode
from world_models.envs.doom import DoomTakeCover
from world_models.models.actor_critic import DiscreteActor
from world_models.models.rssm import load_rssm
from world_models.train_online import K_EVAL, lane_key

ARMS = ("random", "actor", "actor_refit")


def summarize(steps: list[int], rewards: list[float],
              action_repeat: int) -> dict:
    s = np.asarray(steps, np.float64)
    return {
        "mean_steps": float(s.mean()),
        "std_steps": float(s.std()),
        "median_steps": float(np.median(s)),
        "min_steps": int(s.min()),
        "max_steps": int(s.max()),
        "mean_tics": float(s.mean() * action_repeat),
        "mean_reward": float(np.mean(rewards)),
        "steps": [int(x) for x in steps],
        "rewards": [float(r) for r in rewards],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Big-sample evaluation of a finished train-doom run")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--arms", type=str, default=",".join(ARMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run
    cfg = json.loads((run_dir / "config.json").read_text())
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(
                f"unknown arm {arm!r}; pick from {', '.join(ARMS)}")
    if "actor_refit" in arms and not (run_dir / "actor_refit.msgpack").exists():
        print("actor_refit.msgpack not found (final_ac_steps was 0?); "
              "skipping that arm")
        arms = [a for a in arms if a != "actor_refit"]
    if not arms:
        raise SystemExit("no arms left to evaluate")

    # The env is seeded with this script's seed, not the run's: the
    # point is an evaluation independent of the episodes training saw.
    env = DoomTakeCover(frame_size=cfg["frame_size"],
                        action_repeat=cfg["action_repeat"],
                        seed=args.seed)
    max_steps = cfg["max_episode_steps"]
    root_key = jax.random.PRNGKey(args.seed)

    greedy_policy = None
    actor_params = {}
    actor_arms = [a for a in arms if a != "random"]
    if actor_arms:
        model, wm_params = load_rssm(run_dir)
        actor = DiscreteActor(action_dim=cfg["action_dim"])
        s_dim = cfg["hidden"] + cfg["latent_dim"]
        template = actor.init(jax.random.PRNGKey(0), jnp.zeros((1, s_dim)))
        for arm in actor_arms:
            actor_params[arm] = serialization.from_bytes(
                template, (run_dir / f"{arm}.msgpack").read_bytes())
        # train_doom's greedy path verbatim: argmax actor, key unused,
        # the RSSMPolicy filtering the belief from raw frames.
        greedy_fn = jax.jit(lambda p, st: actor.apply(
            p, st[None], None, method=DiscreteActor.act)[0])
        greedy_policy = RSSMPolicy(
            model, lambda p, st, k: greedy_fn(p, st), root_key,
            cfg["action_dim"])

    results = {}
    for arm in arms:
        if arm != "random":
            greedy_policy.set_params(wm_params, actor_params[arm])
        steps, rewards = [], []
        for i in range(args.episodes):
            if arm == "random":
                policy = RandomPolicy(env.action_dim,
                                      seed=args.seed * 1_000_003 + i)
            else:
                greedy_policy.reseed(lane_key(root_key, K_EVAL, i))
                policy = greedy_policy
            ep = collect_episode(env, policy, max_steps)
            steps.append(ep["obs"].shape[0] - 1)
            rewards.append(float(ep["reward"][1:].sum()))
        results[arm] = summarize(steps, rewards, cfg["action_repeat"])
        r = results[arm]
        print(f"{arm:12s} {r['mean_steps']:7.1f} +/- {r['std_steps']:6.1f} "
              f"steps  {r['mean_tics']:8.1f} tics  (n={args.episodes})")
    env.close()

    if "random" in results:
        ratios = [
            f"{arm}/random {results[arm]['mean_steps'] / results['random']['mean_steps']:.2f}x"
            for arm in arms if arm != "random"
        ]
        if ratios:
            print("  ".join(ratios))

    out = args.out or run_dir / "eval_big.json"
    out.write_text(json.dumps({
        "run": str(run_dir),
        "episodes": args.episodes,
        "seed": args.seed,
        "action_repeat": cfg["action_repeat"],
        "arms": results,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
