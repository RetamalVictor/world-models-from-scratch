"""Train an actor-critic purely in the RSSM's imagination (Step 4).

    uv run train-ac --wm-run runs/rssm/hover --data data/ball_hover.npz

Phase 2 of the offline Dreamer recipe: the world model is frozen, the
policy never sees a real transition during training, and the final eval
runs the actor in the actual (pure JAX) environment against zero-action
and random-nudge baselines.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from flax import serialization
from flax.training import train_state

matplotlib.use("Agg")

from world_models import data as data_lib
from world_models import rollout as rollout_lib
from world_models.envs import BallGoalEnv, GoalEnvParams
from world_models.models.actor_critic import Actor, Critic, lambda_returns
from world_models.models.rssm import RSSM, load_rssm
from world_models.tracking import Tracker, metrics_figure
from world_models.train_rssm import filter_episodes, sample_pixel_batch


@dataclass(frozen=True)
class Config:
    seed: int = 0
    data_path: str = "data/ball_hover.npz"
    wm_run: str = "runs/rssm/hover"
    goal_speed: float = 0.0          # must match the dataset's task
    horizon: int = 15
    gamma: float = 0.95
    lam: float = 0.95
    ent_coef: float = 1e-4
    lr: float = 3e-4
    grad_clip: float = 100.0
    seq_batch: int = 16              # sequences filtered per step
    seq_len: int = 16                # transitions per sequence
    steps: int = 5_000
    log_every: int = 50
    eval_every: int = 1_000
    eval_episodes: int = 20
    final_eval_episodes: int = 100
    run_name: str = "hover"
    run_root: str = "runs/ac"
    wandb_project: str | None = None


def make_imagination(wm: RSSM, wm_params, actor: Actor, critic: Critic,
                     config: Config):
    max_action = GoalEnvParams().max_nudge

    def imagine(actor_params, h0, z0, key):
        """Roll the frozen prior with the actor for H steps.

        h0, z0: (S, ·) start states from real posterior filtering.
        Returns states (H+1, S, dim) with s_0 first, rewards (H, S), and
        a mean entropy proxy for the bonus.
        """
        s0 = jnp.concatenate([h0, z0], axis=-1)
        k_act, k_prior = jax.random.split(key)
        act_noise = jax.random.normal(
            k_act, (config.horizon,) + h0.shape[:1] + (wm.action_dim,))
        prior_noise = jax.random.normal(
            k_prior, (config.horizon,) + z0.shape)

        def step(carry, xs):
            h, z = carry
            an, pn = xs
            s = jnp.concatenate([h, z], axis=-1)
            mu, sigma = actor.apply(actor_params, s)
            a = max_action * jnp.tanh(mu + sigma * an)
            h = wm.apply(wm_params, h, z, a, method=RSSM.core_step)
            mu_p, sig_p = wm.apply(wm_params, h, method=RSSM.prior_dist)
            z = mu_p + sig_p * pn
            r = wm.apply(wm_params, h, z, method=RSSM.reward)
            ent = jnp.log(sigma).sum(axis=-1).mean()
            return (h, z), (jnp.concatenate([h, z], axis=-1), r, ent)

        _, (states, rewards, ents) = jax.lax.scan(
            step, (h0, z0), (act_noise, prior_noise))
        states = jnp.concatenate([s0[None], states], axis=0)
        return states, rewards, ents.mean()

    def actor_loss_fn(actor_params, critic_params, h0, z0, key):
        states, rewards, entropy = imagine(actor_params, h0, z0, key)
        values = critic.apply(critic_params, states)          # (H+1, S)
        returns = lambda_returns(rewards, values[1:], config.gamma,
                                 config.lam)
        loss = -returns.mean() - config.ent_coef * entropy
        return loss, (states, rewards, returns)

    def critic_loss_fn(critic_params, states_sg, rewards_sg):
        values = critic.apply(critic_params, states_sg)
        returns = lambda_returns(rewards_sg, values[1:], config.gamma,
                                 config.lam)
        target = jax.lax.stop_gradient(returns)
        return ((values[:-1] - target) ** 2).mean()

    return actor_loss_fn, critic_loss_fn


def imagine_frames(wm: RSSM, wm_params, actor: Actor, actor_params,
                   h, z, horizon: int):
    """Deterministic actor-driven prior rollout, decoded to frames."""
    def step(carry, _):
        h, z = carry
        s = jnp.concatenate([h, z], axis=-1)
        mu, _ = actor.apply(actor_params, s)
        a = actor.max_action * jnp.tanh(mu)
        h = wm.apply(wm_params, h, z, a, method=RSSM.core_step)
        mu_p, _ = wm.apply(wm_params, h, method=RSSM.prior_dist)
        return (h, mu_p), (h, mu_p)

    _, (hs, zs) = jax.lax.scan(step, (h, z), None, length=horizon)
    frames = wm.apply(wm_params, hs.reshape(-1, hs.shape[-1]),
                      zs.reshape(-1, zs.shape[-1]), method=RSSM.decode)
    return frames.reshape(horizon, -1, *frames.shape[1:])


def make_real_eval(wm: RSSM, wm_params, actor: Actor, env_params, n_steps=200):
    """Jitted real-env rollouts: the actor acts from the RSSM's filtered
    state, updated online from pixels. Returns mean per-step reward."""

    def episode(key, actor_params, policy: str):
        k_reset, k_steps = jax.random.split(key)
        obs, env_state = BallGoalEnv.reset(k_reset, env_params)
        e = wm.apply(wm_params, obs[None], method=RSSM.encode)
        h = wm.initial_state(1)
        z, _ = wm.apply(wm_params, h, e, method=RSSM.post_dist)

        def step(carry, k):
            env_state, h, z = carry
            s = jnp.concatenate([h, z], axis=-1)
            if policy == "actor":
                mu, _ = actor.apply(actor_params, s)
                a = actor.max_action * jnp.tanh(mu)[0]
            elif policy == "random":
                a = jax.random.uniform(
                    jax.random.fold_in(k, 1), (2,),
                    minval=-env_params.max_nudge,
                    maxval=env_params.max_nudge)
            else:
                a = jnp.zeros(2)
            obs, env_state, reward, done, info = BallGoalEnv.step(
                k, env_state, a, env_params)
            h_new = wm.apply(wm_params, h, z, a[None], method=RSSM.core_step)
            e = wm.apply(wm_params, obs[None], method=RSSM.encode)
            z_new, _ = wm.apply(wm_params, h_new, e, method=RSSM.post_dist)
            return (env_state, h_new, z_new), (reward, obs, a)

        keys = jax.random.split(k_steps, n_steps)
        _, (rewards, frames, acts) = jax.lax.scan(
            step, (env_state, h, z), keys)
        return rewards.mean(), frames, acts

    def batch_eval(keys, actor_params, policy: str):
        returns, _, _ = jax.vmap(
            lambda k: episode(k, actor_params, policy))(keys)
        return returns

    evals = {
        p: jax.jit(lambda ks, ap, _p=p: batch_eval(ks, ap, _p))
        for p in ("actor", "zero", "random")
    }
    return evals, episode


def train(config: Config) -> dict:
    run_dir = Path(config.run_root) / config.run_name
    if run_dir.exists():
        raise SystemExit(
            f"{run_dir} already exists; pick another --run-name or delete it"
        )

    wm, wm_params = load_rssm(config.wm_run)
    tracker = Tracker(
        run_dir,
        {
            **dataclasses.asdict(config),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
        },
        wandb_project=config.wandb_project,
        run_name=config.run_name,
    )

    dataset = data_lib.load(config.data_path)
    train_split, _, _ = data_lib.splits(dataset["obs"].shape[0])
    obs, actions = dataset["obs"], dataset["action"]

    actor = Actor(action_dim=wm.action_dim,
                  max_action=GoalEnvParams().max_nudge)
    critic = Critic()
    root_key = jax.random.PRNGKey(config.seed)
    ka, kc = jax.random.split(root_key)
    s_dim = wm.hidden + wm.latent_dim
    actor_params = actor.init(ka, jnp.zeros((1, s_dim)))
    critic_params = critic.init(kc, jnp.zeros((1, s_dim)))
    actor_state = train_state.TrainState.create(
        apply_fn=actor.apply, params=actor_params,
        tx=optax.chain(optax.clip_by_global_norm(config.grad_clip),
                       optax.adam(config.lr)),
    )
    critic_state = train_state.TrainState.create(
        apply_fn=critic.apply, params=critic_params,
        tx=optax.chain(optax.clip_by_global_norm(config.grad_clip),
                       optax.adam(config.lr)),
    )

    actor_loss_fn, critic_loss_fn = make_imagination(
        wm, wm_params, actor, critic, config)

    @jax.jit
    def train_step(actor_state, critic_state, obs_seq, act_seq, key):
        h_seq, z_seq = filter_episodes(wm, wm_params, obs_seq, act_seq)
        h0 = h_seq.reshape(-1, h_seq.shape[-1])
        z0 = z_seq.reshape(-1, z_seq.shape[-1])

        grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
        (a_loss, (states, rewards, returns)), a_grads = grad_fn(
            actor_state.params, critic_state.params, h0, z0, key)
        actor_state = actor_state.apply_gradients(grads=a_grads)

        states_sg = jax.lax.stop_gradient(states)
        rewards_sg = jax.lax.stop_gradient(rewards)
        c_loss, c_grads = jax.value_and_grad(critic_loss_fn)(
            critic_state.params, states_sg, rewards_sg)
        critic_state = critic_state.apply_gradients(grads=c_grads)

        return (actor_state, critic_state, a_loss, c_loss,
                rewards.mean(), returns.mean())

    env_params = GoalEnvParams(goal_speed=config.goal_speed)
    evals, episode_fn = make_real_eval(wm, wm_params, actor, env_params)

    rng = np.random.default_rng(config.seed)
    for step in range(1, config.steps + 1):
        obs_seq, act_seq, _ = sample_pixel_batch(
            rng, obs, actions, None, train_split, config.seq_len,
            config.seq_batch,
        )
        step_key = jax.random.fold_in(root_key, step)
        (actor_state, critic_state, a_loss, c_loss, imag_reward,
         imag_return) = train_step(actor_state, critic_state, obs_seq,
                                   act_seq, step_key)
        if step % config.log_every == 0:
            tracker.log(step, actor_loss=a_loss, critic_loss=c_loss,
                        imag_reward=imag_reward, imag_return=imag_return)
        if step % config.eval_every == 0:
            keys = jax.random.split(
                jax.random.fold_in(root_key, step + 10_000_000),
                config.eval_episodes)
            real = evals["actor"](keys, actor_state.params)
            tracker.log(step, real_reward=real.mean())
            print(f"step {step:6d}  actor {a_loss:8.3f}  critic {c_loss:7.4f}  "
                  f"imag reward {imag_reward:6.3f}  real reward {real.mean():6.3f}")

    (run_dir / "actor.msgpack").write_bytes(
        serialization.to_bytes(actor_state.params))
    (run_dir / "critic.msgpack").write_bytes(
        serialization.to_bytes(critic_state.params))

    # Final eval: actor vs baselines, fresh episodes.
    final = {}
    eval_keys = jax.random.split(jax.random.PRNGKey(config.seed + 99),
                                 config.final_eval_episodes)
    for policy in ("actor", "zero", "random"):
        returns = np.asarray(evals[policy](eval_keys, actor_state.params))
        final[policy] = {"mean_reward": float(returns.mean()),
                         "std": float(returns.std())}
    tracker.log_json("eval", final)

    # Gifs: the actor working in the real env, and real-vs-imagined from
    # the same start (the model's imagination of its own plan, decoded).
    _, frames, acts = episode_fn(jax.random.PRNGKey(config.seed + 123),
                                 actor_state.params, "actor")
    frames, acts = np.asarray(frames), np.asarray(acts)
    rollout_lib.single_gif(frames[:150], run_dir / "real_rollout.gif")

    fobs = jnp.asarray(frames[:5])[:, None]               # (5, 1, H, W, C)
    fact = jnp.asarray(acts[:5])[:, None]
    h_seq, z_seq = filter_episodes(wm, wm_params, fobs, fact)
    imag = imagine_frames(wm, wm_params, actor, actor_state.params,
                          h_seq[-1], z_seq[-1], 30)
    rollout_lib.side_by_side_gif(frames[5:35], np.asarray(imag)[:, 0],
                                 run_dir / "imagination.gif")
    tracker.log_figure("loss_curves", metrics_figure(run_dir))
    tracker.finish()

    print("final eval:", json.dumps(final, indent=2))
    return final


def main():
    parser = argparse.ArgumentParser(
        description="Train the Step 4 actor-critic in imagination")
    defaults = Config()
    parser.add_argument("--wm-run", type=str, default=defaults.wm_run)
    parser.add_argument("--data", type=str, default=defaults.data_path)
    parser.add_argument("--goal-speed", type=float,
                        default=defaults.goal_speed)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--run-name", type=str, default=defaults.run_name)
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()
    config = Config(
        seed=args.seed,
        data_path=args.data,
        wm_run=args.wm_run,
        goal_speed=args.goal_speed,
        steps=args.steps,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
    )
    train(config)


if __name__ == "__main__":
    main()
