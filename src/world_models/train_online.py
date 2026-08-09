"""Online Dreamer on the ball (Phase A of the Doom plan).

    uv run train-online --run-name follow

The loop Steps 1-4 deliberately skipped: the current actor (sampling
from its own policy for exploration) collects episodes into a replay
buffer, the world model trains on buffer sequences, the actor-critic
trains in the model's imagination, repeat. Model, losses, and the eval
protocol are the Step 3/4 recipe unchanged, so the offline follow score
(0.88 mean reward) is the bar to match — with the data budget logged.

Every random draw is keyed by a persistent counter (episodes collected,
updates done), never by wall-clock position in the process, so a
killed-and-resumed run continues exactly the run it interrupted.
tests/test_online.py holds this to byte-identical checkpoints.

After the loop, the co-trained actor is chasing a world model that kept
moving underneath it and plateaus at 0.801 mean reward, well short of
the offline bar. A fresh actor-critic trained offline-style against
that same world model, now frozen, reaches 0.911 (beats the 0.879
offline number) — the lag is in the actor, not the model. The
`--final-ac-steps` flag (0 to disable) re-fits that fresh actor-critic
after the loop; it is deliberately not checkpointed, since it is cheap
to redo from the frozen model and buffer alone. Because it runs after
the loop, resuming a finished run with `--rounds` set to its completed
round count skips the loop and goes straight to the refit.
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

from world_models import rollout as rollout_lib
from world_models.checkpoint import Checkpointer
from world_models.envs import BallGoalEnv, GoalEnvParams
from world_models.envs.bouncing_ball_goal import _reward as goal_reward
from world_models.models.actor_critic import Actor, Critic, lambda_returns
from world_models.models.rssm import RSSM
from world_models.replay import ReplayBuffer
from world_models.tracking import Tracker, metrics_figure
from world_models.train_ac import imagine_frames
from world_models.train_rssm import Config as RSSMConfig
from world_models.train_rssm import filter_episodes, make_losses


@dataclass(frozen=True)
class Config:
    seed: int = 0
    goal_speed: float = 1.0
    episode_steps: int = 200
    # world model — Step 3 defaults
    latent_dim: int = 16
    hidden: int = 128
    min_sigma: float = 0.1
    alpha: float = 0.8
    beta: float = 1.0
    wm_lr: float = 1e-3
    wm_grad_clip: float = 1.0
    transitions: int = 32
    batch_size: int = 32
    # actor-critic — Step 4 defaults
    horizon: int = 15
    gamma: float = 0.95
    lam: float = 0.95
    ent_coef: float = 1e-4
    ac_lr: float = 3e-4
    ac_grad_clip: float = 100.0
    seq_batch: int = 16
    seq_len: int = 16
    # the online loop
    buffer_capacity: int = 60_000    # frames (~300 ball episodes)
    seed_episodes: int = 10          # random policy, before round 1
    rounds: int = 300
    episodes_per_round: int = 1
    wm_updates: int = 50             # per round
    ac_updates: int = 50             # per round
    final_ac_steps: int = 5000       # fresh actor-critic refit, 0 disables
    # bookkeeping
    log_every: int = 1               # rounds
    eval_every: int = 25             # rounds
    eval_episodes: int = 20
    final_eval_episodes: int = 100
    checkpoint_every: int = 25       # rounds
    keep_checkpoints: int = 3
    artifacts: bool = True
    resume: bool = False
    run_name: str = "follow"
    run_root: str = "runs/online"
    wandb_project: str | None = None


# Disjoint fold_in lanes so no counter can collide with another's keys.
K_EPISODE, K_WM, K_AC, K_EVAL, K_REFIT = 1, 2, 3, 4, 5


def lane_key(root: jax.Array, lane: int, counter: int) -> jax.Array:
    return jax.random.fold_in(jax.random.fold_in(root, lane), counter)


def make_runners(wm: RSSM, actor: Actor, env_params: GoalEnvParams,
                 n_steps: int):
    """Jitted batched env episodes, one per policy, with the RSSM
    filtering online from pixels (the agent never sees env state).

    Policies: explore (sampled actor), actor (deterministic), random,
    zero. Each runner maps a (B,) key array to the dataset-convention
    episode dict — obs/action/reward of length T+1, reset frame first
    with a zero action and its computed reward.
    """

    def episode(key, wm_params, actor_params, policy):
        k_reset, k_steps = jax.random.split(key)
        obs0, env_state = BallGoalEnv.reset(k_reset, env_params)
        e = wm.apply(wm_params, obs0[None], method=RSSM.encode)
        h = wm.initial_state(1)
        z, _ = wm.apply(wm_params, h, e, method=RSSM.post_dist)

        def step(carry, k):
            env_state, h, z = carry
            k_act, k_env = jax.random.split(k)
            s = jnp.concatenate([h, z], axis=-1)
            if policy == "explore":
                a = actor.apply(actor_params, s, k_act, method=Actor.act)[0]
            elif policy == "actor":
                a = actor.apply(actor_params, s, None, method=Actor.act)[0]
            elif policy == "random":
                a = jax.random.uniform(
                    k_act, (2,), minval=-env_params.max_nudge,
                    maxval=env_params.max_nudge)
            else:
                a = jnp.zeros(2)
            obs, env_state, reward, done, info = BallGoalEnv.step(
                k_env, env_state, a, env_params)
            h = wm.apply(wm_params, h, z, a[None], method=RSSM.core_step)
            e = wm.apply(wm_params, obs[None], method=RSSM.encode)
            z, _ = wm.apply(wm_params, h, e, method=RSSM.post_dist)
            return (env_state, h, z), (obs, a, reward)

        keys = jax.random.split(k_steps, n_steps)
        _, (obs_seq, act_seq, rew_seq) = jax.lax.scan(
            step, (env_state, h, z), keys)
        # env_state still binds the reset state here; its reward is a pure
        # function of state, computed rather than zero-filled (the reward
        # head trains on frame 0 too).
        return {
            "obs": jnp.concatenate([obs0[None], obs_seq]),
            "action": jnp.concatenate([jnp.zeros((1, 2)), act_seq]),
            "reward": jnp.concatenate(
                [goal_reward(env_state, env_params)[None], rew_seq]),
        }

    def batch(keys, wm_params, actor_params, policy):
        return jax.vmap(
            lambda k: episode(k, wm_params, actor_params, policy))(keys)

    return {
        p: jax.jit(lambda keys, wp, ap, _p=p: batch(keys, wp, ap, _p))
        for p in ("explore", "actor", "random", "zero")
    }


def make_imagination(wm: RSSM, actor: Actor, critic: Critic, config: Config,
                     max_action: float):
    """Step 4's imagination losses, with wm_params as a call-time argument
    because online training changes the world model between updates."""

    def imagine(wm_params, actor_params, h0, z0, key):
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

    def actor_loss_fn(actor_params, wm_params, critic_params, h0, z0, key):
        states, rewards, entropy = imagine(wm_params, actor_params, h0, z0,
                                           key)
        values = critic.apply(critic_params, states)
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


def train(config: Config) -> dict:
    run_dir = Path(config.run_root) / config.run_name
    if config.resume:
        if not run_dir.exists():
            raise SystemExit(f"{run_dir} does not exist; nothing to resume")
    elif run_dir.exists():
        raise SystemExit(
            f"{run_dir} already exists; --resume it, or pick another "
            "--run-name or delete it"
        )

    env_params = GoalEnvParams(goal_speed=config.goal_speed)
    model = RSSM(
        latent_dim=config.latent_dim, action_dim=2, hidden=config.hidden,
        min_sigma=config.min_sigma, obs_channels=3,
    )
    actor = Actor(action_dim=2, max_action=env_params.max_nudge)
    critic = Critic()

    root_key = jax.random.PRNGKey(config.seed)
    k_wm, k_actor, k_critic = jax.random.split(root_key, 3)
    wm_params = model.init(
        k_wm, jnp.zeros((1, env_params.img_h, env_params.img_w, 3)),
        model.initial_state(1), jnp.zeros((1, config.latent_dim)),
        jnp.zeros((1, 2)),
    )
    s_dim = config.hidden + config.latent_dim
    wm_state = train_state.TrainState.create(
        apply_fn=model.apply, params=wm_params,
        tx=optax.chain(optax.clip_by_global_norm(config.wm_grad_clip),
                       optax.adam(config.wm_lr)),
    )
    actor_state = train_state.TrainState.create(
        apply_fn=actor.apply, params=actor.init(k_actor, jnp.zeros((1, s_dim))),
        tx=optax.chain(optax.clip_by_global_norm(config.ac_grad_clip),
                       optax.adam(config.ac_lr)),
    )
    critic_state = train_state.TrainState.create(
        apply_fn=critic.apply,
        params=critic.init(k_critic, jnp.zeros((1, s_dim))),
        tx=optax.chain(optax.clip_by_global_norm(config.ac_grad_clip),
                       optax.adam(config.ac_lr)),
    )

    counters = {"round": 0, "episode": 0, "wm_update": 0, "ac_update": 0}
    ckpt = Checkpointer(run_dir, keep=config.keep_checkpoints)
    buffer = ReplayBuffer(config.buffer_capacity)
    if config.resume:
        template = {"wm": wm_state, "actor": actor_state,
                    "critic": critic_state, "counters": counters}
        step0, restored = ckpt.restore(template)
        wm_state, actor_state = restored["wm"], restored["actor"]
        critic_state, counters = restored["critic"], dict(restored["counters"])
        buffer = ReplayBuffer.load(run_dir / "buffer.npz")
        print(f"resumed from round {counters['round']} "
              f"({buffer.frames_added} env frames collected so far)")

    tracker = Tracker(
        run_dir,
        {
            **dataclasses.asdict(config),
            "obs_channels": 3,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
        },
        wandb_project=config.wandb_project,
        run_name=config.run_name,
    )

    rssm_config = RSSMConfig(latent_dim=config.latent_dim, alpha=config.alpha)
    sequence_loss = make_losses(model, rssm_config, has_reward=True)
    actor_loss_fn, critic_loss_fn = make_imagination(
        model, actor, critic, config, env_params.max_nudge)

    @jax.jit
    def wm_step(state, obs_seq, act_seq, rew_seq, key):
        grad_fn = jax.value_and_grad(sequence_loss, has_aux=True)
        (loss, (recon, kl, rew)), grads = grad_fn(
            state.params, obs_seq, act_seq, rew_seq, key,
            jnp.float32(config.beta))
        return state.apply_gradients(grads=grads), loss, recon, kl, rew

    @jax.jit
    def ac_step(wm_params, actor_state, critic_state, obs_seq, act_seq, key):
        h_seq, z_seq = filter_episodes(model, wm_params, obs_seq, act_seq)
        h0 = h_seq.reshape(-1, h_seq.shape[-1])
        z0 = z_seq.reshape(-1, z_seq.shape[-1])
        grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
        (a_loss, (states, rewards, returns)), a_grads = grad_fn(
            actor_state.params, wm_params, critic_state.params, h0, z0, key)
        actor_state = actor_state.apply_gradients(grads=a_grads)
        states_sg = jax.lax.stop_gradient(states)
        rewards_sg = jax.lax.stop_gradient(rewards)
        c_loss, c_grads = jax.value_and_grad(critic_loss_fn)(
            critic_state.params, states_sg, rewards_sg)
        critic_state = critic_state.apply_gradients(grads=c_grads)
        return (actor_state, critic_state, a_loss, c_loss, rewards.mean(),
                returns.mean())

    runners = make_runners(model, actor, env_params, config.episode_steps)

    def collect_episode(policy: str) -> float:
        key = lane_key(root_key, K_EPISODE, counters["episode"])
        ep = jax.device_get(
            runners[policy](key[None], wm_state.params, actor_state.params))
        buffer.add_episode(
            np.round(ep["obs"][0] * 255.0).astype(np.uint8),
            ep["action"][0], ep["reward"][0])
        counters["episode"] += 1
        return float(ep["reward"][0, 1:].mean())

    def real_eval(n_episodes: int, key_counter: int, policy: str,
                 actor_params=None) -> np.ndarray:
        keys = jax.random.split(
            lane_key(root_key, K_EVAL, key_counter), n_episodes)
        params = actor_state.params if actor_params is None else actor_params
        ep = runners[policy](keys, wm_state.params, params)
        return np.asarray(ep["reward"][:, 1:].mean(axis=1))

    def save_checkpoint(metric: float | None):
        tree = {"wm": wm_state, "actor": actor_state, "critic": critic_state,
                "counters": dict(counters)}
        ckpt.save(counters["round"], tree, metric=metric)
        buffer.save(run_dir / "buffer.npz")

    while counters["episode"] < config.seed_episodes:
        collect_episode("random")

    for r in range(counters["round"] + 1, config.rounds + 1):
        collect_rewards = [collect_episode("explore")
                           for _ in range(config.episodes_per_round)]

        for _ in range(config.wm_updates):
            u = counters["wm_update"]
            batch_rng = np.random.default_rng([config.seed, K_WM, u])
            obs_seq, act_seq, rew_seq = buffer.sample_sequences(
                batch_rng, config.batch_size, config.transitions)
            wm_state, loss, recon, kl, rew_mse = wm_step(
                wm_state, obs_seq, act_seq, rew_seq,
                lane_key(root_key, K_WM, u))
            counters["wm_update"] += 1

        for _ in range(config.ac_updates):
            u = counters["ac_update"]
            batch_rng = np.random.default_rng([config.seed, K_AC, u])
            obs_seq, act_seq, _ = buffer.sample_sequences(
                batch_rng, config.seq_batch, config.seq_len)
            (actor_state, critic_state, a_loss, c_loss, imag_reward,
             imag_return) = ac_step(
                wm_state.params, actor_state, critic_state, obs_seq, act_seq,
                lane_key(root_key, K_AC, u))
            counters["ac_update"] += 1

        counters["round"] = r
        if r % config.log_every == 0:
            tracker.log(
                r, collect_reward=np.mean(collect_rewards), recon=recon,
                kl=kl, reward_mse=rew_mse, actor_loss=a_loss,
                critic_loss=c_loss, imag_reward=imag_reward,
                env_frames=buffer.frames_added,
            )
        real = None
        if r % config.eval_every == 0 or r == config.rounds:
            real = float(real_eval(config.eval_episodes, r, "actor").mean())
            tracker.log(r, real_reward=real)
            print(f"round {r:4d}  collect {np.mean(collect_rewards):6.3f}  "
                  f"recon {recon:8.3f}  kl {kl:6.3f}  "
                  f"imag {imag_reward:6.3f}  real {real:6.3f}  "
                  f"frames {buffer.frames_added}")
        if r % config.checkpoint_every == 0 or r == config.rounds:
            save_checkpoint(metric=real)

    # Final eval, fresh episodes, actor vs the Step 4 baselines.
    final = {"env_frames": buffer.frames_added}
    for policy in ("actor", "zero", "random"):
        returns = real_eval(config.final_eval_episodes,
                            config.rounds + 1, policy)
        final[policy] = {"mean_reward": float(returns.mean()),
                         "std": float(returns.std())}

    # Fresh actor-critic, re-fit offline-style against the frozen final
    # WM: the co-trained actor above lags a WM that kept moving under it.
    # Own K_REFIT lane and counter, untouched by the loop's counters, so
    # this is a pure function of (config, wm_state.params, buffer) — not
    # checkpointed, cheap enough to redo on demand.
    refit_actor_state = refit_critic_state = None
    if config.final_ac_steps > 0:
        k_actor_r, k_critic_r = jax.random.split(
            lane_key(root_key, K_REFIT, 0))
        refit_actor_state = train_state.TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(k_actor_r, jnp.zeros((1, s_dim))),
            tx=optax.chain(optax.clip_by_global_norm(config.ac_grad_clip),
                           optax.adam(config.ac_lr)),
        )
        refit_critic_state = train_state.TrainState.create(
            apply_fn=critic.apply,
            params=critic.init(k_critic_r, jnp.zeros((1, s_dim))),
            tx=optax.chain(optax.clip_by_global_norm(config.ac_grad_clip),
                           optax.adam(config.ac_lr)),
        )
        for u in range(1, config.final_ac_steps + 1):
            batch_rng = np.random.default_rng([config.seed, K_REFIT, u])
            obs_seq, act_seq, _ = buffer.sample_sequences(
                batch_rng, config.seq_batch, config.seq_len)
            (refit_actor_state, refit_critic_state, a_loss, c_loss,
             imag_reward, imag_return) = ac_step(
                wm_state.params, refit_actor_state, refit_critic_state,
                obs_seq, act_seq, lane_key(root_key, K_REFIT, u))
            if u % 500 == 0 or u == config.final_ac_steps:
                tracker.log(config.rounds + u, refit_actor_loss=a_loss,
                           refit_critic_loss=c_loss,
                           refit_imag_reward=imag_reward)

        returns = real_eval(config.final_eval_episodes, config.rounds + 1,
                            "actor", actor_params=refit_actor_state.params)
        final["actor_refit"] = {"mean_reward": float(returns.mean()),
                                "std": float(returns.std())}

    tracker.log_json("eval", final)

    if config.artifacts:
        # wm params in the load_rssm layout, so the figure scripts work
        # on online runs unchanged.
        (run_dir / "checkpoint.msgpack").write_bytes(
            serialization.to_bytes(wm_state.params))
        (run_dir / "actor.msgpack").write_bytes(
            serialization.to_bytes(actor_state.params))
        (run_dir / "critic.msgpack").write_bytes(
            serialization.to_bytes(critic_state.params))
        if config.final_ac_steps > 0:
            (run_dir / "actor_refit.msgpack").write_bytes(
                serialization.to_bytes(refit_actor_state.params))
            (run_dir / "critic_refit.msgpack").write_bytes(
                serialization.to_bytes(refit_critic_state.params))

        # Gifs come from the refit actor once one exists — it is the
        # better policy — else the co-trained actor, as before.
        gif_actor_params = (refit_actor_state.params
                            if config.final_ac_steps > 0
                            else actor_state.params)
        ep = jax.device_get(runners["actor"](
            lane_key(root_key, K_EVAL, config.rounds + 2)[None],
            wm_state.params, gif_actor_params))
        frames, acts = ep["obs"][0], ep["action"][0]
        rollout_lib.single_gif(frames[:150], run_dir / "real_rollout.gif")
        fobs = jnp.asarray(frames[:5])[:, None]
        fact = jnp.asarray(acts[:5])[:, None]
        h_seq, z_seq = filter_episodes(model, wm_state.params, fobs, fact)
        imag = imagine_frames(model, wm_state.params, actor,
                              gif_actor_params, h_seq[-1], z_seq[-1], 30)
        rollout_lib.side_by_side_gif(frames[5:35], np.asarray(imag)[:, 0],
                                     run_dir / "imagination.gif")
        tracker.log_figure("loss_curves", metrics_figure(run_dir))

    tracker.finish()
    print("final eval:", json.dumps(final, indent=2))
    return final


def main():
    parser = argparse.ArgumentParser(
        description="Online Dreamer on the ball (collect -> train loop)")
    defaults = Config()
    parser.add_argument("--goal-speed", type=float,
                        default=defaults.goal_speed)
    parser.add_argument("--rounds", type=int, default=defaults.rounds)
    parser.add_argument("--wm-updates", type=int, default=defaults.wm_updates)
    parser.add_argument("--ac-updates", type=int, default=defaults.ac_updates)
    parser.add_argument("--final-ac-steps", type=int,
                        default=defaults.final_ac_steps)
    parser.add_argument("--episodes-per-round", type=int,
                        default=defaults.episodes_per_round)
    parser.add_argument("--seed-episodes", type=int,
                        default=defaults.seed_episodes)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--run-name", type=str, default=defaults.run_name)
    parser.add_argument("--resume", action="store_true")
    # Long-run default: wandb on, jsonl canonical. --no-wandb for local-only.
    parser.add_argument("--wandb-project", type=str, default="world-models")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    config = Config(
        seed=args.seed,
        goal_speed=args.goal_speed,
        rounds=args.rounds,
        wm_updates=args.wm_updates,
        ac_updates=args.ac_updates,
        final_ac_steps=args.final_ac_steps,
        episodes_per_round=args.episodes_per_round,
        seed_episodes=args.seed_episodes,
        run_name=args.run_name,
        resume=args.resume,
        wandb_project=None if args.no_wandb else args.wandb_project,
    )
    train(config)


if __name__ == "__main__":
    main()
