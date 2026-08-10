"""Online Dreamer on ViZDoom take_cover.

    uv run train-doom --wm-only --seed-episodes 1000 \
        --episodes-per-round 0 --rounds 200 --run-name pretrain

The train-online loop rebuilt for an env that lives on the CPU: the
engine is a stateful C++ process, so collection is the plain-Python
collect.py loop instead of a jitted scan, and everything after the
buffer — world model, imagination, actor-critic — stays jitted and
identical in shape to the ball recipe. Three things are new here, all
forced by the env: the continue head trains on death targets and its
predictions truncate imagined lambda-returns; the actor is categorical
(straight-through samples keep value gradients flowing through the
dynamics); and the world-model-first recipe gets a --wm-only mode plus
episodes_per_round=0, so a run can collect a big random buffer, train
the model alone, then --resume into the full loop in the same run dir.

Rng follows the train-online contract: every draw keyed by a persistent
counter, resume restores states, counters and buffer exactly. The one
honest exception is the engine itself — its internal state cannot be
checkpointed, so collection after a resume plays different episodes
than the uninterrupted run would have. Training math given the same
buffer is still bit-reproducible, and the resume test pins that with a
deterministic fake env.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from functools import partial
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
from world_models.collect import RandomPolicy, RSSMPolicy, collect_episode
from world_models.models.actor_critic import (Critic, DiscreteActor,
                                              lambda_returns)
from world_models.models.rssm import RSSM
from world_models.replay import ReplayBuffer
from world_models.tracking import Tracker, metrics_figure
from world_models.train_online import (K_AC, K_EPISODE, K_EVAL, K_REFIT,
                                       K_WM, lane_key)
from world_models.train_rssm import Config as RSSMConfig
from world_models.train_rssm import (filter_episodes, filmstrip_figure,
                                     make_losses, rollout_prior)


@dataclass(frozen=True)
class Config:
    seed: int = 0
    # env
    frame_size: int = 64
    action_repeat: int = 4
    max_episode_steps: int = 1000    # cap; take_cover has no own timeout
    # world model
    latent_dim: int = 32
    hidden: int = 256
    min_sigma: float = 0.1
    alpha: float = 0.8
    beta: float = 1.0
    wm_lr: float = 3e-4
    wm_grad_clip: float = 100.0
    # A take_cover episode is ~75 frames with one death at the end, so
    # unweighted the death class is ~1% of the continue head's targets
    # and "alive everywhere" is a near-optimal head. 64 puts the two
    # classes on the same order so the head is paid to find the cue.
    continue_death_weight: float = 64.0
    transitions: int = 32
    batch_size: int = 16
    # actor-critic
    horizon: int = 15
    gamma: float = 0.99              # survival task, long episodes
    lam: float = 0.95
    ent_coef: float = 1e-3           # categorical entropy, in nats
    ac_lr: float = 3e-4
    ac_grad_clip: float = 100.0
    seq_batch: int = 8
    seq_len: int = 16
    # the loop
    buffer_capacity: int = 200_000   # frames (~800 MB of uint8)
    seed_episodes: int = 20          # random policy, before round 1
    rounds: int = 500
    episodes_per_round: int = 1      # 0 = train on the buffer alone
    wm_updates: int = 100            # per round
    ac_updates: int = 100            # per round
    wm_only: bool = False            # random collection, no actor-critic
    final_ac_steps: int = 0          # fresh refit after the loop
    # bookkeeping
    log_every: int = 1
    eval_every: int = 25             # rounds; real episodes are slow
    eval_episodes: int = 3
    checkpoint_every: int = 25
    keep_checkpoints: int = 3
    artifacts: bool = True
    resume: bool = False
    run_name: str = "take-cover"
    run_root: str = "runs/doom"
    wandb_project: str | None = None


def make_env(config: Config):
    from world_models.envs.doom import DoomTakeCover
    return DoomTakeCover(frame_size=config.frame_size,
                         action_repeat=config.action_repeat,
                         seed=config.seed)


def make_imagination(wm: RSSM, actor: DiscreteActor, critic: Critic,
                     config: Config):
    """The ball imagination with the two Doom differences: actions are
    straight-through categorical samples, and the predicted continue
    probability truncates the lambda-returns — without it the actor
    would happily value what happens after it dies."""

    def imagine(wm_params, actor_params, h0, z0, key):
        s0 = jnp.concatenate([h0, z0], axis=-1)
        k_act, k_prior = jax.random.split(key)
        act_keys = jax.random.split(k_act, config.horizon)
        prior_noise = jax.random.normal(
            k_prior, (config.horizon,) + z0.shape)

        def step(carry, xs):
            h, z = carry
            k_t, pn = xs
            s = jnp.concatenate([h, z], axis=-1)
            a = actor.apply(actor_params, s, k_t,
                            method=DiscreteActor.sample_st)
            ent = actor.apply(actor_params, s,
                              method=DiscreteActor.entropy).mean()
            h = wm.apply(wm_params, h, z, a, method=RSSM.core_step)
            mu_p, sig_p = wm.apply(wm_params, h, method=RSSM.prior_dist)
            z = mu_p + sig_p * pn
            r = wm.apply(wm_params, h, z, method=RSSM.reward)
            c = jax.nn.sigmoid(
                wm.apply(wm_params, h, z, method=RSSM.continue_logit))
            return (h, z), (jnp.concatenate([h, z], axis=-1), r, c, ent)

        _, (states, rewards, continues, ents) = jax.lax.scan(
            step, (h0, z0), (act_keys, prior_noise))
        states = jnp.concatenate([s0[None], states], axis=0)
        return states, rewards, continues, ents.mean()

    def actor_loss_fn(actor_params, wm_params, critic_params, h0, z0, key):
        states, rewards, continues, entropy = imagine(
            wm_params, actor_params, h0, z0, key)
        values = critic.apply(critic_params, states)
        returns = lambda_returns(rewards, values[1:], config.gamma,
                                 config.lam, continues=continues)
        loss = -returns.mean() - config.ent_coef * entropy
        return loss, (states, rewards, continues, returns)

    def critic_loss_fn(critic_params, states_sg, rewards_sg, continues_sg):
        values = critic.apply(critic_params, states_sg)
        returns = lambda_returns(rewards_sg, values[1:], config.gamma,
                                 config.lam, continues=continues_sg)
        target = jax.lax.stop_gradient(returns)
        return ((values[:-1] - target) ** 2).mean()

    return actor_loss_fn, critic_loss_fn


def train(config: Config, env_factory=make_env) -> dict:
    run_dir = Path(config.run_root) / config.run_name
    if config.resume:
        if not run_dir.exists():
            raise SystemExit(f"{run_dir} does not exist; nothing to resume")
    elif run_dir.exists():
        raise SystemExit(
            f"{run_dir} already exists; --resume it, or pick another "
            "--run-name or delete it"
        )

    env = env_factory(config)
    action_dim = env.action_dim
    model = RSSM(
        latent_dim=config.latent_dim, action_dim=action_dim,
        hidden=config.hidden, min_sigma=config.min_sigma,
        obs_channels=1, obs_size=config.frame_size, predict_continue=True,
    )
    actor = DiscreteActor(action_dim=action_dim)
    critic = Critic()

    root_key = jax.random.PRNGKey(config.seed)
    k_wm, k_ac = jax.random.split(root_key)
    s = config.frame_size
    wm_params = model.init(
        k_wm, jnp.zeros((1, s, s, 1)), model.initial_state(1),
        jnp.zeros((1, config.latent_dim)), jnp.zeros((1, action_dim)),
    )
    s_dim = config.hidden + config.latent_dim
    wm_state = train_state.TrainState.create(
        apply_fn=model.apply, params=wm_params,
        tx=optax.chain(optax.clip_by_global_norm(config.wm_grad_clip),
                       optax.adam(config.wm_lr)),
    )

    def fresh_ac(key):
        ka, kc = jax.random.split(key)
        a_state = train_state.TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(ka, jnp.zeros((1, s_dim))),
            tx=optax.chain(optax.clip_by_global_norm(config.ac_grad_clip),
                           optax.adam(config.ac_lr)),
        )
        c_state = train_state.TrainState.create(
            apply_fn=critic.apply,
            params=critic.init(kc, jnp.zeros((1, s_dim))),
            tx=optax.chain(optax.clip_by_global_norm(config.ac_grad_clip),
                           optax.adam(config.ac_lr)),
        )
        return a_state, c_state

    actor_state, critic_state = fresh_ac(k_ac)

    counters = {"round": 0, "episode": 0, "wm_update": 0, "ac_update": 0}
    ckpt = Checkpointer(run_dir, keep=config.keep_checkpoints)
    buffer = ReplayBuffer(config.buffer_capacity)
    if config.resume:
        template = {"wm": wm_state, "actor": actor_state,
                    "critic": critic_state, "counters": counters}
        _, restored = ckpt.restore(template)
        wm_state, actor_state = restored["wm"], restored["actor"]
        critic_state, counters = restored["critic"], dict(restored["counters"])
        buffer = ReplayBuffer.load(run_dir / "buffer.npz")
        print(f"resumed from round {counters['round']} "
              f"({buffer.frames_added} env frames collected so far)")

    tracker = Tracker(
        run_dir,
        {
            **dataclasses.asdict(config),
            "action_dim": action_dim,
            "obs_channels": 1,
            "obs_size": config.frame_size,
            "predict_continue": True,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
        },
        wandb_project=config.wandb_project,
        run_name=config.run_name,
    )

    rssm_config = RSSMConfig(latent_dim=config.latent_dim,
                             alpha=config.alpha)
    sequence_loss = make_losses(model, rssm_config, has_reward=True,
                                has_continue=True,
                                death_weight=config.continue_death_weight)
    actor_loss_fn, critic_loss_fn = make_imagination(
        model, actor, critic, config)

    # Donated states: every call site rebinds its state to the returned
    # one, so the incoming params and optimizer moments are dead the
    # moment the step returns and XLA can write the update on top of
    # them instead of allocating a second copy.
    @partial(jax.jit, donate_argnums=0)
    def wm_step(state, obs_seq, act_seq, rew_seq, con_seq, key):
        grad_fn = jax.value_and_grad(sequence_loss, has_aux=True)
        (loss, (recon, kl, rew, con)), grads = grad_fn(
            state.params, obs_seq, act_seq, rew_seq, con_seq, key,
            jnp.float32(config.beta))
        return state.apply_gradients(grads=grads), loss, recon, kl, rew, con

    @partial(jax.jit, donate_argnums=(1, 2))
    def ac_step(wm_params, actor_state, critic_state, obs_seq, act_seq, key):
        h_seq, z_seq = filter_episodes(model, wm_params, obs_seq, act_seq)
        h0 = h_seq.reshape(-1, h_seq.shape[-1])
        z0 = z_seq.reshape(-1, z_seq.shape[-1])
        grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
        (a_loss, (states, rewards, continues, returns)), a_grads = grad_fn(
            actor_state.params, wm_params, critic_state.params, h0, z0, key)
        actor_state = actor_state.apply_gradients(grads=a_grads)
        c_loss, c_grads = jax.value_and_grad(critic_loss_fn)(
            critic_state.params, jax.lax.stop_gradient(states),
            jax.lax.stop_gradient(rewards),
            jax.lax.stop_gradient(continues))
        critic_state = critic_state.apply_gradients(grads=c_grads)
        return (actor_state, critic_state, a_loss, c_loss, rewards.mean(),
                returns.mean())

    # Collection policies. The RSSMPolicy's step jit lives for the whole
    # process; params are re-pointed per episode, never re-traced.
    explore_fn = jax.jit(lambda p, st, k: actor.apply(
        p, st[None], k, method=DiscreteActor.act)[0])
    greedy_fn = jax.jit(lambda p, st: actor.apply(
        p, st[None], None, method=DiscreteActor.act)[0])
    explore_policy = RSSMPolicy(
        model, lambda p, st, k: explore_fn(p, st, k), root_key, action_dim)
    greedy_policy = RSSMPolicy(
        model, lambda p, st, k: greedy_fn(p, st), root_key, action_dim)

    def collect(policy_kind: str) -> tuple[int, float]:
        """One episode into the buffer; returns (steps survived, reward)."""
        n = counters["episode"]
        if policy_kind == "random":
            policy = RandomPolicy(action_dim,
                                  seed=config.seed * 1_000_003 + n)
        else:
            policy = explore_policy
            policy.reseed(lane_key(root_key, K_EPISODE, n))
            policy.set_params(wm_state.params, actor_state.params)
        ep = collect_episode(env, policy, config.max_episode_steps)
        buffer.add_episode(ep["obs"], ep["action"], ep["reward"],
                           ep["terminated"])
        counters["episode"] += 1
        return ep["obs"].shape[0] - 1, float(ep["reward"][1:].sum())

    def real_eval(n_episodes: int, actor_params) -> dict:
        greedy_policy.set_params(wm_state.params, actor_params)
        steps, rewards = [], []
        for i in range(n_episodes):
            greedy_policy.reseed(lane_key(root_key, K_EVAL, i))
            ep = collect_episode(env, greedy_policy,
                                 config.max_episode_steps)
            steps.append(ep["obs"].shape[0] - 1)
            rewards.append(float(ep["reward"][1:].sum()))
        return {
            "mean_steps": float(np.mean(steps)),
            "std_steps": float(np.std(steps)),
            "mean_tics": float(np.mean(steps) * config.action_repeat),
            "mean_reward": float(np.mean(rewards)),
        }

    def save_checkpoint(metric):
        tree = {"wm": wm_state, "actor": actor_state, "critic": critic_state,
                "counters": dict(counters)}
        ckpt.save(counters["round"], tree, metric=metric)
        buffer.save(run_dir / "buffer.npz")

    while counters["episode"] < config.seed_episodes:
        collect("random")

    for r in range(counters["round"] + 1, config.rounds + 1):
        t0 = time.monotonic()
        collected = [collect("random" if config.wm_only else "explore")
                     for _ in range(config.episodes_per_round)]
        collect_seconds = time.monotonic() - t0

        for _ in range(config.wm_updates):
            u = counters["wm_update"]
            batch_rng = np.random.default_rng([config.seed, K_WM, u])
            obs_seq, act_seq, rew_seq, con_seq = (
                buffer.sample_sequences_with_continues(
                    batch_rng, config.batch_size, config.transitions))
            wm_state, loss, recon, kl, rew_mse, con_bce = wm_step(
                wm_state, obs_seq, act_seq, rew_seq, con_seq,
                lane_key(root_key, K_WM, u))
            counters["wm_update"] += 1

        if not config.wm_only:
            for _ in range(config.ac_updates):
                u = counters["ac_update"]
                batch_rng = np.random.default_rng([config.seed, K_AC, u])
                obs_seq, act_seq, _ = buffer.sample_sequences(
                    batch_rng, config.seq_batch, config.seq_len)
                (actor_state, critic_state, a_loss, c_loss, imag_reward,
                 imag_return) = ac_step(
                    wm_state.params, actor_state, critic_state, obs_seq,
                    act_seq, lane_key(root_key, K_AC, u))
                counters["ac_update"] += 1

        counters["round"] = r
        if r % config.log_every == 0:
            metrics = {"recon": recon, "kl": kl, "reward_mse": rew_mse,
                       "continue_bce": con_bce,
                       "env_frames": buffer.frames_added}
            if collected:
                metrics["collect_steps"] = np.mean([c[0] for c in collected])
                metrics["collect_reward"] = np.mean([c[1] for c in collected])
                metrics["collect_seconds"] = collect_seconds
            if not config.wm_only:
                metrics.update(actor_loss=a_loss, critic_loss=c_loss,
                               imag_reward=imag_reward)
            tracker.log(r, **metrics)
        evaluated = None
        if (not config.wm_only
                and (r % config.eval_every == 0 or r == config.rounds)):
            evaluated = real_eval(config.eval_episodes, actor_state.params)
            tracker.log(r, real_steps=evaluated["mean_steps"])
            print(f"round {r:4d}  recon {recon:8.3f}  kl {kl:6.3f}  "
                  f"con {con_bce:6.4f}  "
                  f"survival {evaluated['mean_steps']:6.1f} steps")
        if r % config.checkpoint_every == 0 or r == config.rounds:
            save_checkpoint(evaluated["mean_steps"] if evaluated else None)

    final = {"env_frames": buffer.frames_added}
    n_rand = max(config.eval_episodes, 3)
    rand_steps = []
    for i in range(n_rand):
        ep = collect_episode(
            env, RandomPolicy(action_dim, seed=config.seed * 7 + i),
            config.max_episode_steps)
        rand_steps.append(ep["obs"].shape[0] - 1)
    final["random"] = {"mean_steps": float(np.mean(rand_steps)),
                       "mean_tics": float(np.mean(rand_steps)
                                          * config.action_repeat)}
    if not config.wm_only:
        final["actor"] = real_eval(config.eval_episodes, actor_state.params)

    refit_actor_state = None
    if config.final_ac_steps > 0:
        refit_actor_state, refit_critic_state = fresh_ac(
            lane_key(root_key, K_REFIT, 0))
        for u in range(1, config.final_ac_steps + 1):
            batch_rng = np.random.default_rng([config.seed, K_REFIT, u])
            obs_seq, act_seq, _ = buffer.sample_sequences(
                batch_rng, config.seq_batch, config.seq_len)
            (refit_actor_state, refit_critic_state, a_loss, c_loss,
             imag_reward, _) = ac_step(
                wm_state.params, refit_actor_state, refit_critic_state,
                obs_seq, act_seq, lane_key(root_key, K_REFIT, u))
            if u % 500 == 0 or u == config.final_ac_steps:
                tracker.log(config.rounds + u, refit_actor_loss=a_loss,
                            refit_imag_reward=imag_reward)
        final["actor_refit"] = real_eval(config.eval_episodes,
                                         refit_actor_state.params)
    tracker.log_json("eval", final)

    if config.artifacts:
        (run_dir / "checkpoint.msgpack").write_bytes(
            serialization.to_bytes(wm_state.params))
        (run_dir / "actor.msgpack").write_bytes(
            serialization.to_bytes(actor_state.params))
        if refit_actor_state is not None:
            (run_dir / "actor_refit.msgpack").write_bytes(
                serialization.to_bytes(refit_actor_state.params))

        # Dream filmstrip: warm the filter on a few real frames from the
        # longest stored episode, then let the prior run free. This is
        # the Phase C1 checkpoint — fireballs continuing their arcs.
        warmup, horizon = 4, 30
        episode = buffer.longest_episode()
        need = warmup + horizon + 1
        if episode["obs"].shape[0] >= need:
            obs = jnp.asarray(episode["obs"][None], jnp.float32) / 255.0
            obs = obs.transpose(1, 0, 2, 3, 4)          # (T+1, 1, H, W, 1)
            act = jnp.asarray(episode["action"][None]).transpose(1, 0, 2)
            h_seq, z_seq = filter_episodes(model, wm_state.params,
                                           obs[:warmup], act[:warmup])
            # Sampled prior, not the mean. The mean rollout almost
            # never spawns a fireball and smears the ones already in
            # flight, so a mean filmstrip undersells the model and
            # shows a world the actor never trains in. The key is fixed
            # off the run seed so the artifact stays reproducible.
            noise = jax.random.normal(jax.random.PRNGKey(config.seed + 7),
                                      (horizon, 1, config.latent_dim))
            frames = rollout_prior(model, wm_state.params,
                                   h_seq[warmup - 1], z_seq[warmup - 1],
                                   act[warmup:warmup + horizon], noise)
            true = obs[warmup:warmup + horizon]
            tracker.log_figure(
                "dream_filmstrip",
                filmstrip_figure(np.asarray(frames), np.asarray(true)))
            rollout_lib.side_by_side_gif(
                np.asarray(true)[:, 0, :, :, 0],
                np.asarray(frames)[:, 0, :, :, 0],
                run_dir / "dream.gif")
        tracker.log_figure("loss_curves", metrics_figure(run_dir))

    env.close()
    tracker.finish()
    print("final eval:", json.dumps(final, indent=2))
    return final


def main():
    parser = argparse.ArgumentParser(
        description="Online Dreamer on ViZDoom take_cover")
    defaults = Config()
    parser.add_argument("--rounds", type=int, default=defaults.rounds)
    parser.add_argument("--wm-updates", type=int, default=defaults.wm_updates)
    parser.add_argument("--ac-updates", type=int, default=defaults.ac_updates)
    parser.add_argument("--episodes-per-round", type=int,
                        default=defaults.episodes_per_round)
    parser.add_argument("--seed-episodes", type=int,
                        default=defaults.seed_episodes)
    parser.add_argument("--buffer-capacity", type=int,
                        default=defaults.buffer_capacity)
    parser.add_argument("--wm-only", action="store_true")
    parser.add_argument("--final-ac-steps", type=int,
                        default=defaults.final_ac_steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--run-name", type=str, default=defaults.run_name)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="world-models")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    config = Config(
        seed=args.seed,
        rounds=args.rounds,
        wm_updates=args.wm_updates,
        ac_updates=args.ac_updates,
        episodes_per_round=args.episodes_per_round,
        seed_episodes=args.seed_episodes,
        buffer_capacity=args.buffer_capacity,
        wm_only=args.wm_only,
        final_ac_steps=args.final_ac_steps,
        run_name=args.run_name,
        resume=args.resume,
        wandb_project=None if args.no_wandb else args.wandb_project,
    )
    train(config)


if __name__ == "__main__":
    main()
