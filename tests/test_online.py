import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_models.envs import GoalEnvParams
from world_models.models.actor_critic import Actor
from world_models.models.rssm import RSSM
from world_models.train_online import Config, make_runners, train


def _tiny_config(tmp_path, run_name, rounds, resume=False):
    return Config(
        seed=0, goal_speed=1.0, episode_steps=10,
        latent_dim=4, hidden=8,
        buffer_capacity=2000, seed_episodes=2,
        rounds=rounds, episodes_per_round=1,
        wm_updates=1, ac_updates=1, final_ac_steps=0,
        transitions=5, batch_size=2,
        horizon=3, seq_batch=1, seq_len=4,
        log_every=1, eval_every=1000, eval_episodes=1,
        final_eval_episodes=1,
        checkpoint_every=2, keep_checkpoints=3,
        artifacts=False, resume=resume,
        run_name=run_name, run_root=str(tmp_path),
    )


def test_runner_episode_contract():
    env_params = GoalEnvParams(goal_speed=1.0)
    wm = RSSM(latent_dim=4, hidden=8, obs_channels=3)
    actor = Actor(max_action=env_params.max_nudge)
    key = jax.random.PRNGKey(0)
    wm_params = wm.init(key, jnp.zeros((1, 32, 32, 3)), wm.initial_state(1),
                        jnp.zeros((1, 4)), jnp.zeros((1, 2)))
    actor_params = actor.init(key, jnp.zeros((1, 12)))

    runners = make_runners(wm, actor, env_params, n_steps=6)
    ep = runners["explore"](jax.random.split(key, 2), wm_params, actor_params)
    assert ep["obs"].shape == (2, 7, 32, 32, 3)
    assert ep["action"].shape == (2, 7, 2)
    assert ep["reward"].shape == (2, 7)
    assert float(ep["obs"].min()) >= 0.0 and float(ep["obs"].max()) <= 1.0
    # dataset convention: reset frame first, zero action, computed reward
    assert np.allclose(np.asarray(ep["action"][:, 0]), 0.0)
    assert float(ep["reward"].min()) > 0.0
    # explore actually acts; zero doesn't
    assert float(jnp.abs(ep["action"][:, 1:]).max()) > 0.0
    ep_zero = runners["zero"](jax.random.split(key, 2), wm_params,
                              actor_params)
    assert float(jnp.abs(ep_zero["action"]).max()) == 0.0


def test_online_run_writes_run_dir_and_resume_matches(tmp_path):
    # The Phase A criterion in miniature: an uninterrupted run and a
    # killed-and-resumed run must be the same run, byte for byte.
    final_a = train(_tiny_config(tmp_path, "a", rounds=4))
    train(_tiny_config(tmp_path, "b", rounds=2))
    train(_tiny_config(tmp_path, "b", rounds=4, resume=True))

    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    assert (dir_a / "metrics.jsonl").exists()
    assert (dir_a / "buffer.npz").exists()
    assert (dir_a / "eval.json").exists()
    assert 0.0 <= final_a["actor"]["mean_reward"] <= 1.0
    assert final_a["env_frames"] == (2 + 4) * 11   # seed + rounds, T+1 each

    ckpt_a = (dir_a / "checkpoints" / "step_00000004.msgpack").read_bytes()
    ckpt_b = (dir_b / "checkpoints" / "step_00000004.msgpack").read_bytes()
    assert ckpt_a == ckpt_b


def test_fresh_run_refuses_existing_dir(tmp_path):
    config = _tiny_config(tmp_path, "c", rounds=2)
    (tmp_path / "c").mkdir()
    with pytest.raises(SystemExit):
        train(config)
    with pytest.raises(SystemExit):
        train(dataclasses.replace(config, run_name="missing", resume=True))


def test_final_actor_refit_evals_both_actors_and_is_deterministic(tmp_path):
    # The refit is a pure function of (config, final WM, buffer): two
    # independent runs of the same config must land on the same number.
    config = dataclasses.replace(
        _tiny_config(tmp_path, "refit-a", rounds=2), final_ac_steps=2)
    final_a = train(config)
    final_b = train(dataclasses.replace(config, run_name="refit-b"))

    for final in (final_a, final_b):
        assert isinstance(final["actor"]["mean_reward"], float)
        assert isinstance(final["actor_refit"]["mean_reward"], float)

    eval_a = json.loads((tmp_path / "refit-a" / "eval.json").read_text())
    assert "actor" in eval_a and "actor_refit" in eval_a
    assert (final_a["actor_refit"]["mean_reward"]
            == final_b["actor_refit"]["mean_reward"])
