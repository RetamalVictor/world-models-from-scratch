import dataclasses
import json

import numpy as np

from world_models.train_doom import Config, train


class FakeDoom:
    """Deterministic stand-in for the engine: frame value = step index,
    dies at die_at (cycling through a small list per episode so the
    buffer holds both deaths and timeouts). Constructed fresh per
    train() call, so an interrupted-and-resumed run sees exactly the
    episodes the uninterrupted one saw — the property the resume test
    needs and the real engine cannot offer.

    collect.py drops the killing step, so a death episode stores
    exactly die_at frames. Every die_at here is > transitions in the
    tiny config, which is what keeps each stored death episode
    sampleable."""

    action_dim = 3

    def __init__(self, die_ats=(6, None, 8), size=32):
        self.die_ats = die_ats
        self.size = size
        self._episode = -1
        self._t = 0
        self._died = False

    def reset(self):
        self._episode += 1
        self._t = 0
        self._died = False
        return self._frame()

    def step(self, action):
        self._t += 1
        die_at = self.die_ats[self._episode % len(self.die_ats)]
        done = die_at is not None and self._t >= die_at
        self._died = done
        return self._frame(), 0.0 if done else 0.01, done

    @property
    def died(self):
        return self._died

    def close(self):
        pass

    def _frame(self):
        v = (self._t % 32) / 32.0
        return np.full((self.size, self.size, 1), v, np.float32)


def _tiny_config(tmp_path, run_name, rounds, **overrides):
    base = Config(
        seed=0, frame_size=32, max_episode_steps=10,
        latent_dim=4, hidden=8,
        transitions=5, batch_size=2,
        horizon=3, seq_batch=1, seq_len=4,
        buffer_capacity=5000, seed_episodes=2,
        rounds=rounds, episodes_per_round=1,
        wm_updates=1, ac_updates=1, final_ac_steps=0,
        log_every=1, eval_every=1000, eval_episodes=1,
        checkpoint_every=2, keep_checkpoints=3,
        artifacts=False, resume=False,
        run_name=run_name, run_root=str(tmp_path),
    )
    return dataclasses.replace(base, **overrides)


def test_wm_only_trains_and_checkpoints(tmp_path):
    config = _tiny_config(tmp_path, "wm", rounds=2, wm_only=True)
    final = train(config, env_factory=lambda c: FakeDoom())

    run_dir = tmp_path / "wm"
    assert (run_dir / "buffer.npz").exists()
    assert (run_dir / "checkpoints" / "step_00000002.msgpack").exists()
    rows = [json.loads(line) for line in
            (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert any("continue_bce" in r for r in rows)
    assert all("actor_loss" not in r for r in rows)
    # wm_only never evaluates the untrained actor
    assert "actor" not in final and "random" in final


def test_full_loop_evals_actor_and_refit(tmp_path):
    config = _tiny_config(tmp_path, "full", rounds=2, final_ac_steps=2)
    final = train(config, env_factory=lambda c: FakeDoom())
    assert final["actor"]["mean_steps"] > 0
    assert final["actor_refit"]["mean_steps"] > 0
    assert final["random"]["mean_steps"] > 0
    eval_json = json.loads((tmp_path / "full" / "eval.json").read_text())
    assert set(eval_json) >= {"actor", "actor_refit", "random"}


def test_resume_matches_uninterrupted(tmp_path):
    # A constant death schedule: FakeDoom counts episodes per instance,
    # and a resumed run starts a fresh instance mid-run, so any
    # index-dependent schedule would desynchronize from the global
    # episode counter. Byte-identity is about the training math.
    train(_tiny_config(tmp_path, "a", rounds=4),
          env_factory=lambda c: FakeDoom(die_ats=(7,)))
    train(_tiny_config(tmp_path, "b", rounds=2),
          env_factory=lambda c: FakeDoom(die_ats=(7,)))
    train(_tiny_config(tmp_path, "b", rounds=4, resume=True),
          env_factory=lambda c: FakeDoom(die_ats=(7,)))

    ckpt_a = (tmp_path / "a" / "checkpoints" /
              "step_00000004.msgpack").read_bytes()
    ckpt_b = (tmp_path / "b" / "checkpoints" /
              "step_00000004.msgpack").read_bytes()
    assert ckpt_a == ckpt_b
