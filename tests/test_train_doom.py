import dataclasses
import json

import numpy as np
from flax import serialization

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


def test_imagination_temperature_neutral_at_one(tmp_path):
    # tau only scales the prior sample inside ac_step's imagination, so
    # at 1.0 the whole run must stay byte-identical to the default
    # (IEEE multiply by 1.0 is exact) and at 2.0 the actor and critic
    # in the checkpoint must move.
    for name, tau in (("t-default", None), ("t-one", 1.0), ("t-two", 2.0)):
        overrides = {} if tau is None else {"imagination_temperature": tau}
        train(_tiny_config(tmp_path, name, rounds=1, **overrides),
              env_factory=lambda c: FakeDoom())

    def ckpt(name):
        return (tmp_path / name / "checkpoints" /
                "step_00000001.msgpack").read_bytes()

    assert ckpt("t-default") == ckpt("t-one")
    assert ckpt("t-default") != ckpt("t-two")


def test_target_and_reset_defaults_are_neutral(tmp_path):
    # critic_ema=0.0 keeps target handling out of the traced ac_step
    # and out of the checkpoint tree, and ac_reset_every=0 never fires,
    # so explicit zeros must reproduce the default run byte for byte.
    train(_tiny_config(tmp_path, "z-default", rounds=1),
          env_factory=lambda c: FakeDoom())
    train(_tiny_config(tmp_path, "z-zeros", rounds=1, critic_ema=0.0,
                       ac_reset_every=0),
          env_factory=lambda c: FakeDoom())

    def ckpt(name):
        return (tmp_path / name / "checkpoints" /
                "step_00000001.msgpack").read_bytes()

    assert ckpt("z-default") == ckpt("z-zeros")


def test_each_feature_changes_training(tmp_path):
    # Two rounds because the EMA target equals the live critic until
    # the first ac step has moved the live params; only the second
    # step bootstraps from a target that lags.
    for name, overrides in (("f-default", {}),
                            ("f-ema", {"critic_ema": 0.9}),
                            ("f-reset", {"ac_reset_every": 1})):
        train(_tiny_config(tmp_path, name, rounds=2, **overrides),
              env_factory=lambda c: FakeDoom())

    def ckpt(name):
        return (tmp_path / name / "checkpoints" /
                "step_00000002.msgpack").read_bytes()

    # The ema tree differs trivially by carrying the target, so compare
    # the shared actor subtree: it only moves differently if the
    # lagging bootstrap actually changed the imagined returns.
    default = serialization.msgpack_restore(ckpt("f-default"))
    ema = serialization.msgpack_restore(ckpt("f-ema"))
    assert "critic_target" in ema and "critic_target" not in default
    assert (serialization.msgpack_serialize(ema["actor"])
            != serialization.msgpack_serialize(default["actor"]))
    # A reset fires at round 1 (round 2 is final, exempt) and reinits
    # from the K_RESET lane, so the whole trajectory moves.
    assert ckpt("f-reset") != ckpt("f-default")


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


def test_resume_with_target_and_resets(tmp_path):
    # test_resume_matches_uninterrupted with both features on. The
    # schedule puts one reset before the interrupt (r=2) and one after
    # the resume (r=4, keyed only by seed and round number), with r=6
    # exempt as the final round; critic_ema puts the target params in
    # the checkpoint tree. Byte-identity at round 6 proves both
    # survive the round trip.
    kw = dict(critic_ema=0.9, ac_reset_every=2)
    train(_tiny_config(tmp_path, "ta", rounds=6, **kw),
          env_factory=lambda c: FakeDoom(die_ats=(7,)))
    train(_tiny_config(tmp_path, "tb", rounds=3, **kw),
          env_factory=lambda c: FakeDoom(die_ats=(7,)))
    train(_tiny_config(tmp_path, "tb", rounds=6, resume=True, **kw),
          env_factory=lambda c: FakeDoom(die_ats=(7,)))

    ckpt_a = (tmp_path / "ta" / "checkpoints" /
              "step_00000006.msgpack").read_bytes()
    ckpt_b = (tmp_path / "tb" / "checkpoints" /
              "step_00000006.msgpack").read_bytes()
    assert ckpt_a == ckpt_b


def test_resume_feature_off_checkpoint_into_ema_config(tmp_path):
    # A default-config pretrain saves no critic_target, but resuming it
    # into a critic_ema run puts that key in the restore template,
    # which used to crash flax restore on the mismatch. train() now
    # drops the key and seeds the target from the restored critic, so
    # the resumed full loop must run through: r=3 does ac updates on
    # the seeded target, r=4 exercises a reset under the ema config,
    # and the final checkpoint carries the target and the new rounds.
    train(_tiny_config(tmp_path, "off-ema", rounds=2, wm_only=True),
          env_factory=lambda c: FakeDoom())
    train(_tiny_config(tmp_path, "off-ema", rounds=5, resume=True,
                       critic_ema=0.9, ac_reset_every=4),
          env_factory=lambda c: FakeDoom())

    tree = serialization.msgpack_restore(
        (tmp_path / "off-ema" / "checkpoints" /
         "step_00000005.msgpack").read_bytes())
    assert tree["counters"]["round"] == 5
    assert "critic_target" in tree
