import numpy as np
import pytest

from world_models.replay import ReplayBuffer


def _episode(value: int, length: int):
    obs = np.full((length, 4, 4, 3), value, np.uint8)
    action = np.full((length, 2), float(value), np.float32)
    reward = np.full((length,), float(value), np.float32)
    return obs, action, reward


def _variables(value: int, length: int, n_vars: int = 3):
    return np.full((length, n_vars), float(value), np.float32)


def test_rejects_float_frames():
    buffer = ReplayBuffer(capacity=100)
    obs, action, reward = _episode(1, 10)
    with pytest.raises(ValueError):
        buffer.add_episode(obs.astype(np.float32), action, reward)


def test_eviction_drops_whole_oldest_episodes():
    buffer = ReplayBuffer(capacity=25)
    for value in (1, 2, 3):
        buffer.add_episode(*_episode(value, 10))
    # 30 frames > 25: episode 1 is gone entirely, 2 and 3 intact
    assert len(buffer) == 20
    assert buffer.n_episodes == 2
    assert buffer.frames_added == 30
    rng = np.random.default_rng(0)
    obs, _, _ = buffer.sample_sequences(rng, 64, 4)
    values = np.unique(np.round(np.asarray(obs) * 255.0))
    assert 1.0 not in values


def test_sequences_never_straddle_a_reset():
    buffer = ReplayBuffer(capacity=1000)
    for value in (1, 2, 3):
        buffer.add_episode(*_episode(value, 8))
    rng = np.random.default_rng(0)
    obs, action, reward = buffer.sample_sequences(rng, 128, 5)
    assert obs.shape == (6, 128, 4, 4, 3)
    assert action.shape == (6, 128, 2)
    assert reward.shape == (6, 128)
    # every sampled sequence stays inside one constant-valued episode
    per_seq = np.asarray(reward)
    assert (per_seq.min(axis=0) == per_seq.max(axis=0)).all()


def test_sampling_uses_every_valid_start():
    buffer = ReplayBuffer(capacity=100)
    obs = np.zeros((6, 4, 4, 3), np.uint8)
    action = np.zeros((6, 2), np.float32)
    reward = np.arange(6, dtype=np.float32)
    buffer.add_episode(obs, action, reward)
    rng = np.random.default_rng(0)
    _, _, rew = buffer.sample_sequences(rng, 256, 3)
    # length 6, 3 transitions -> starts 0, 1, 2 all reachable
    assert set(np.asarray(rew)[0].astype(int)) == {0, 1, 2}


def test_too_short_episodes_raise():
    buffer = ReplayBuffer(capacity=100)
    buffer.add_episode(*_episode(1, 4))
    with pytest.raises(ValueError):
        buffer.sample_sequences(np.random.default_rng(0), 8, 10)


def test_save_load_roundtrip(tmp_path):
    buffer = ReplayBuffer(capacity=50)
    for value in (1, 2, 3, 4, 5, 6):
        buffer.add_episode(*_episode(value, 10))
    path = tmp_path / "buffer.npz"
    buffer.save(path)
    loaded = ReplayBuffer.load(path)

    assert loaded.capacity == buffer.capacity
    assert len(loaded) == len(buffer)
    assert loaded.n_episodes == buffer.n_episodes
    assert loaded.frames_added == buffer.frames_added
    a = buffer.sample_sequences(np.random.default_rng(7), 32, 4)
    b = loaded.sample_sequences(np.random.default_rng(7), 32, 4)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_continue_targets_mark_only_deaths():
    buffer = ReplayBuffer(capacity=100)
    buffer.add_episode(*_episode(1, 6), terminated=True)
    buffer.add_episode(*_episode(2, 6))          # timed out, did not die
    rng = np.random.default_rng(0)
    obs, act, rew, cont = buffer.sample_sequences_with_continues(rng, 64, 5)
    assert cont.shape == (6, 64)
    # 5 transitions out of 6 frames: every sequence is a whole episode
    died = np.asarray(rew)[0] == 1.0
    np.testing.assert_array_equal(np.asarray(cont)[:5], 1.0)
    np.testing.assert_array_equal(np.asarray(cont)[5], (~died).astype(float))


def test_continues_only_appear_at_the_true_episode_end():
    buffer = ReplayBuffer(capacity=100)
    buffer.add_episode(*_episode(1, 8), terminated=True)
    rng = np.random.default_rng(0)
    _, _, rew, cont = buffer.sample_sequences_with_continues(rng, 128, 3)
    # starts 0..4 are valid; only start 4 reaches the terminal frame
    np.testing.assert_array_equal(np.asarray(cont)[:3], 1.0)
    assert set(np.asarray(cont)[3].tolist()) == {0.0, 1.0}


def test_sampling_with_continues_matches_the_plain_sampler():
    buffer = ReplayBuffer(capacity=200)
    for value in (1, 2, 3):
        buffer.add_episode(*_episode(value, 10), terminated=value == 2)
    plain = buffer.sample_sequences(np.random.default_rng(3), 16, 4)
    with_c = buffer.sample_sequences_with_continues(
        np.random.default_rng(3), 16, 4)
    for x, y in zip(plain, with_c):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_terminated_survives_save_load(tmp_path):
    buffer = ReplayBuffer(capacity=100)
    buffer.add_episode(*_episode(1, 6), terminated=True)
    buffer.add_episode(*_episode(2, 6))
    path = tmp_path / "buffer.npz"
    buffer.save(path)
    loaded = ReplayBuffer.load(path)
    a = buffer.sample_sequences_with_continues(np.random.default_rng(1), 32, 5)
    b = loaded.sample_sequences_with_continues(np.random.default_rng(1), 32, 5)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_campaign_fields_survive_save_load(tmp_path):
    buffer = ReplayBuffer(capacity=100)
    for value, reason in ((1, "exit"), (2, "death"), (3, "timeout")):
        buffer.add_episode(*_episode(value, 6), terminated=reason != "timeout",
                           variables=_variables(value, 6), reason=reason)
    path = tmp_path / "campaign.npz"
    buffer.save(path)
    loaded = ReplayBuffer.load(path)

    assert loaded.n_episodes == 3 and len(loaded) == len(buffer)
    for before, after in zip(buffer._episodes, loaded._episodes):
        np.testing.assert_array_equal(before["variables"], after["variables"])
        assert before["reason"] == after["reason"]


def test_campaign_fields_are_evicted_with_their_episode():
    buffer = ReplayBuffer(capacity=25)
    for value in (1, 2, 3):
        buffer.add_episode(*_episode(value, 10),
                           variables=_variables(value, 10), reason="exit")
    assert buffer.n_episodes == 2
    kept = [ep["variables"][0, 0] for ep in buffer._episodes]
    assert kept == [2.0, 3.0]


def test_sampling_ignores_the_campaign_fields():
    """Sequence sampling is untouched: the same seed draws the same
    batch whether or not variables ride along."""
    plain, extra = ReplayBuffer(capacity=200), ReplayBuffer(capacity=200)
    for value in (1, 2, 3):
        plain.add_episode(*_episode(value, 10), terminated=value == 2)
        extra.add_episode(*_episode(value, 10), terminated=value == 2,
                          variables=_variables(value, 10), reason="exit")
    a = plain.sample_sequences_with_continues(np.random.default_rng(3), 16, 4)
    b = extra.sample_sequences_with_continues(np.random.default_rng(3), 16, 4)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_buffers_without_variables_save_the_file_they_always_did(tmp_path):
    """The byte-identical guard: no campaign episodes, no new arrays in
    the npz, so a take_cover dataset written today is one an older
    checkout could read."""
    buffer = ReplayBuffer(capacity=100)
    buffer.add_episode(*_episode(1, 6), terminated=True)
    path = tmp_path / "legacy.npz"
    buffer.save(path)
    with np.load(path) as f:
        assert sorted(f.files) == ["action", "capacity", "frames_added",
                                   "lengths", "obs", "reward", "terminated"]


def test_mixing_episodes_with_and_without_variables_is_rejected():
    buffer = ReplayBuffer(capacity=100)
    buffer.add_episode(*_episode(1, 6), variables=_variables(1, 6))
    with pytest.raises(ValueError):
        buffer.add_episode(*_episode(2, 6))

    other = ReplayBuffer(capacity=100)
    other.add_episode(*_episode(1, 6))
    with pytest.raises(ValueError):
        other.add_episode(*_episode(2, 6), variables=_variables(2, 6))


def test_variables_must_have_one_row_per_frame():
    buffer = ReplayBuffer(capacity=100)
    with pytest.raises(ValueError):
        buffer.add_episode(*_episode(1, 6), variables=_variables(1, 5))


def test_variables_must_keep_the_same_width_across_a_buffer():
    buffer = ReplayBuffer(capacity=100)
    buffer.add_episode(*_episode(1, 6), variables=_variables(1, 6, n_vars=3))
    with pytest.raises(ValueError):
        buffer.add_episode(*_episode(2, 6), variables=_variables(2, 6, n_vars=4))


def test_loads_buffers_written_before_the_terminated_flag(tmp_path):
    """Older files have no terminated array; the ball only times out."""
    obs, action, reward = _episode(1, 6)
    path = tmp_path / "old_buffer.npz"
    np.savez_compressed(
        path, obs=obs, action=action, reward=reward,
        lengths=np.array([6]), capacity=100, frames_added=6,
    )
    loaded = ReplayBuffer.load(path)
    assert len(loaded) == 6 and loaded.n_episodes == 1
    _, _, _, cont = loaded.sample_sequences_with_continues(
        np.random.default_rng(0), 8, 5)
    np.testing.assert_array_equal(np.asarray(cont), 1.0)
