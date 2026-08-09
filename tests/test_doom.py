import numpy as np
import pytest

pytest.importorskip("vizdoom")

from world_models.envs.doom import DoomTakeCover

STEP_CAP = 2000  # take_cover kills a scripted policy in well under this


@pytest.fixture(scope="module")
def env():
    """One engine for every test that does not need its own seed.

    Booting the C++ process is the expensive part of these tests, and
    each test resets, so sharing one costs nothing in isolation.
    """
    with DoomTakeCover(seed=0) as game:
        yield game


def test_reset_and_step_frames(env):
    obs = env.reset()
    assert obs.shape == (64, 64, 1)
    assert obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0
    assert obs.max() > 0.0
    obs, reward, done = env.step(0)
    assert obs.shape == (64, 64, 1) and obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0
    assert isinstance(done, bool) and not done
    assert reward == pytest.approx(0.01)


def test_rejects_malformed_actions(env):
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.zeros(4))
    with pytest.raises(ValueError):
        env.step(7)


def test_scripted_policy_dies_and_reward_counts_survival(env):
    """Strafing one way into the fireballs ends the episode."""
    env.reset()
    total, survived, done = 0.0, 0, False
    for _ in range(STEP_CAP):
        _, reward, done = env.step(0)
        total += reward
        survived += not done
        if done:
            break
    assert done, f"no death in {STEP_CAP} steps"
    assert total == pytest.approx(survived / 100.0)


def test_frame_size_is_configurable():
    with DoomTakeCover(frame_size=32, seed=0) as small:
        assert small.reset().shape == (32, 32, 1)


def test_index_and_one_hot_actions_agree():
    frames = []
    for action in (1, np.array([0.0, 1.0, 0.0])):
        with DoomTakeCover(seed=3) as game:
            game.reset()
            frames.append(np.stack([game.step(action)[0] for _ in range(5)]))
    np.testing.assert_array_equal(frames[0], frames[1])


def test_same_seed_replays_the_same_frames():
    def run(seed, n=20):
        with DoomTakeCover(seed=seed) as game:
            frames = [game.reset()]
            for i in range(n):
                frames.append(game.step(i % 3)[0])
        return np.stack(frames)

    first, again, other = run(11), run(11), run(12)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
