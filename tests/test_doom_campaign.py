"""DoomCampaign against the real engine.

Skipped without the doom extra, exactly as tests/test_doom.py is, and
skipped again without an IWAD, because these tests never download
anything: point WORLD_MODELS_WAD at one, or run scripts/fetch_wads.py,
which fills data/wads/. freedoom1.wad is the stand-in with boring
redistribution terms and the same E?M? map names as the shareware WAD,
so the same test reads either.

What lives here is what only the engine can answer: that the map loads
and renders, that the variables it reports move, and that the terminal
flavors come back the way the wrapper claims. Everything the wrapper
computes for itself is tested engine-free in tests/test_campaign_env.py.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("vizdoom")

from world_models.collect import ScriptedExplorer, collect_episode
from world_models.envs.doom import DoomCampaign

WAD = Path(os.environ.get("WORLD_MODELS_WAD")
           or Path(__file__).resolve().parents[1] / "data/wads/freedoom1.wad")

if not WAD.exists():
    pytest.skip(f"no IWAD at {WAD}: set WORLD_MODELS_WAD or run "
                "scripts/fetch_wads.py", allow_module_level=True)

# Ten seconds of game time: long enough to walk out of the first room,
# short enough that a test suite does not notice.
TIMEOUT_TICS = 350


@pytest.fixture(scope="module")
def env():
    """One engine for every test that does not need its own config.

    Booting the C++ process and loading a map is the expensive part,
    and each test resets, so sharing costs nothing in isolation.
    """
    with DoomCampaign(WAD, episode_timeout_tics=TIMEOUT_TICS, seed=0) as game:
        yield game


def test_reset_and_step_render_rgb_frames(env):
    obs = env.reset()
    assert obs.shape == (64, 64, 3) and obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0
    assert obs.max() > 0.0
    obs, reward, done = env.step(0)
    assert obs.shape == (64, 64, 3) and obs.dtype == np.float32
    assert isinstance(done, bool) and not done
    # No living reward, and nothing has been killed or picked up yet.
    assert reward == pytest.approx(0.0)


def test_variables_are_finite_and_move(env):
    env.reset()
    start = env.variables()
    assert start.shape == (len(DoomCampaign.VARIABLE_NAMES),)
    assert np.isfinite(start).all()
    assert start[DoomCampaign.VARIABLE_NAMES.index("health")] > 0

    for _ in range(20):
        env.step(0)  # forward
    moved = env.variables()
    assert np.isfinite(moved).all()
    assert not np.array_equal(start[:3], moved[:3]), "position never changed"


def test_finished_reason_is_none_until_the_episode_ends(env):
    env.reset()
    assert env.finished_reason is None and env.died is False


def test_a_tiny_tic_cap_ends_in_timeout():
    with DoomCampaign(WAD, episode_timeout_tics=40, seed=0) as game:
        game.reset()
        done = False
        for _ in range(50):
            _, _, done = game.step(1)  # turn in place, nothing can kill us
            if done:
                break
        assert done, "the 40 tic cap never fired"
        assert game.finished_reason == "timeout"
        assert game.died is False


def test_collect_episode_round_trip_with_the_scripted_explorer(env):
    explorer = ScriptedExplorer(env, seed=0)
    ep = collect_episode(env, explorer, 200)

    assert ep["obs"].dtype == np.uint8
    assert ep["obs"].shape[1:] == (64, 64, 3)
    assert ep["action"].shape == (ep["obs"].shape[0], 12)
    assert ep["reward"].shape == (ep["obs"].shape[0],)
    assert ep["variables"].shape == (ep["obs"].shape[0],
                                     len(DoomCampaign.VARIABLE_NAMES))
    assert np.isfinite(ep["variables"]).all()
    assert ep["reason"] in ("death", "exit", "timeout")
    assert ep["terminated"] is (ep["reason"] != "timeout")
    # The explorer only ever picks from the enabled part of the
    # vocabulary, so the reserved slots stay empty in the dataset.
    assert not ep["action"][:, 9:].any()


def test_same_seed_replays_the_same_frames():
    def run(seed, n=10):
        with DoomCampaign(WAD, episode_timeout_tics=TIMEOUT_TICS,
                          seed=seed) as game:
            frames = [game.reset()]
            for i in range(n):
                frames.append(game.step(i % 9)[0])
        return np.stack(frames)

    first, again = run(11), run(11)
    np.testing.assert_array_equal(first, again)
