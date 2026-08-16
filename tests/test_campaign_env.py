"""DoomCampaign with a fake engine standing in for vizdoom.

Everything the wrapper does around the engine (reward from variable
deltas, terminal flavors, the action vocabulary, the repeat convention
at the end of an episode) is plain Python over whatever the engine
reports, so it is testable on a machine that has no engine at all, and
this is where that is done. tests/test_doom_campaign.py covers what
only the real thing can answer, and skips without it.
"""

import sys
import types

import numpy as np
import pytest

from world_models.envs.doom import DoomCampaign, _action_index, _resize_frame


class FakeEnums:
    """vizdoom's enums as their own names.

    The wrapper only ever looks a member up and hands it straight back
    to the engine, so a name is as good as an enum here, and it reads
    better in an assertion.
    """

    def __getattr__(self, name):
        return name


class FakeState:
    def __init__(self, screen_buffer, game_variables):
        self.screen_buffer = screen_buffer
        self.game_variables = game_variables


class FakeGame:
    """The slice of vizdoom.DoomGame the wrapper touches.

    Scripted with one variable row per live state (row 0 is the reset
    state); once they run out the episode is over and get_state()
    returns None, which is the engine behavior the whole terminal-frame
    convention rests on. Every setter call is recorded in order, so the
    configuration tests can read back what was asked for and when.
    """

    def __init__(self, rows, ending="exit"):
        self.calls = []
        self.rows = [np.asarray(row, np.float32) for row in rows]
        self.ending = ending
        self.buttons = []
        self.time = 0
        self._t = 0

    def __getattr__(self, name):
        if name.startswith("set_"):
            def setter(*args):
                self.calls.append((name, args))
            return setter
        raise AttributeError(name)

    def init(self):
        self.calls.append(("init", ()))

    def close(self):
        self.calls.append(("close", ()))

    def new_episode(self):
        self._t = 0
        self.time = 0

    def make_action(self, buttons, tics):
        self.buttons.append(list(buttons))
        self._t += 1
        self.time += tics

    def is_episode_finished(self):
        return self._t >= len(self.rows)

    def is_player_dead(self):
        return self.is_episode_finished() and self.ending == "death"

    def get_episode_time(self):
        return self.time

    def get_state(self):
        if self.is_episode_finished():
            return None
        # Brightness is the live step index plus one, so a frame says
        # which state rendered it and a repeat is visible as a repeat.
        screen = np.full((3, 120, 160), self._t + 1, np.uint8)
        return FakeState(screen, self.rows[self._t])


def _row(**values):
    """A variable row by name, everything unmentioned zero."""
    row = np.zeros(len(DoomCampaign.VARIABLE_NAMES), np.float32)
    for name, value in values.items():
        row[DoomCampaign.VARIABLE_NAMES.index(name)] = value
    return row


def _campaign(monkeypatch, rows, ending="exit", **kwargs):
    game = FakeGame(rows, ending)
    module = types.ModuleType("vizdoom")
    module.DoomGame = lambda: game
    for enum in ("ScreenFormat", "ScreenResolution", "Mode", "Button",
                 "GameVariable"):
        setattr(module, enum, FakeEnums())
    monkeypatch.setitem(sys.modules, "vizdoom", module)
    return DoomCampaign("fake.wad", **kwargs), game


def _drain(env):
    """Step until the episode ends; returns every (obs, reward, done)."""
    steps = []
    while not steps or not steps[-1][2]:
        steps.append(env.step(0))
    return steps


def test_frames_are_rgb_in_the_unit_range(monkeypatch):
    env, _ = _campaign(monkeypatch, [_row(health=100)] * 3)
    obs = env.reset()
    assert obs.shape == (64, 64, 3) and obs.dtype == np.float32
    np.testing.assert_allclose(obs, 1 / 255.0, atol=1e-6)
    obs, _, done = env.step(0)
    assert not done
    np.testing.assert_allclose(obs, 2 / 255.0, atol=1e-6)


def test_frame_size_is_configurable(monkeypatch):
    env, _ = _campaign(monkeypatch, [_row()] * 2, frame_size=32)
    assert env.reset().shape == (32, 32, 3)


def test_the_terminal_observation_repeats_the_last_live_frame(monkeypatch):
    """The engine has dropped its buffer by then, at the exit exactly as
    at death, so this is all the wrapper can hand back. collect.py
    drops the row it belongs to."""
    env, _ = _campaign(monkeypatch, [_row()] * 2)
    env.reset()
    live, _, _ = env.step(0)
    repeat, _, done = env.step(0)
    assert done
    np.testing.assert_array_equal(live, repeat)


@pytest.mark.parametrize("ending", ["exit", "death", "timeout"])
def test_finished_reason_names_the_three_endings(monkeypatch, ending):
    # Two steps at action_repeat 4 put the engine clock at 8 tics, so a
    # cap of 8 is reached exactly when the timeout case wants it.
    timeout = 8 if ending == "timeout" else 4200
    env, _ = _campaign(monkeypatch, [_row()] * 2, ending=ending,
                       episode_timeout_tics=timeout)
    env.reset()
    assert env.finished_reason is None
    _drain(env)
    assert env.finished_reason == ending


def test_a_disabled_engine_cap_has_no_timeout_flavor(monkeypatch):
    """0 tics is vizdoom's "no limit", so an episode that ends alive can
    only have ended by leaving the map."""
    env, _ = _campaign(monkeypatch, [_row()] * 2, ending="timeout",
                       episode_timeout_tics=0)
    env.reset()
    _drain(env)
    assert env.finished_reason == "exit"


@pytest.mark.parametrize("ending, died", [("death", True), ("exit", False),
                                          ("timeout", False)])
def test_died_means_death_and_nothing_else(monkeypatch, ending, died):
    env, _ = _campaign(monkeypatch, [_row()] * 2, ending=ending,
                       episode_timeout_tics=8)
    env.reset()
    assert env.died is False
    _drain(env)
    assert env.died is died


def test_reward_prices_kills_items_and_damage(monkeypatch):
    rows = [_row(health=100),
            _row(health=90, killcount=1),
            _row(health=90, killcount=1, itemcount=2)]
    env, _ = _campaign(monkeypatch, rows, ending="death")
    env.reset()

    _, hurt_kill, _ = env.step(0)
    assert hurt_kill == pytest.approx(0.05 - 10 * 0.0005)
    assert env.last_reward_components == pytest.approx(
        {"exit": 0.0, "kill": 0.05, "item": 0.0, "damage": -0.005})

    _, picked, _ = env.step(0)
    assert picked == pytest.approx(0.02)
    assert env.last_reward_components["item"] == pytest.approx(0.02)


def test_standing_still_pays_nothing(monkeypatch):
    """No living reward: a corridor is not progress."""
    env, _ = _campaign(monkeypatch, [_row(health=100)] * 4, ending="death")
    env.reset()
    rewards = [reward for _, reward, _ in _drain(env)]
    assert rewards == [pytest.approx(0.0)] * len(rewards)


def test_only_the_exit_pays_the_bonus(monkeypatch):
    rows = [_row(health=100)] * 2
    exiting, _ = _campaign(monkeypatch, rows, ending="exit")
    exiting.reset()
    assert _drain(exiting)[-1][1] == pytest.approx(1.0)
    assert exiting.last_reward_components["exit"] == pytest.approx(1.0)

    dying, _ = _campaign(monkeypatch, rows, ending="death")
    dying.reset()
    assert _drain(dying)[-1][1] == pytest.approx(0.0)


def test_reward_constants_are_overridable(monkeypatch):
    rows = [_row(health=100), _row(health=100, killcount=1)]
    env, _ = _campaign(monkeypatch, rows, ending="exit", exit_bonus=5.0,
                       kill_reward=0.5)
    env.reset()
    assert env.step(0)[1] == pytest.approx(0.5)
    assert env.step(0)[1] == pytest.approx(5.0)


def test_the_vocabulary_is_twelve_slots_with_nine_live():
    assert DoomCampaign.action_dim == 12
    assert len(DoomCampaign.ACTIONS) == 12
    assert len(DoomCampaign._ACTION_BUTTONS) == 12
    assert DoomCampaign.available_actions.sum() == 9
    assert DoomCampaign.available_actions[:9].all()
    assert DoomCampaign.ACTIONS[9:] == ("back", "forward+attack",
                                        "back+attack")
    with pytest.raises(ValueError):
        DoomCampaign.available_actions[0] = False


def test_actions_press_the_buttons_they_are_named_after(monkeypatch):
    env, game = _campaign(monkeypatch, [_row()] * 5)
    env.reset()
    for index in (0, 3, 8, 11):
        env.step(index)
    pressed = [{name for name, held in zip(DoomCampaign._BUTTONS, row) if held}
               for row in game.buttons]
    assert pressed == [{"MOVE_FORWARD"},
                       {"MOVE_FORWARD", "TURN_LEFT"},
                       {"USE"},
                       {"MOVE_BACKWARD", "ATTACK"}]


def test_index_and_one_hot_actions_agree(monkeypatch):
    env, game = _campaign(monkeypatch, [_row()] * 4)
    env.reset()
    env.step(2)
    env.step(np.eye(12)[2])
    assert game.buttons[0] == game.buttons[1]


def test_rejects_malformed_actions(monkeypatch):
    env, _ = _campaign(monkeypatch, [_row()] * 4)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.zeros(3))
    with pytest.raises(ValueError):
        env.step(12)


def test_variables_follow_the_documented_order_and_repeat_at_the_end(
        monkeypatch):
    rows = [_row(position_x=1.0, health=100), _row(position_x=2.0, health=75)]
    env, _ = _campaign(monkeypatch, rows, ending="exit")
    env.reset()
    assert len(DoomCampaign.VARIABLE_NAMES) == len(env.variables())
    np.testing.assert_array_equal(env.variables(), rows[0])
    env.step(0)
    np.testing.assert_array_equal(env.variables(), rows[1])
    env.step(0)
    # Nothing left to read once the episode is over, so the last live
    # values repeat, exactly as the frame does.
    np.testing.assert_array_equal(env.variables(), rows[1])


def test_variables_hand_out_a_copy(monkeypatch):
    env, _ = _campaign(monkeypatch, [_row(health=100)] * 2)
    env.reset()
    env.variables()[:] = -1.0
    assert env.variables()[DoomCampaign.VARIABLE_NAMES.index("health")] == 100


def test_the_key_slots_are_reserved_at_the_end_of_the_row():
    assert DoomCampaign.VARIABLE_NAMES[-3:] == ("key_blue", "key_yellow",
                                                "key_red")
    assert DoomCampaign._GAME_VARIABLES[-3:] == ("USER1", "USER2", "USER3")
    assert len(DoomCampaign._GAME_VARIABLES) == len(
        DoomCampaign.VARIABLE_NAMES)


def test_the_engine_is_configured_headless_seeded_and_then_frozen(monkeypatch):
    env, game = _campaign(monkeypatch, [_row()] * 2, map_name="E1M3", skill=3,
                          episode_timeout_tics=700, seed=7)
    asked = dict(game.calls)
    assert asked["set_window_visible"] == (False,)
    assert asked["set_sound_enabled"] == (False,)
    assert asked["set_console_enabled"] == (False,)
    assert asked["set_screen_format"] == ("RGB24",)
    assert asked["set_screen_resolution"] == ("RES_160X120",)
    assert asked["set_mode"] == ("PLAYER",)
    assert asked["set_seed"] == (7,)
    assert asked["set_doom_game_path"] == ("fake.wad",)
    assert asked["set_doom_map"] == ("E1M3",)
    assert asked["set_doom_skill"] == (3,)
    assert asked["set_episode_timeout"] == (700,)
    # The status bar stays in frame: keys, health and ammo live there.
    assert asked["set_render_hud"] == (True,)
    assert asked["set_available_buttons"] == (list(DoomCampaign._BUTTONS),)
    assert asked["set_available_game_variables"] == (
        list(DoomCampaign._GAME_VARIABLES),)
    # init() freezes all of it, so it has to come last.
    names = [name for name, _ in game.calls]
    assert names[-1] == "init"


def test_no_seed_leaves_the_engine_unseeded(monkeypatch):
    _, game = _campaign(monkeypatch, [_row()] * 2)
    assert "set_seed" not in [name for name, _ in game.calls]


def test_close_is_idempotent_and_the_context_manager_closes(monkeypatch):
    env, game = _campaign(monkeypatch, [_row()] * 2)
    with env:
        env.reset()
    env.close()
    assert [name for name, _ in game.calls].count("close") == 1


def test_resize_frame_keeps_both_screen_formats_channel_last():
    """The helper take_cover and the campaign share: grayscale stays
    single-channel, RGB arrives channel-first and leaves channel-last
    with its channels in order."""
    gray = np.full((120, 160), 255, np.uint8)
    assert _resize_frame(gray, 32).shape == (32, 32, 1)
    np.testing.assert_allclose(_resize_frame(gray, 32), 1.0)
    # Some builds hand grayscale back with a leading channel axis.
    np.testing.assert_array_equal(_resize_frame(gray[None], 32),
                                  _resize_frame(gray, 32))

    rgb = np.zeros((3, 120, 160), np.uint8)
    rgb[0], rgb[1], rgb[2] = 255, 128, 0
    small = _resize_frame(rgb, 16)
    assert small.shape == (16, 16, 3) and small.dtype == np.float32
    np.testing.assert_allclose(small[..., 0], 1.0)
    np.testing.assert_allclose(small[..., 1], 128 / 255.0, atol=1e-6)
    np.testing.assert_allclose(small[..., 2], 0.0)


def test_action_index_accepts_indices_and_one_hots():
    assert _action_index(2, 12) == 2
    assert _action_index(np.eye(12)[5], 12) == 5
    assert _action_index(np.float32(3), 12) == 3
    with pytest.raises(ValueError):
        _action_index(np.zeros(11), 12)
    with pytest.raises(ValueError):
        _action_index(-1, 12)
    with pytest.raises(ValueError):
        _action_index(12, 12)
