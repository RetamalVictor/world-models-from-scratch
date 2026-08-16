"""ViZDoom take_cover as a plain Python environment.

This is the CPU boundary. Everything else in the repo is jittable JAX;
the Doom engine is a stateful C++ process behind a Python handle, so
this wrapper is deliberately a stateful, non-jittable class and the
training loop pays a host round-trip once per policy step. What comes
back is the ball envs' convention — (frame_size, frame_size, 1) float32
in [0, 1] — so replay and the RSSM cannot tell the difference.

The scenario, its config and its WAD all ship inside the vizdoom pip
package (uv sync --extra doom): nothing is bought or downloaded. Its
living_reward is ignored on purpose. Reward is computed here, +1 per
policy step survived scaled by 1/100, so the reward head's MSE sits in
the range the ball taught us to read; a config file is the wrong place
to keep a number the loss depends on.

Two environments live here. DoomTakeCover is that scenario.
DoomCampaign plays the real game, one map of an IWAD per episode, and
is what campaign training collects from. They share only the engine
boilerplate, through the module-level helpers below: everything the
learning problem is made of (frames, actions, reward, terminal
flavors) differs between them, deliberately. vizdoom is imported
inside functions rather than at module scope, so this module imports
without the extra installed and the parts that are plain Python stay
testable on a machine that has no engine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _configure_headless(game, screen_format, resolution, seed) -> None:
    """Window off, sound off, console off, player mode, screen, seed.

    init() freezes every one of these: setting any of them afterwards
    is ignored, and a seed set after init() does not take, so both
    wrappers call this once, before init(), and never again.
    """
    import vizdoom as vzd

    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_console_enabled(False)
    game.set_screen_format(screen_format)
    game.set_screen_resolution(resolution)
    game.set_mode(vzd.Mode.PLAYER)
    if seed is not None:
        game.set_seed(seed)


def _action_index(action, action_dim: int) -> int:
    """A scalar index passes through, a one-hot becomes its argmax.

    Policies hand back either form (RandomPolicy an int, RSSMPolicy a
    vector), so the wrappers accept both and reject anything else here
    instead of indexing something meaningless. collect.py's
    _as_action_vector is the inverse, for storage.
    """
    a = np.asarray(action)
    if a.ndim == 0:
        index = int(a)
    elif a.shape == (action_dim,):
        index = int(a.argmax())
    else:
        raise ValueError(
            f"action must be an index or a length-{action_dim} "
            f"one-hot, got shape {a.shape}"
        )
    if not 0 <= index < action_dim:
        raise ValueError(f"action index {index} is out of range")
    return index


def _resize_frame(screen_buffer, frame_size: int) -> np.ndarray:
    """Engine screen buffer to (frame_size, frame_size, C) float32 in [0, 1].

    GRAY8 arrives as (H, W), or with a leading singleton axis on some
    builds; RGB24 arrives channel-first, (3, H, W). Both leave here
    channel-last, which is the convention replay, the encoder and the
    ball envs all speak.
    """
    buf = np.asarray(screen_buffer)
    if buf.ndim == 3 and buf.shape[0] == 3:
        buf = buf.transpose(1, 2, 0)
    else:
        buf = np.squeeze(buf)
    small = Image.fromarray(buf).resize(
        (frame_size, frame_size), Image.BILINEAR
    )
    frame = np.asarray(small, np.float32) / 255.0
    return frame if frame.ndim == 3 else frame[:, :, None]


class DoomTakeCover:
    """take_cover: strafe left or right, dodge fireballs, survive.

    Actions are {move_left, move_right, noop}, accepted as an index or
    a length-3 one-hot. Each policy step holds the button for
    action_repeat engine tics. The episode ends on death, and the
    engine discards its screen buffer when it does, so the terminal
    step can only repeat the last live frame. That repeat is an
    artifact of this wrapper, not something the world rendered:
    collect.py drops it and labels death on the last real observation
    instead, so the continue head keys on what a frame shows — the
    fireball about to land — rather than on the picture freezing. See
    collect_episode for why that distinction decides whether the head
    is any use inside a dream.

    Determinism: the engine replays exactly when the seed is set before
    init() and it is fed back the same action sequence. Nothing weaker
    holds; a different action sequence diverges immediately, and a seed
    set after init() does not take.
    """

    action_dim = 3

    def __init__(self, frame_size: int = 64, action_repeat: int = 4,
                 seed: int | None = None):
        import vizdoom as vzd

        self.frame_size = frame_size
        self.action_repeat = action_repeat
        game = vzd.DoomGame()
        game.load_config(str(Path(vzd.scenarios_path) / "take_cover.cfg"))
        _configure_headless(game, vzd.ScreenFormat.GRAY8,
                            vzd.ScreenResolution.RES_160X120, seed)
        game.init()
        self._game = game
        # In take_cover.cfg's button order: MOVE_LEFT, MOVE_RIGHT.
        self._buttons = ([1, 0], [0, 1], [0, 0])
        self._last_obs = np.zeros((frame_size, frame_size, 1), np.float32)

    def reset(self) -> np.ndarray:
        self._game.new_episode()
        self._last_obs = self._frame()
        return self._last_obs

    def step(self, action) -> tuple[np.ndarray, float, bool]:
        """-> (obs, reward, done). The step that kills you pays nothing,
        and its obs is the previous frame again — the buffer is gone by
        then. collect.py discards that row rather than store it."""
        self._game.make_action(
            self._buttons[_action_index(action, self.action_dim)],
            self.action_repeat)
        done = self._game.is_episode_finished()
        if not done:
            self._last_obs = self._frame()
        return self._last_obs, 0.0 if done else 0.01, done

    @property
    def died(self) -> bool:
        """True iff the episode is over and it ended in death.

        take_cover's own timeout is off by default (episode_timeout=0
        in its cfg) so death is currently the only way an episode ends,
        but the distinction matters for the replay buffer's continue
        targets (see replay.py), so it is checked rather than assumed.
        False before any episode has run and right after every reset.
        """
        return self._game.is_episode_finished() and self._game.is_player_dead()

    def close(self):
        if self._game is not None:
            self._game.close()
            self._game = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def _frame(self) -> np.ndarray:
        return _resize_frame(self._game.get_state().screen_buffer,
                             self.frame_size)


class DoomCampaign:
    """One map of the real game, played from pixels: the campaign env.

    take_cover is a training scenario with a WAD in the pip package.
    This is Doom itself: an IWAD on disk (shareware doom1.wad, or
    freedoom1.wad where redistribution terms have to stay boring), one
    map per episode, pistol start, skill 1 by default. The engine
    config is built here in code rather than loaded from a .cfg, for
    the same reason the reward is authored here: the numbers the
    learning problem is made of belong next to the code that explains
    them.

    Observation: (frame_size, frame_size, 3) float32 in [0, 1], RGB,
    status bar rendered and nothing cropped. Both are load-bearing.
    Keycards and the doors they open are color coded, so grayscale
    erases the one bit that gates progress through a map, and the HUD
    turns keys, health and ammo into pixels, which makes them
    perception instead of memory.

    Actions: a fixed 12-slot vocabulary (ACTIONS), the first nine
    enabled (available_actions). The one-hot dimension is 12 from the
    first collected frame, in replay, in the RSSM's action input and in
    the actor's head, so switching a reserved slot on later is a
    config flip: no checkpoint migration, no dataset
    schema change, no shape change anywhere, and every frame collected
    before it stays valid, having simply never used that slot. Do not
    shrink this to the nine that are live.

    Reward is computed here from game-variable deltas, never from the
    engine's own scoring: the exit bonus for finishing the map, a small
    payment per kill and per item, a small charge per point of health
    lost. There is no living reward, because standing in a corridor is
    not progress. The constants are constructor arguments so a run
    config can override them, and the per-step breakdown stays readable
    through last_reward_components so a dataset report can say what the
    reward was actually made of.

    Terminal flavors: finished_reason is None while the episode runs,
    then "death", "exit" or "timeout". They need different labels in
    the buffer, so they are read from the engine rather than inferred.
    died keeps its narrow meaning, death only, because that is what the
    protocol promised.

    The exit gets the death treatment. The engine discards its screen
    buffer at map end exactly as it does at death, so the exiting step
    can only hand back a repeat of the last live frame, and that repeat
    is an artifact of this wrapper rather than something the world
    rendered. collect.py drops it, puts the terminal label on the last
    real observation (which shows the exit switch or door up close, a
    cue the prior can actually dream) and folds this step's exit bonus
    into that stored row. Two costs come with that, both accepted: the
    bonus is paid one step early, attached to standing at the switch
    rather than to pressing use at it, and the exit-triggering action
    is lost with the row, the same way the killing action already is.
    The alternative stores a duplicated frame against a zero continue
    target and teaches the head "exit means the picture froze", which
    is the death-label mistake again with a positive sign. See
    collect_episode for the full reasoning.

    Determinism: as in DoomTakeCover, the engine replays exactly when
    the seed is set before init() and it is fed the same action
    sequence, and nothing weaker holds.
    """

    action_dim = 12

    ACTIONS = (
        "forward", "turn_left", "turn_right", "forward+turn_left",
        "forward+turn_right", "strafe_left", "strafe_right", "attack",
        "use", "back", "forward+attack", "back+attack",
    )
    # The last three slots are reserved: present in every shape, never
    # chosen, waiting for retreat fire to earn its place.
    available_actions = np.array([True] * 9 + [False] * 3)
    available_actions.flags.writeable = False

    # Button names rather than vizdoom enums, so the vocabulary is
    # readable without the engine installed; resolved at construction.
    _BUTTONS = ("MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT",
                "MOVE_LEFT", "MOVE_RIGHT", "ATTACK", "USE")
    _ACTION_BUTTONS = (
        ("MOVE_FORWARD",),
        ("TURN_LEFT",),
        ("TURN_RIGHT",),
        ("MOVE_FORWARD", "TURN_LEFT"),
        ("MOVE_FORWARD", "TURN_RIGHT"),
        ("MOVE_LEFT",),
        ("MOVE_RIGHT",),
        ("ATTACK",),
        ("USE",),
        ("MOVE_BACKWARD",),
        ("MOVE_FORWARD", "ATTACK"),
        ("MOVE_BACKWARD", "ATTACK"),
    )

    VARIABLE_NAMES = (
        "position_x", "position_y", "position_z", "angle", "health",
        "armor", "killcount", "itemcount", "selected_weapon",
        "selected_weapon_ammo", "bullets", "shells", "rockets", "cells",
        "key_blue", "key_yellow", "key_red",
    )
    # AMMOn is "ammo of the weapon in slot n": pistol, shotgun, rocket
    # launcher, plasma rifle. The three key slots are reserved the way
    # the reserved actions are: the engine has no keycard variable, ACS
    # in a custom WAD is the only source, and USER1 to USER3 are where
    # such a script would publish them. Until one exists they read 0,
    # and the HUD, which is in frame, is where key state is actually
    # perceived. Recording them now keeps the array's width fixed, so
    # the day a script lands nothing about the dataset changes.
    _GAME_VARIABLES = (
        "POSITION_X", "POSITION_Y", "POSITION_Z", "ANGLE", "HEALTH",
        "ARMOR", "KILLCOUNT", "ITEMCOUNT", "SELECTED_WEAPON",
        "SELECTED_WEAPON_AMMO", "AMMO2", "AMMO3", "AMMO5", "AMMO6",
        "USER1", "USER2", "USER3",
    )
    _HEALTH = VARIABLE_NAMES.index("health")
    _KILLCOUNT = VARIABLE_NAMES.index("killcount")
    _ITEMCOUNT = VARIABLE_NAMES.index("itemcount")

    def __init__(self, wad_path: str | Path, map_name: str = "E1M1",
                 frame_size: int = 64, action_repeat: int = 4,
                 skill: int = 1, episode_timeout_tics: int = 4200,
                 seed: int | None = None, exit_bonus: float = 1.0,
                 kill_reward: float = 0.05, item_reward: float = 0.01,
                 damage_penalty: float = 0.0005):
        """wad_path is an IWAD: the game itself, not a scenario on top.

        episode_timeout_tics is the engine's backstop, not the usual
        clock: at 35 tics per second and action_repeat 4 the default is
        two minutes of game time, about a thousand policy steps, and
        the collector's own step cap normally fires first. Passing 0
        disables it, which is vizdoom's convention, and then no episode
        can end with the "timeout" flavor.
        """
        import vizdoom as vzd

        self.frame_size = frame_size
        self.action_repeat = action_repeat
        self.map_name = map_name
        self.skill = skill
        self.episode_timeout_tics = episode_timeout_tics
        self.exit_bonus = exit_bonus
        self.kill_reward = kill_reward
        self.item_reward = item_reward
        self.damage_penalty = damage_penalty

        game = vzd.DoomGame()
        game.set_doom_game_path(str(wad_path))
        game.set_doom_map(map_name)
        game.set_doom_skill(skill)
        game.set_episode_timeout(episode_timeout_tics)
        game.set_available_buttons(
            [getattr(vzd.Button, name) for name in self._BUTTONS])
        game.set_available_game_variables(
            [getattr(vzd.GameVariable, name) for name in self._GAME_VARIABLES])
        # The status bar stays: it is where keys, health and ammo are.
        game.set_render_hud(True)
        _configure_headless(game, vzd.ScreenFormat.RGB24,
                            vzd.ScreenResolution.RES_160X120, seed)
        game.init()
        self._game = game

        self._buttons = tuple(
            [1 if name in pressed else 0 for name in self._BUTTONS]
            for pressed in self._ACTION_BUTTONS
        )
        self._last_obs = np.zeros((frame_size, frame_size, 3), np.float32)
        self._last_variables = np.zeros(len(self.VARIABLE_NAMES), np.float32)
        self._components = {"exit": 0.0, "kill": 0.0, "item": 0.0,
                            "damage": 0.0}

    def reset(self) -> np.ndarray:
        self._game.new_episode()
        self._components = dict.fromkeys(self._components, 0.0)
        self._read_state()
        return self._last_obs

    def step(self, action) -> tuple[np.ndarray, float, bool]:
        """-> (obs, reward, done).

        The step that ends the map, whichever way it ends, gets no new
        observation and no new variables: the engine has dropped its
        state by then, so both repeat. Only the exit bonus can be paid
        on such a step, since every other component is a delta and the
        deltas are all zero against a repeat. collect.py knows which
        rows are repeats and what to do with each (see the class
        docstring).
        """
        previous = self._last_variables
        self._game.make_action(
            self._buttons[_action_index(action, self.action_dim)],
            self.action_repeat)
        self._read_state()
        done = self._game.is_episode_finished()
        return self._last_obs, self._reward(previous), done

    @property
    def finished_reason(self) -> str | None:
        """None while the episode runs, else "death", "exit" or "timeout".

        Exit is what the campaign is about and death is the world
        ending, so both are terminal and both get a zero continue
        target; a timeout is only a clock running out, which is why
        replay leaves its continue target at 1. Exit is the fallback
        case rather than a positive test because the engine reports no
        "you finished the map" flag: it reports that the episode is
        over, that the player is alive, and how many tics have passed.
        """
        if not self._game.is_episode_finished():
            return None
        if self._game.is_player_dead():
            return "death"
        if (self.episode_timeout_tics
                and self._game.get_episode_time() >= self.episode_timeout_tics):
            return "timeout"
        return "exit"

    @property
    def died(self) -> bool:
        """True iff the episode is over and it ended in death.

        Narrower than finished_reason on purpose: this is the protocol
        every other env in the repo speaks, and an exit is not a death.
        """
        return self.finished_reason == "death"

    @property
    def last_reward_components(self) -> dict:
        """What the last step's reward was made of, by component.

        A copy: dataset reports accumulate these across a run and have
        no business reaching into the wrapper's state to do it. Zeroed
        by reset, so reading it before the first step is meaningful.
        """
        return dict(self._components)

    def variables(self) -> np.ndarray:
        """Recorded game state, VARIABLE_NAMES order, float32.

        A copy of the wrapper's last live read. Once the episode is
        over the engine has no state left to ask, so this repeats
        exactly as the observation does, and the collector drops both
        together on the rows it drops.
        """
        return self._last_variables.copy()

    def close(self):
        if self._game is not None:
            self._game.close()
            self._game = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def _read_state(self) -> None:
        """Pull frame and variables, or keep the last live pair."""
        state = self._game.get_state()
        if state is None:
            return
        self._last_obs = _resize_frame(state.screen_buffer, self.frame_size)
        self._last_variables = np.asarray(state.game_variables, np.float32)

    def _reward(self, previous: np.ndarray) -> float:
        """Deltas since the previous step, priced.

        Health gains are not charged for and not paid for: a medkit is
        a pickup, and itemcount has already paid for it. Kills and
        items are counters, so their deltas are clamped at zero rather
        than trusted to be monotonic across an engine reset.
        """
        current = self._last_variables
        self._components = {
            "exit": (self.exit_bonus if self.finished_reason == "exit"
                     else 0.0),
            "kill": self.kill_reward * max(
                float(current[self._KILLCOUNT] - previous[self._KILLCOUNT]),
                0.0),
            "item": self.item_reward * max(
                float(current[self._ITEMCOUNT] - previous[self._ITEMCOUNT]),
                0.0),
            "damage": -self.damage_penalty * max(
                float(previous[self._HEALTH] - current[self._HEALTH]), 0.0),
        }
        return float(sum(self._components.values()))
