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
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class DoomTakeCover:
    """take_cover: strafe left or right, dodge fireballs, survive.

    Actions are {move_left, move_right, noop}, accepted as an index or
    a length-3 one-hot. Each policy step holds the button for
    action_repeat engine tics. The episode ends on death, and the
    engine discards its screen buffer when it does, so the terminal
    step repeats the last live frame — the continue head, not the
    pixels, is what tells the model that one apart.

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
        # Headless, silent, and seeded: init() freezes all of it, and
        # setting any of these afterwards is ignored.
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_console_enabled(False)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_mode(vzd.Mode.PLAYER)
        if seed is not None:
            game.set_seed(seed)
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
        """-> (obs, reward, done). The step that kills you pays nothing."""
        self._game.make_action(self._buttons[self._index(action)],
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

    def _index(self, action) -> int:
        a = np.asarray(action)
        if a.ndim == 0:
            index = int(a)
        elif a.shape == (self.action_dim,):
            index = int(a.argmax())
        else:
            raise ValueError(
                f"action must be an index or a length-{self.action_dim} "
                f"one-hot, got shape {a.shape}"
            )
        if not 0 <= index < self.action_dim:
            raise ValueError(f"action index {index} is out of range")
        return index

    def _frame(self) -> np.ndarray:
        # GRAY8 gives (H, W) here; some builds add a leading channel axis.
        buf = np.squeeze(np.asarray(self._game.get_state().screen_buffer))
        small = Image.fromarray(buf).resize(
            (self.frame_size, self.frame_size), Image.BILINEAR
        )
        return (np.asarray(small, np.float32) / 255.0)[:, :, None]
