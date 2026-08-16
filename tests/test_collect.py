import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_models.collect import (RandomPolicy, RSSMPolicy, ScriptedExplorer,
                                  collect_episode)
from world_models.envs.doom import DoomCampaign
from world_models.models.rssm import RSSM
from world_models.replay import ReplayBuffer


class FakeEnv:
    """Scripted env, no vizdoom involved: frame value = offset + step
    index, so a stored uint8 frame tells you exactly which step and
    which env instance produced it. die_at=None never ends the episode
    on its own; max_steps (a timeout) is what stops collect_episode.

    Unlike the real engine this keeps rendering a fresh frame on the
    killing step, which is the point: if collect_episode ever stored
    that step, its value would show up in the frame sequence."""

    action_dim = 3

    def __init__(self, die_at=None, size=4, channels=1, offset=0):
        self.die_at = die_at
        self.size = size
        self.channels = channels
        self.offset = offset
        self._t = 0
        self._died = False

    def reset(self):
        self._t = 0
        self._died = False
        return self._frame()

    def step(self, action):
        self._t += 1
        done = self.die_at is not None and self._t >= self.die_at
        self._died = done
        return self._frame(), 0.01, done

    @property
    def died(self):
        return self._died

    def _frame(self):
        v = (self.offset + self._t) / 255.0
        return np.full((self.size, self.size, self.channels), v, np.float32)


class FakeCampaignEnv:
    """Campaign-shaped env with no engine behind it.

    It speaks DoomCampaign's vocabulary (borrowed, so the two cannot
    drift apart) and copies the one engine behavior the collector's
    contract hangs on: an episode that ends gets no new frame and no
    new variables, whichever way it ends, because the engine has thrown
    its state away by then. Frames and variables carry the index of the
    step that rendered them, so a stored row can be traced back to it
    and a repeat is visible as a repeat.

    reason picks the ending reached at ends_at steps; ends_at None
    never ends, leaving max_steps to stop the episode.
    """

    ACTIONS = DoomCampaign.ACTIONS
    VARIABLE_NAMES = DoomCampaign.VARIABLE_NAMES
    action_dim = DoomCampaign.action_dim

    def __init__(self, reason="exit", ends_at=None, step_reward=0.02,
                 exit_reward=1.0, size=4):
        self.reason = reason
        self.ends_at = ends_at
        self.step_reward = step_reward
        self.exit_reward = exit_reward
        self.size = size
        self._t = 0
        self._live = 0
        self._done = False

    def reset(self):
        self._t = self._live = 0
        self._done = False
        return self._frame()

    def step(self, action):
        self._t += 1
        self._done = self.ends_at is not None and self._t >= self.ends_at
        if self._done:
            reward = self.exit_reward if self.reason == "exit" else 0.0
            return self._frame(), reward, True
        self._live = self._t
        return self._frame(), self.step_reward, False

    @property
    def finished_reason(self):
        return self.reason if self._done else None

    @property
    def died(self):
        return self._done and self.reason == "death"

    def variables(self):
        return np.full(len(self.VARIABLE_NAMES), float(self._live), np.float32)

    def _frame(self):
        v = self._live / 255.0
        return np.full((self.size, self.size, 3), v, np.float32)


def test_campaign_exit_folds_the_bonus_onto_the_last_real_frame():
    """The exit step's frame is a repeat, so its row is dropped and its
    bonus moves onto the row before it, which is a real frame showing
    the exit. The terminal label goes there too."""
    env = FakeCampaignEnv(reason="exit", ends_at=5, step_reward=0.02,
                          exit_reward=1.0)
    ep = collect_episode(env, RandomPolicy(env.action_dim, seed=0), 20)

    assert ep["obs"].shape == (5, 4, 4, 3)
    np.testing.assert_array_equal(ep["obs"][:, 0, 0, 0], np.arange(5))
    assert ep["terminated"] is True
    assert ep["reason"] == "exit"
    assert ep["reward"][0] == pytest.approx(0.0)
    np.testing.assert_allclose(ep["reward"][1:-1], 0.02, rtol=1e-6)
    assert ep["reward"][-1] == pytest.approx(0.02 + 1.0)
    assert float(ep["reward"].sum()) == pytest.approx(4 * 0.02 + 1.0)


def test_campaign_death_drops_the_repeat_and_pays_nothing_extra():
    env = FakeCampaignEnv(reason="death", ends_at=5, step_reward=0.02)
    ep = collect_episode(env, RandomPolicy(env.action_dim, seed=0), 20)

    assert ep["obs"].shape == (5, 4, 4, 3)
    np.testing.assert_array_equal(ep["obs"][:, 0, 0, 0], np.arange(5))
    assert ep["terminated"] is True and ep["reason"] == "death"
    assert ep["reward"][-1] == pytest.approx(0.02)


def test_campaign_timeout_stores_its_final_row_unterminated():
    """A timeout is the clock, not the world ending, so nothing is
    dropped and nothing is labelled. The stored final row repeats the
    frame before it, which costs nothing: its continue target is 1
    either way, and dropping it would lose that step's reward."""
    env = FakeCampaignEnv(reason="timeout", ends_at=5, step_reward=0.02)
    ep = collect_episode(env, RandomPolicy(env.action_dim, seed=0), 20)

    assert ep["obs"].shape == (6, 4, 4, 3)
    np.testing.assert_array_equal(ep["obs"][:, 0, 0, 0],
                                  [0, 1, 2, 3, 4, 4])
    assert ep["terminated"] is False and ep["reason"] == "timeout"


def test_campaign_step_cap_is_a_timeout_too():
    """max_steps is the collector's own clock; the episode never ended."""
    env = FakeCampaignEnv(reason="exit", ends_at=None)
    ep = collect_episode(env, RandomPolicy(env.action_dim, seed=0), 4)

    assert ep["obs"].shape == (5, 4, 4, 3)
    assert ep["terminated"] is False and ep["reason"] == "timeout"


@pytest.mark.parametrize("reason, rows", [("exit", 5), ("death", 5),
                                          ("timeout", 6)])
def test_campaign_records_one_variable_row_per_stored_frame(reason, rows):
    env = FakeCampaignEnv(reason=reason, ends_at=5)
    ep = collect_episode(env, RandomPolicy(env.action_dim, seed=0), 20)

    n_vars = len(FakeCampaignEnv.VARIABLE_NAMES)
    assert ep["variables"].shape == (rows, n_vars)
    assert ep["variables"].dtype == np.float32
    # Same length as obs, and the same step indices: variables that
    # belong to a dropped row are dropped with it.
    np.testing.assert_array_equal(ep["variables"][:, 0],
                                  ep["obs"][:, 0, 0, 0].astype(np.float32))


def test_legacy_envs_produce_exactly_the_dict_they_always_did():
    """The guard on every extension: an env that knows nothing about
    finished_reason or variables gets the old four keys and the old
    living-reward pin, byte for byte."""
    ep = collect_episode(FakeEnv(die_at=5), RandomPolicy(3, seed=0), 20)
    assert set(ep) == {"obs", "action", "reward", "terminated"}
    assert ep["reward"][0] == pytest.approx(0.01)


class FakeCorridor:
    """A corridor with a wall at the end, for the explorer's bump rule.

    Only position matters here, so that is all this simulates: forward
    advances until the wall, turning frees the way (the wall moves
    back, which is what turning down a side corridor amounts to), and
    everything else stands still. Actions arrive as indices and are
    applied by the test loop, since a policy does not step its env.
    """

    ACTIONS = DoomCampaign.ACTIONS
    VARIABLE_NAMES = DoomCampaign.VARIABLE_NAMES
    action_dim = DoomCampaign.action_dim

    def __init__(self, wall_at=90.0, stride=30.0):
        self.wall_at = wall_at
        self.stride = stride
        self.x = 0.0
        self.taken = []

    def variables(self):
        v = np.zeros(len(self.VARIABLE_NAMES), np.float32)
        v[self.VARIABLE_NAMES.index("position_x")] = self.x
        return v

    def apply(self, index):
        name = self.ACTIONS[index]
        self.taken.append(name)
        if name == "forward" and self.x + self.stride <= self.wall_at:
            self.x += self.stride
        elif name.startswith("turn"):
            self.wall_at += 3 * self.stride


def _explore(env, explorer, steps):
    explorer.reset(None)
    for _ in range(steps):
        env.apply(explorer(None))


def test_explorer_turns_when_it_bumps_and_then_walks_on():
    env = FakeCorridor()
    explorer = ScriptedExplorer(env, seed=0, attack_prob=0.0, use_prob=0.0)
    _explore(env, explorer, 40)

    assert "turn_left" in env.taken or "turn_right" in env.taken
    # It got past the wall it started against, several times over: the
    # bump rule fired and the walk resumed after each turn.
    assert env.x > 90.0


def test_explorer_never_leaves_the_enabled_vocabulary():
    env = FakeCorridor()
    explorer = ScriptedExplorer(env, seed=1)
    _explore(env, explorer, 300)

    enabled = set(np.asarray(DoomCampaign.ACTIONS)[DoomCampaign.available_actions])
    assert set(env.taken) <= enabled


def test_explorer_presses_attack_and_use_only_when_asked_to():
    quiet = FakeCorridor()
    _explore(quiet, ScriptedExplorer(quiet, seed=2, attack_prob=0.0,
                                     use_prob=0.0), 200)
    assert not {"attack", "use"} & set(quiet.taken)

    noisy = FakeCorridor()
    _explore(noisy, ScriptedExplorer(noisy, seed=2), 400)
    assert {"attack", "use"} <= set(noisy.taken)


def test_explorer_is_deterministic_in_its_seed():
    def run(seed):
        env = FakeCorridor()
        _explore(env, ScriptedExplorer(env, seed=seed), 100)
        return env.taken

    assert run(5) == run(5)
    assert run(5) != run(6)


def test_death_episode_contract():
    env = FakeEnv(die_at=5)
    ep = collect_episode(env, RandomPolicy(env.action_dim, seed=0), 20)

    # Five steps taken, five rows stored: the reset frame plus the four
    # steps that were survived. The killing step is dropped, so a death
    # episode stores exactly die_at frames.
    assert ep["obs"].shape == (5, 4, 4, 1) and ep["obs"].dtype == np.uint8
    assert ep["action"].shape == (5, 3) and ep["action"].dtype == np.float32
    assert ep["reward"].shape == (5,) and ep["reward"].dtype == np.float32
    assert ep["reward"][0] == pytest.approx(0.01)
    np.testing.assert_array_equal(ep["action"][0], 0.0)
    assert ep["terminated"] is True
    # frame values round-trip to the step index that produced them, and
    # step 5 — the one that killed — is absent
    np.testing.assert_array_equal(ep["obs"][:, 0, 0, 0], np.arange(5))


def test_truncation_episode_contract():
    env = FakeEnv(die_at=None)
    ep = collect_episode(env, RandomPolicy(env.action_dim, seed=0), 7)

    assert ep["obs"].shape == (8, 4, 4, 1)
    assert ep["action"].shape == (8, 3)
    assert ep["reward"].shape == (8,)
    assert ep["reward"][0] == pytest.approx(0.01)
    np.testing.assert_array_equal(ep["action"][0], 0.0)
    assert ep["terminated"] is False
    np.testing.assert_array_equal(ep["obs"][:, 0, 0, 0], np.arange(8))


def test_death_and_truncation_give_correct_continue_targets():
    """Round-trip through ReplayBuffer: the death episode's last frame
    is the only 0.0 continue target anywhere, and it is the last frame
    the env actually rendered; the timeout stays all ones. die_at=5
    stores 5 frames and 4 timeout steps store 5 too, so transitions=4
    leaves exactly one valid start each and every sample is a whole
    episode."""
    policy = RandomPolicy(3, seed=1)
    death_ep = collect_episode(FakeEnv(die_at=5, offset=0), policy, 10)
    timeout_ep = collect_episode(FakeEnv(die_at=None, offset=200), policy, 4)
    assert death_ep["obs"].shape[0] == timeout_ep["obs"].shape[0] == 5

    buffer = ReplayBuffer(capacity=1000)
    buffer.add_episode(death_ep["obs"], death_ep["action"],
                       death_ep["reward"], death_ep["terminated"])
    buffer.add_episode(timeout_ep["obs"], timeout_ep["action"],
                       timeout_ep["reward"], timeout_ep["terminated"])

    rng = np.random.default_rng(0)
    obs, _, _, cont = buffer.sample_sequences_with_continues(rng, 64, 4)
    assert cont.shape == (5, 64)
    is_death = np.asarray(obs)[0, :, 0, 0, 0] < 0.5  # offset 0 vs 200/255
    cont = np.asarray(cont)
    np.testing.assert_array_equal(cont[:4], 1.0)
    np.testing.assert_array_equal(cont[4, is_death], 0.0)
    np.testing.assert_array_equal(cont[4, ~is_death], 1.0)
    assert is_death.any() and (~is_death).any()


def test_random_policy_is_deterministic_in_its_seed():
    action_dim = FakeEnv.action_dim
    ep1 = collect_episode(FakeEnv(), RandomPolicy(action_dim, seed=7), 15)
    ep2 = collect_episode(FakeEnv(), RandomPolicy(action_dim, seed=7), 15)
    np.testing.assert_array_equal(ep1["action"], ep2["action"])

    ep3 = collect_episode(FakeEnv(), RandomPolicy(action_dim, seed=8), 15)
    assert not np.array_equal(ep1["action"], ep3["action"])


def _tiny_rssm(action_dim: int):
    model = RSSM(latent_dim=4, hidden=8, action_dim=action_dim,
                obs_channels=1, obs_size=32)
    params = model.init(
        jax.random.PRNGKey(0), jnp.zeros((1, 32, 32, 1)),
        model.initial_state(1), jnp.zeros((1, 4)),
        jnp.zeros((1, action_dim)),
    )
    return model, params


def _constant_one_hot(action_dim: int, index: int):
    """Ignores params, s and key: a stub standing in for a trained actor."""
    def act_fn(actor_params, s, key):
        return jnp.zeros(action_dim).at[index].set(1.0)
    return act_fn


def _policy(model, params, act_fn):
    policy = RSSMPolicy(model, act_fn, jax.random.PRNGKey(0), 3)
    policy.set_params(params)
    return policy


def test_rssm_policy_runs_and_is_deterministic():
    model, params = _tiny_rssm(action_dim=3)
    act_fn = _constant_one_hot(3, 1)

    ep1 = collect_episode(
        FakeEnv(size=32, channels=1),
        _policy(model, params, act_fn),
        6,
    )
    assert ep1["obs"].shape == (7, 32, 32, 1)
    assert ep1["action"].shape == (7, 3)
    # actions arrive in env-acceptable (one-hot vector) form
    np.testing.assert_array_equal(
        ep1["action"][1:], np.tile([0.0, 1.0, 0.0], (6, 1)))

    ep2 = collect_episode(
        FakeEnv(size=32, channels=1),
        _policy(model, params, act_fn),
        6,
    )
    np.testing.assert_array_equal(ep1["action"], ep2["action"])
    np.testing.assert_array_equal(ep1["obs"], ep2["obs"])
