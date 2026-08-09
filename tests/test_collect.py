import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_models.collect import RandomPolicy, RSSMPolicy, collect_episode
from world_models.models.rssm import RSSM
from world_models.replay import ReplayBuffer


class FakeEnv:
    """Scripted env, no vizdoom involved: frame value = offset + step
    index, so a stored uint8 frame tells you exactly which step and
    which env instance produced it. die_at=None never ends the episode
    on its own; max_steps (a timeout) is what stops collect_episode."""

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


def test_death_episode_contract():
    env = FakeEnv(die_at=5)
    ep = collect_episode(env, RandomPolicy(env.action_dim, seed=0), 20)

    assert ep["obs"].shape == (6, 4, 4, 1) and ep["obs"].dtype == np.uint8
    assert ep["action"].shape == (6, 3) and ep["action"].dtype == np.float32
    assert ep["reward"].shape == (6,) and ep["reward"].dtype == np.float32
    assert ep["reward"][0] == pytest.approx(0.01)
    np.testing.assert_array_equal(ep["action"][0], 0.0)
    assert ep["terminated"] is True
    # frame values round-trip to the step index that produced them
    np.testing.assert_array_equal(ep["obs"][:, 0, 0, 0], np.arange(6))


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
    is the only 0.0 continue target anywhere; the timeout stays all
    ones. Both episodes are 5 frames long and transitions=4 leaves
    exactly one valid start each, so every sample is a whole episode."""
    policy = RandomPolicy(3, seed=1)
    death_ep = collect_episode(FakeEnv(die_at=4, offset=0), policy, 10)
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
