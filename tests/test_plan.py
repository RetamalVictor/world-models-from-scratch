import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import unfreeze

from world_models.collect import RSSMPolicy, collect_episode
from world_models.imagination import imagine_rollout
from world_models.models.actor_critic import (Critic, DiscreteActor,
                                              lambda_returns)
from world_models.models.rssm import RSSM
from world_models.plan import MPCConfig, MPCPolicy, make_planner, score_rollout


def _tiny_rssm(action_dim=3, hidden=8, latent=4, seed=0):
    model = RSSM(latent_dim=latent, hidden=hidden, action_dim=action_dim,
                 obs_channels=1, obs_size=32, predict_continue=True)
    params = model.init(
        jax.random.PRNGKey(seed), jnp.zeros((1, 32, 32, 1)),
        model.initial_state(1), jnp.zeros((1, latent)),
        jnp.zeros((1, action_dim)),
    )
    return model, params


def _heads(action_dim=3, s_dim=12, seed=1):
    """A real actor and critic, randomly initialized: the planner talks
    to them only through apply, which is how an eval script wires them."""
    actor = DiscreteActor(action_dim=action_dim)
    critic = Critic()
    a_params = actor.init(jax.random.PRNGKey(seed), jnp.zeros((1, s_dim)))
    c_params = critic.init(jax.random.PRNGKey(seed + 1), jnp.zeros((1, s_dim)))
    return actor, critic, a_params, c_params


def _frames(n, size=32):
    """Distinct frames, so a stale belief would show up as a wrong action."""
    return [np.full((size, size, 1), (i + 1) / (n + 2), np.float32)
            for i in range(n)]


def _constant_action(action_dim, index, batch):
    def action_fn(step_key, s):
        return jnp.tile(jnp.zeros(action_dim).at[index].set(1.0), (batch, 1))
    return action_fn


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _planted_rssm(action_dim=3, hidden=4, latent=3, values=(0.0, 1.0, 2.0),
                  continue_logit=4.0, gain=5.0):
    """A tiny RSSM whose imagined reward depends only on the last action.

    Surgery on four parts, each removing one source of state dependence
    so the score of a candidate is a closed-form function of the action
    indices it contains:

    - the GRU's update gate is pinned shut and its hidden-side kernels
      zeroed, so the new h is tanh of the input projection alone;
    - that projection reads only the action, mapping action j to hidden
      unit j with weight `gain`, so h becomes one-hot-ish at tanh(gain);
    - the prior emits mu 0 at its sigma floor, so z is zero to within a
      millionth and contributes nothing anywhere;
    - the reward head reads hidden unit j with weight values[j] through
      one live unit, and the continue head emits a constant logit.

    The reward of a step that played action j is therefore
    silu(values[j] * tanh(gain)), which is strictly increasing in j for
    non-negative values, and the continue probability is constant.
    """
    model = RSSM(latent_dim=latent, hidden=hidden, action_dim=action_dim,
                 obs_channels=1, obs_size=32, min_sigma=1e-6,
                 predict_continue=True)
    p = unfreeze(model.init(
        jax.random.PRNGKey(0), jnp.zeros((1, 32, 32, 1)),
        model.initial_state(1), jnp.zeros((1, latent)),
        jnp.zeros((1, action_dim)),
    ))["params"]

    core = p["core"]
    core["iz"]["kernel"] = jnp.zeros_like(core["iz"]["kernel"])
    core["iz"]["bias"] = jnp.full_like(core["iz"]["bias"], -20.0)
    core["hz"]["kernel"] = jnp.zeros_like(core["hz"]["kernel"])
    core["hn"]["kernel"] = jnp.zeros_like(core["hn"]["kernel"])
    core["hn"]["bias"] = jnp.zeros_like(core["hn"]["bias"])
    core["in"]["bias"] = jnp.zeros_like(core["in"]["bias"])
    # The GRU input is [z_prev, action]; only the action rows are live.
    kernel = jnp.zeros_like(core["in"]["kernel"])
    for j in range(action_dim):
        kernel = kernel.at[latent + j, j].set(gain)
    core["in"]["kernel"] = kernel

    prior = p["prior_head"]
    prior["Dense_1"]["kernel"] = jnp.zeros_like(prior["Dense_1"]["kernel"])
    prior["Dense_1"]["bias"] = jnp.zeros_like(prior["Dense_1"]["bias"])
    prior["Dense_2"]["kernel"] = jnp.zeros_like(prior["Dense_2"]["kernel"])
    prior["Dense_2"]["bias"] = jnp.full_like(prior["Dense_2"]["bias"], -20.0)

    reward = p["reward_head"]
    trunk = jnp.zeros_like(reward["Dense_0"]["kernel"])
    for j in range(action_dim):
        trunk = trunk.at[j, 0].set(values[j])
    reward["Dense_0"]["kernel"] = trunk
    reward["Dense_0"]["bias"] = jnp.zeros_like(reward["Dense_0"]["bias"])
    reward["Dense_1"]["kernel"] = (
        jnp.zeros_like(reward["Dense_1"]["kernel"]).at[0, 0].set(1.0))
    reward["Dense_1"]["bias"] = jnp.zeros_like(reward["Dense_1"]["bias"])

    cont = p["continue_head"]
    cont["Dense_0"]["kernel"] = jnp.zeros_like(cont["Dense_0"]["kernel"])
    cont["Dense_0"]["bias"] = jnp.zeros_like(cont["Dense_0"]["bias"])
    cont["Dense_1"]["kernel"] = jnp.zeros_like(cont["Dense_1"]["kernel"])
    cont["Dense_1"]["bias"] = jnp.full_like(cont["Dense_1"]["bias"],
                                            continue_logit)
    return model, {"params": p}


class FakeEnv:
    """Enough env for collect_episode: constant frames, never dies."""

    action_dim = 3

    def __init__(self, size=32):
        self.size = size
        self._t = 0

    def reset(self):
        self._t = 0
        return self._frame()

    def step(self, action):
        self._t += 1
        return self._frame(), 0.01, False

    @property
    def died(self):
        return False

    def _frame(self):
        return np.full((self.size, self.size, 1), (self._t % 8) / 8.0,
                       np.float32)


def test_imagine_rollout_shape_contract():
    model, params = _tiny_rssm()
    h0 = jnp.zeros((5, 8))
    z0 = jnp.zeros((5, 4))
    states, actions, rewards, continues = imagine_rollout(
        model, params, _constant_action(3, 1, 5), h0, z0,
        jax.random.PRNGKey(0), horizon=6)

    assert states.shape == (7, 5, 12)
    assert actions.shape == (6, 5, 3)
    assert rewards.shape == (6, 5) and continues.shape == (6, 5)
    # s0 sits in front, so states[i] is the state actions[i] was taken from
    np.testing.assert_array_equal(
        np.asarray(states[0]), np.asarray(jnp.concatenate([h0, z0], axis=-1)))
    # continues arrive as probabilities, not logits
    c = np.asarray(continues)
    assert (c > 0.0).all() and (c < 1.0).all()


def test_imagine_rollout_rows_do_not_interact():
    """Row 0 of a mixed batch matches row 0 of a batch whose other rows
    were replaced. Same batch size, so the key stream and the noise row
    are untouched and the only thing that changed is the neighbours."""
    model, params = _tiny_rssm()
    key = jax.random.PRNGKey(3)
    row = jax.random.normal(jax.random.PRNGKey(9), (1, 8))
    others = jax.random.normal(jax.random.PRNGKey(10), (2, 8))
    z0 = jnp.zeros((3, 4))

    mixed, *_ = imagine_rollout(
        model, params, _constant_action(3, 0, 3),
        jnp.concatenate([row, others]), z0, key, horizon=4)
    alone, *_ = imagine_rollout(
        model, params, _constant_action(3, 0, 3),
        jnp.tile(row, (3, 1)), z0, key, horizon=4)
    np.testing.assert_array_equal(np.asarray(mixed[:, 0]),
                                  np.asarray(alone[:, 0]))


def test_zero_temperature_is_the_prior_mean():
    """temperature multiplies sigma, so at 0 the rollout stops depending
    on the prior key at all."""
    model, params = _tiny_rssm()
    h0, z0 = jnp.zeros((2, 8)), jnp.zeros((2, 4))
    args = (model, params, _constant_action(3, 2, 2), h0, z0)
    a, *_ = imagine_rollout(*args, jax.random.PRNGKey(0), 5, temperature=0.0)
    b, *_ = imagine_rollout(*args, jax.random.PRNGKey(1), 5, temperature=0.0)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    sampled, *_ = imagine_rollout(*args, jax.random.PRNGKey(0), 5)
    assert not np.array_equal(np.asarray(a), np.asarray(sampled))


def test_score_rollout_is_lambda_returns_at_lam_one():
    """Hand-computed weights, then the same numbers out of the training
    path: score_rollout is the lam=1 lambda-return of the dream, folded
    forward into one number per candidate."""
    rewards = jnp.array([[1.0], [2.0]])
    continues = jnp.array([[0.5], [1.0]])
    tail = jnp.array([8.0])
    # weights are 1, 0.5 * 0.5 = 0.25, and 0.5**2 * (0.5 * 1.0) = 0.125
    score = score_rollout(rewards, continues, 0.5, tail)
    assert float(score[0]) == pytest.approx(1.0 + 0.5 + 1.0, abs=1e-5)

    values = jnp.array([[0.0], [8.0]])   # only the bootstrap matters at lam=1
    returns = lambda_returns(rewards, values, gamma=0.5, lam=1.0,
                             continues=continues)
    assert float(returns[0, 0]) == pytest.approx(float(score[0]), abs=1e-5)

    no_tail = score_rollout(rewards, continues, 0.5)
    assert float(no_tail[0]) == pytest.approx(1.5, abs=1e-5)


def test_batched_scoring_equals_a_sequential_loop():
    """Five candidates scored in one batch, then one at a time from the
    same dream. Cross-candidate leakage in the reduction would show up
    here as a mismatch."""
    model, params = _tiny_rssm()
    actor, critic, a_params, c_params = _heads()

    def action_fn(step_key, s):
        return jax.nn.one_hot(jax.random.categorical(
            step_key, jnp.zeros((5, 3))), 3)

    states, _, rewards, continues = imagine_rollout(
        model, params, action_fn, jnp.zeros((5, 8)), jnp.zeros((5, 4)),
        jax.random.PRNGKey(11), horizon=7)
    tail = critic.apply(c_params, states[-1])

    batched = np.asarray(score_rollout(rewards, continues, 0.97, tail))
    one_at_a_time = np.array([
        float(score_rollout(rewards[:, j:j + 1], continues[:, j:j + 1],
                            0.97, tail[j:j + 1])[0])
        for j in range(5)
    ])
    np.testing.assert_allclose(batched, one_at_a_time, rtol=1e-6, atol=1e-6)


def test_planner_selects_the_planted_dominant_sequence():
    """On the planted model the reward of a step is a strictly
    increasing function of the action index, and the discount is hard
    enough that no later step can pay for a worse first action: the
    dominant sequence starts with the top action, and the planner has
    to return it. The closed-form score of every sampled candidate is
    checked too, which pins the discount and continue weighting."""
    values = (0.0, 1.0, 2.0)
    continue_logit, gain, discount, horizon = 4.0, 5.0, 0.1, 4
    model, params = _planted_rssm(values=values, continue_logit=continue_logit,
                                  gain=gain)
    config = MPCConfig(n_candidates=32, horizon=horizon, actor_frac=0.0,
                       discount=discount, use_value_tail=False)
    score_candidates, plan = make_planner(model, None, None, 3, config)

    h, z = jnp.zeros(4), jnp.zeros(3)
    key = jax.random.PRNGKey(5)
    actions, scores = score_candidates(params, None, None, h, z, key)

    index = np.asarray(actions).argmax(-1)                      # (H, N)
    step_reward = _silu(np.asarray(values)[index] * np.tanh(gain))
    survival = 1.0 / (1.0 + np.exp(-continue_logit))
    weights = (discount * survival) ** np.arange(horizon)
    expected = (weights[:, None] * step_reward).sum(0)
    np.testing.assert_allclose(np.asarray(scores), expected,
                               rtol=1e-4, atol=1e-4)

    # every action shows up first somewhere, so picking the top one is a
    # real choice rather than the only one on offer
    assert set(index[0].tolist()) == {0, 1, 2}
    winner = plan(params, None, None, h, z, key)
    assert winner.shape == (horizon, 3)
    np.testing.assert_array_equal(np.asarray(winner[0]), [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(np.asarray(winner),
                                  np.asarray(actions[:, expected.argmax()]))


def test_one_candidate_at_low_temperature_is_the_greedy_actor():
    """The graceful-degradation check: with a single actor candidate and
    the temperature at the greedy limit there is nothing to choose
    between, so the planner has to play what the actor would have."""
    model, params = _tiny_rssm()
    actor, critic, a_params, c_params = _heads()
    config = MPCConfig(n_candidates=1, horizon=5, actor_frac=1.0,
                       temperature=1e-6)

    planner = MPCPolicy(model, actor.apply, critic.apply,
                        jax.random.PRNGKey(2), 3, config)
    planner.set_params(params, a_params, c_params)
    greedy = RSSMPolicy(
        model,
        lambda p, s, k: actor.apply(p, s[None], None,
                                    method=DiscreteActor.act)[0],
        jax.random.PRNGKey(2), 3)
    greedy.set_params(params, a_params)

    frames = _frames(6)
    planner.reset(frames[0])
    greedy.reset(frames[0])
    for frame in frames:
        np.testing.assert_array_equal(planner(frame), greedy(frame))


def test_temperature_collapses_actor_candidates_onto_the_argmax():
    """The temperature divides the actor's logits before they are
    sampled, so at the greedy limit every actor candidate opens with the
    action the actor itself would have taken, and a hot one spreads the
    pool back out over the whole vocabulary."""
    model, params = _tiny_rssm()
    actor, _, a_params, _ = _heads()
    # A belief the actor has an opinion about: its trunk carries no
    # biases worth the name, so a zero state comes out of it as zero
    # logits and every temperature looks the same.
    h = jax.random.normal(jax.random.PRNGKey(13), (8,))
    z = jax.random.normal(jax.random.PRNGKey(14), (4,))
    greedy = int(np.asarray(
        actor.apply(a_params, jnp.concatenate([h, z])[None])).argmax(-1)[0])

    def first_actions(temperature):
        config = MPCConfig(n_candidates=24, horizon=3, actor_frac=1.0,
                           temperature=temperature, use_value_tail=False)
        score_candidates, _ = make_planner(model, actor.apply, None, 3, config)
        actions, _ = score_candidates(params, a_params, None, h, z,
                                      jax.random.PRNGKey(12))
        return np.asarray(actions).argmax(-1)[0]

    np.testing.assert_array_equal(first_actions(1e-6), np.full(24, greedy))
    assert set(first_actions(1e3).tolist()) == {0, 1, 2}


def test_planner_is_deterministic_in_its_key():
    model, params = _tiny_rssm()
    actor, critic, a_params, c_params = _heads()
    config = MPCConfig(n_candidates=16, horizon=6)
    frames = _frames(4)

    def run(seed):
        policy = MPCPolicy(model, actor.apply, critic.apply,
                           jax.random.PRNGKey(seed), 3, config)
        policy.set_params(params, a_params, c_params)
        policy.reset(frames[0])
        return np.stack([policy(f) for f in frames])

    np.testing.assert_array_equal(run(0), run(0))

    # A different key draws a different candidate pool, so the winning
    # sequence moves; the belief it was planned from did not.
    _, plan = make_planner(model, actor.apply, critic.apply, 3, config)
    args = (params, a_params, c_params, jnp.zeros(8), jnp.zeros(4))
    first = plan(*args, jax.random.PRNGKey(0))
    second = plan(*args, jax.random.PRNGKey(1))
    assert not np.array_equal(np.asarray(first), np.asarray(second))


def test_mask_keeps_disabled_actions_out_of_every_candidate():
    model, params = _tiny_rssm(action_dim=4)
    actor, critic, a_params, c_params = _heads(action_dim=4)
    mask = np.array([True, False, True, False])
    config = MPCConfig(n_candidates=32, horizon=5, actor_frac=0.5,
                       use_value_tail=False)
    score_candidates, _ = make_planner(model, actor.apply, None, 4, config,
                                       action_mask=mask)
    actions, _ = score_candidates(params, a_params, None, jnp.zeros(8),
                                  jnp.zeros(4), jax.random.PRNGKey(4))

    index = np.asarray(actions).argmax(-1)               # (H, N)
    n_actor = int(round(32 * 0.5))
    assert set(index[:, n_actor:].ravel().tolist()) <= {0, 2}
    assert set(index.ravel().tolist()) <= {0, 2}
    # the enabled slots are both reachable, so this is not passing by
    # the planner having stopped sampling
    assert set(index[:, n_actor:].ravel().tolist()) == {0, 2}


def test_replan_every_runs_the_cached_plan():
    """The schedule is spelled out here rather than read off the policy:
    plan at even steps, take the next action out of that plan at odd
    ones, and update the belief on every frame either way."""
    model, params = _tiny_rssm()
    actor, critic, a_params, c_params = _heads()
    config = MPCConfig(n_candidates=8, horizon=4, replan_every=2)
    key = jax.random.PRNGKey(6)
    _, plan = make_planner(model, actor.apply, critic.apply, 3, config)

    frames = _frames(5)
    h, cached, expected = model.initial_state(1), None, []
    for t, frame in enumerate(frames):
        e = model.apply(params, jnp.asarray(frame)[None], method=RSSM.encode)
        z, _ = model.apply(params, h, e, method=RSSM.post_dist)
        if t % 2 == 0:
            cached = plan(params, a_params, c_params, h[0], z[0],
                          jax.random.fold_in(key, t))
        action = cached[t % 2]
        expected.append(np.asarray(action))
        h = model.apply(params, h, z, action[None], method=RSSM.core_step)

    policy = MPCPolicy(model, actor.apply, critic.apply, key, 3, config)
    policy.set_params(params, a_params, c_params)
    policy.reset(frames[0])
    np.testing.assert_array_equal(np.stack([policy(f) for f in frames]),
                                  np.stack(expected))


def test_mpc_policy_drops_into_collect_episode():
    model, params = _tiny_rssm()
    actor, critic, a_params, c_params = _heads()
    policy = MPCPolicy(model, actor.apply, critic.apply,
                       jax.random.PRNGKey(8), 3,
                       MPCConfig(n_candidates=8, horizon=4))
    policy.set_params(params, a_params, c_params)

    ep = collect_episode(FakeEnv(), policy, 5)
    assert ep["obs"].shape == (6, 32, 32, 1)
    assert ep["action"].shape == (6, 3)
    assert ep["terminated"] is False
    np.testing.assert_array_equal(ep["action"][1:].sum(-1), np.ones(5))


def test_missing_head_functions_are_refused_at_construction():
    model, _ = _tiny_rssm()
    with pytest.raises(ValueError, match="logits_fn"):
        make_planner(model, None, None, 3, MPCConfig(use_value_tail=False))
    with pytest.raises(ValueError, match="value_fn"):
        make_planner(model, None, None, 3, MPCConfig(actor_frac=0.0))
    with pytest.raises(ValueError, match="replan_every"):
        make_planner(model, None, None, 3,
                     MPCConfig(actor_frac=0.0, use_value_tail=False,
                               horizon=4, replan_every=5))
