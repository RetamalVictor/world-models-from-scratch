import jax
import jax.numpy as jnp
import numpy as np

from world_models.models.actor_critic import (
    Actor,
    Critic,
    DiscreteActor,
    lambda_returns,
)


def _discrete_actor(action_dim=3, batch=6):
    actor = DiscreteActor(action_dim=action_dim)
    s = jax.random.normal(jax.random.PRNGKey(0), (batch, 144))
    return actor, s, actor.init(jax.random.PRNGKey(1), s)


def test_lambda_returns_closed_form():
    # H=2, r=[1,2], v=[10,20], gamma=0.5, lam=0.5:
    #   R[1] = 2 + 0.5*20            = 12
    #   R[0] = 1 + 0.5*(0.5*10+0.5*12) = 6.5
    rewards = jnp.array([[1.0], [2.0]])
    values = jnp.array([[10.0], [20.0]])
    returns = lambda_returns(rewards, values, gamma=0.5, lam=0.5)
    assert abs(float(returns[1, 0]) - 12.0) < 1e-5
    assert abs(float(returns[0, 0]) - 6.5) < 1e-5


def test_lambda_one_is_monte_carlo_with_bootstrap():
    # lam = 1 reduces to discounted sum of rewards + discounted final value.
    rewards = jnp.array([[1.0], [1.0], [1.0]])
    values = jnp.array([[5.0], [5.0], [5.0]])
    returns = lambda_returns(rewards, values, gamma=0.9, lam=1.0)
    expected = 1 + 0.9 * (1 + 0.9 * (1 + 0.9 * 5.0))
    assert abs(float(returns[0, 0]) - expected) < 1e-4


def test_actor_actions_are_bounded():
    actor = Actor(max_action=0.3)
    s = jax.random.normal(jax.random.PRNGKey(0), (7, 144)) * 10
    params = actor.init(jax.random.PRNGKey(1), s)
    a = actor.apply(params, s, jax.random.PRNGKey(2), method=Actor.act)
    assert a.shape == (7, 2)
    assert float(jnp.abs(a).max()) <= 0.3 + 1e-6
    a_det = actor.apply(params, s, None, method=Actor.act)
    assert float(jnp.abs(a_det).max()) <= 0.3 + 1e-6


def test_critic_shape():
    critic = Critic()
    s = jnp.zeros((5, 144))
    params = critic.init(jax.random.PRNGKey(0), s)
    v = critic.apply(params, s)
    assert v.shape == (5,)


def test_all_ones_continues_reduce_to_plain_lambda_returns():
    rewards = jnp.array([[1.0], [2.0], [3.0]])
    values = jnp.array([[10.0], [20.0], [30.0]])
    plain = lambda_returns(rewards, values, gamma=0.9, lam=0.7)
    discounted = lambda_returns(rewards, values, gamma=0.9, lam=0.7,
                                continues=jnp.ones_like(rewards))
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(discounted))


def test_discounted_lambda_returns_closed_form():
    # H=3, r=[1,2,3], v=[10,20,30], gamma=0.5, lam=0.5, c=[1,0,1].
    # The discount is gamma*c, so the death at i=1 truncates:
    #   R[2] = 3 + 0.5*(0.5*30 + 0.5*30)    = 18
    #   R[1] = 2 + 0.0*(...)                = 2
    #   R[0] = 1 + 0.5*(0.5*10 + 0.5*2)     = 4
    # With c all ones R[0] would be 6.375, so the flag is load-bearing.
    rewards = jnp.array([[1.0], [2.0], [3.0]])
    values = jnp.array([[10.0], [20.0], [30.0]])
    continues = jnp.array([[1.0], [0.0], [1.0]])
    returns = lambda_returns(rewards, values, gamma=0.5, lam=0.5,
                             continues=continues)
    assert abs(float(returns[2, 0]) - 18.0) < 1e-5
    assert abs(float(returns[1, 0]) - 2.0) < 1e-5
    assert abs(float(returns[0, 0]) - 4.0) < 1e-5
    plain = lambda_returns(rewards, values, gamma=0.5, lam=0.5)
    assert abs(float(plain[0, 0]) - 6.375) < 1e-5


def test_soft_continues_interpolate():
    # A half-confident continue halves the discount at that step:
    #   R[1] = 2 + 0.5*0.5*20 = 7
    #   R[0] = 1 + 0.5*1.0*(0.5*10 + 0.5*7) = 5.25
    rewards = jnp.array([[1.0], [2.0]])
    values = jnp.array([[10.0], [20.0]])
    continues = jnp.array([[1.0], [0.5]])
    returns = lambda_returns(rewards, values, gamma=0.5, lam=0.5,
                             continues=continues)
    assert abs(float(returns[1, 0]) - 7.0) < 1e-5
    assert abs(float(returns[0, 0]) - 5.25) < 1e-5


def test_discrete_actor_emits_exact_one_hots():
    actor, s, params = _discrete_actor()
    a = actor.apply(params, s, jax.random.PRNGKey(2),
                    method=DiscreteActor.act)
    assert a.shape == (6, 3)
    np.testing.assert_array_equal(np.asarray(a).sum(-1), np.ones(6))
    assert set(np.unique(np.asarray(a))) <= {0.0, 1.0}
    # same key, same draw
    again = actor.apply(params, s, jax.random.PRNGKey(2),
                        method=DiscreteActor.act)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(again))


def test_discrete_actor_greedy_path_is_the_argmax():
    actor, s, params = _discrete_actor()
    logits = actor.apply(params, s)
    det = actor.apply(params, s, None, method=DiscreteActor.act)
    np.testing.assert_array_equal(np.asarray(det).argmax(-1),
                                  np.asarray(logits).argmax(-1))
    np.testing.assert_array_equal(
        np.asarray(det),
        np.asarray(actor.apply(params, s, None, method=DiscreteActor.act)),
    )


def test_straight_through_is_hard_forward_and_soft_backward():
    actor, s, params = _discrete_actor()
    key = jax.random.PRNGKey(3)
    st = actor.apply(params, s, key, method=DiscreteActor.sample_st)
    hard = actor.apply(params, s, key, method=DiscreteActor.act)
    np.testing.assert_allclose(np.asarray(st), np.asarray(hard), atol=1e-6)

    w = jnp.array([1.0, -2.0, 0.5])

    def score(p):
        a = actor.apply(p, s, key, method=DiscreteActor.sample_st)
        return (a * w).sum()

    grads = jax.grad(score)(params)
    biggest = max(float(jnp.abs(g).max())
                  for g in jax.tree_util.tree_leaves(grads))
    assert biggest > 0


def test_discrete_actor_entropy():
    actor, s, params = _discrete_actor(batch=4)
    logits = actor.apply(params, s)
    # log_softmax spelled out, so the test does not lean on the module
    shifted = logits - logits.max(-1, keepdims=True)
    log_p = shifted - jnp.log(jnp.exp(shifted).sum(-1, keepdims=True))
    expected = -(jnp.exp(log_p) * log_p).sum(-1)
    ent = actor.apply(params, s, method=DiscreteActor.entropy)
    assert ent.shape == (4,)
    np.testing.assert_allclose(np.asarray(ent), np.asarray(expected),
                               atol=1e-6)
    # zeroed params give zero logits, i.e. the uniform maximum
    flat = actor.apply(jax.tree.map(jnp.zeros_like, params), s,
                       method=DiscreteActor.entropy)
    np.testing.assert_allclose(np.asarray(flat), np.log(3.0), atol=1e-6)


def test_discrete_actor_mask_zero_frequency_and_probability_mass():
    actor, s, params = _discrete_actor(action_dim=4, batch=1)
    mask = jnp.array([True, False, True, True])
    choices = set()
    for i in range(500):
        a = actor.apply(params, s, jax.random.PRNGKey(1000 + i), mask,
                        method=DiscreteActor.act)
        choices.add(int(np.asarray(a)[0].argmax()))
    assert 1 not in choices
    assert choices <= {0, 2, 3}

    disabled_one_hot = jax.nn.one_hot(jnp.array([1]), 4)
    log_p = actor.apply(params, s, disabled_one_hot, mask,
                        method=DiscreteActor.log_prob)
    assert float(jnp.exp(log_p)[0]) < 1e-6


def test_discrete_actor_mask_entropy_equals_smaller_distribution():
    actor, s, params = _discrete_actor(action_dim=5, batch=4)
    mask = np.array([True, False, True, True, False])
    logits = np.asarray(actor.apply(params, s))
    enabled = logits[:, mask]
    # Same log_softmax spelled out as test_discrete_actor_entropy, but
    # over only the enabled columns: the entropy of a masked 5-way
    # distribution should equal the entropy of the plain 3-way one.
    shifted = enabled - enabled.max(-1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    expected = -(np.exp(log_p) * log_p).sum(-1)
    ent = actor.apply(params, s, jnp.array(mask), method=DiscreteActor.entropy)
    np.testing.assert_allclose(np.asarray(ent), expected, atol=1e-5)


def test_discrete_actor_full_mask_gradients_match_no_mask():
    actor, s, params = _discrete_actor(action_dim=3, batch=4)
    key = jax.random.PRNGKey(7)
    full_mask = jnp.array([True, True, True])
    w = jnp.array([1.0, -2.0, 0.5])

    def score(p, mask):
        a = actor.apply(p, s, key, mask, method=DiscreteActor.sample_st)
        return (a * w).sum()

    grads_none = jax.grad(lambda p: score(p, None))(params)
    grads_full = jax.grad(lambda p: score(p, full_mask))(params)
    for g_none, g_full in zip(jax.tree_util.tree_leaves(grads_none),
                              jax.tree_util.tree_leaves(grads_full)):
        np.testing.assert_array_equal(np.asarray(g_none), np.asarray(g_full))


def test_discrete_actor_mask_straight_through_still_flows_gradient():
    actor, s, params = _discrete_actor(action_dim=4, batch=5)
    key = jax.random.PRNGKey(11)
    mask = jnp.array([True, False, True, False])

    st = actor.apply(params, s, key, mask, method=DiscreteActor.sample_st)
    hard = actor.apply(params, s, key, mask, method=DiscreteActor.act)
    np.testing.assert_allclose(np.asarray(st), np.asarray(hard), atol=1e-6)
    chosen = set(np.asarray(hard).argmax(-1).tolist())
    assert chosen <= {0, 2}

    w = jnp.array([1.0, 0.3, -0.7, 2.0])

    def score(p):
        a = actor.apply(p, s, key, mask, method=DiscreteActor.sample_st)
        return (a * w).sum()

    grads = jax.grad(score)(params)
    biggest = max(float(jnp.abs(g).max())
                  for g in jax.tree_util.tree_leaves(grads))
    assert biggest > 0
