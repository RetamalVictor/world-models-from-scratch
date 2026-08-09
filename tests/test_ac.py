import jax
import jax.numpy as jnp
import numpy as np

from world_models.models.actor_critic import Actor, Critic, lambda_returns


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
