import jax
import jax.numpy as jnp

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
