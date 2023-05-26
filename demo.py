#!/usr/bin/env python

import jax
import jax.numpy as jnp
import optax

from inox.nn import *
from inox.tree_util import tree_repr


if __name__ == '__main__':
    # RNG
    key = jax.random.PRNGKey(0)

    # Data
    key, subkey = jax.random.split(key, 2)

    def f(x):  # (3,) -> ()
        return jnp.sum(x) + jnp.prod(x)

    X = jax.random.normal(subkey, (1024, 3))
    Y = jax.vmap(f)(X)

    def mse(f, x, y):
        return jnp.mean(jnp.square(jax.vmap(f)(x) - y))

    # Network
    key, subkey = jax.random.split(key, 2)

    class MyModule(Module):
        hello: bool = True
        perceptron: Module

        def __init__(self):
            self.perceptron = MLP(subkey, 3, 1, hidden_features=[64, 64])

        def __call__(self, x):
            if self.hello:
                print('Hello, World!')

            return jnp.squeeze(self.perceptron(x))

    net = MyModule()

    # Nice repr
    print(net)

    # Native JIT
    print(jax.jit(mse)(net, X, Y))

    # Training
    net.hello = False
    state, build = net.functional()

    ## Optimizer
    optimizer = optax.adamw(learning_rate=1e-3)
    opt_state = optimizer.init(state)

    ## Loop
    @jax.jit
    def step(state, opt_state):
        def loss(state, x, y):
            return mse(build(state), x, y)

        loss, grads = jax.value_and_grad(loss)(state, X, Y)
        updates, opt_state = optimizer.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)

        return state, opt_state, loss

    for i in range(1024):
        state, opt_state, loss = step(state, opt_state)

        if i % 128 == 0:
            print(f'{i:04d}', ':', loss)
