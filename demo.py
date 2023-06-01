#!/usr/bin/env python

import jax
import jax.numpy as jnp
import optax

from inox.nn import *
from inox.tree_util import *


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
        def __init__(self, key):
            self.hello = True
            self.perceptron = MLP(key, 3, 1, hidden_features=[64, 64])

        def __call__(self, x):
            if self.hello:
                print('Hello, World!')

            return jnp.squeeze(self.perceptron(x))

    net = MyModule(subkey)

    # Nice repr
    print(net)

    # Native JIT
    print(jax.jit(mse)(net, X, Y))

    # Training
    net.hello = False
    params, buffers, build = net.partition()

    def loss(params, buffers, x, y):
        return mse(jax.vmap(build(params, buffers)), x, y)

    ## Optimizer
    optimizer = optax.adamw(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    ## Loop
    @jax.jit
    def step(params, buffers, opt_state):
        lval, grads = jax.value_and_grad(loss)(params, buffers, X, Y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, buffers, opt_state, lval

    for i in range(1024):
        params, buffers, opt_state, lval = step(params, buffers, opt_state)

        if i % 128 == 0:
            print(f'{i:04d}', ':', lval)
