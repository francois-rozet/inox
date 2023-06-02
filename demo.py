#!/usr/bin/env python

import inox
import jax
import jax.numpy as jnp
import optax


if __name__ == '__main__':
    # Random number generator
    rng = inox.random.Generator(0)

    # Data
    def f(x):  # (3,) -> ()
        return jnp.sum(x) + jnp.prod(x)

    X = jax.random.normal(rng(), (1024, 3))
    Y = jax.vmap(f)(X)

    def mse(f, x, y):
        return jnp.mean(jnp.square(f(x) - y))

    # Network
    class MyModule(inox.nn.Module):
        def __init__(self, key):
            self.hello = True
            self.perceptron = inox.nn.MLP(key, 3, 1, hidden_features=[64, 64])

        def __call__(self, x):
            if self.hello:
                print('Hello, World!')

            return jnp.squeeze(self.perceptron(x))

    net = MyModule(rng())

    # Nice module repr
    print(net)

    # JIT-able networks
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
