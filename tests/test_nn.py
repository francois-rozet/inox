r"""Tests for the inox.nn module."""

import jax
import pickle
import pytest

from jax import Array
from typing import *

from inox.nn import *


def test_Module():
    module = Module(
        a=jax.numpy.zeros(()),
        b=Parameter(jax.numpy.ones(1)),
        c=[True, Statistic(jax.numpy.arange(3))],
        d='four',
    )

    # Flatten
    leaves, treedef = jax.tree_util.tree_flatten(module)

    assert all(isinstance(leaf, Array) for leaf in leaves)
    assert isinstance(treedef, Hashable)

    jax.tree_util.tree_unflatten(treedef, leaves)

    # Pure
    modef, state = module.pure()

    print(state)

    assert all(isinstance(leaf, Array) for leaf in state.values())
    assert isinstance(modef, Hashable)

    modef(state)

    # Pure split
    modef, params, others = module.pure(Parameter)

    assert all(isinstance(leaf, Array) for leaf in params.values())
    assert all(isinstance(leaf, Array) for leaf in others.values())
    assert isinstance(modef, Hashable)

    modef(params, others)

    with pytest.raises(KeyError):
        modef(params)

    with pytest.raises(KeyError):
        modef(params, others, {'e': None})

    # Pickle
    data = pickle.dumps(module)
    pickle.loads(data)

    # Print
    assert repr(module)


def test_MLP():
    key = jax.random.key(0)
    x = jax.random.normal(key, (1024, 3))
    y = jax.numpy.linalg.norm(x, axis=-1, keepdims=True)

    class MLP(Module):
        def __init__(self, key):
            keys = jax.random.split(key)

            self.l1 = Linear(keys[0], 3, 64)
            self.l2 = Linear(keys[1], 64, 1)
            self.relu = ReLU()

        def __call__(self, x):
            return self.l2(self.relu(self.l1(x)))

    # __init__
    model = MLP(key)

    # __call__
    z = model(x[0])
    z = model(x)
    z = jax.vmap(model)(x)

    # JIT
    @jax.jit
    def loss(model):
        return jax.numpy.mean((model(x) - y) ** 2)

    l = loss(model)

    # Gradients
    grads = jax.grad(loss)(model)
