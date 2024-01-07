r"""Tests for the inox.nn module."""

import jax
import jax.numpy as jnp
import pickle
import pytest

from jax import Array
from typing import *

from inox import api
from inox.nn import *


def test_Module():
    module = Module(
        a=0.0,
        b=Parameter(jnp.ones(1)),
        c=[jnp.arange(2), 'three'],
        d=False,
        e=Module(f=jnp.zeros(5), g=range(6)),
    )

    # Flatten
    leaves, treedef = jax.tree_util.tree_flatten(module)

    assert not all(isinstance(leaf, Array) for leaf in leaves)
    assert isinstance(treedef, Hashable)

    jax.tree_util.tree_unflatten(treedef, leaves)

    # Partition
    static, arrays = module.partition()

    assert all(isinstance(leaf, Array) for leaf in arrays.values())
    assert isinstance(static, Hashable)

    static(arrays)

    ## filters
    static, params, others = module.partition(Parameter)

    assert all(isinstance(leaf, Array) for leaf in params.values())
    assert all(isinstance(leaf, Array) for leaf in others.values())
    assert isinstance(static, Hashable)

    static(params, others)

    with pytest.raises(KeyError):
        static(params)

    with pytest.raises(KeyError):
        static(params, others, {'none': None})

    # Pickle
    data = pickle.dumps(module)
    pickle.loads(data)

    # Print
    assert repr(module)


def test_MLP():
    key = jax.random.key(0)
    x = jax.random.normal(key, (1024, 3))
    y = jnp.linalg.norm(x, axis=-1, keepdims=True)

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
    @api.jit
    def loss(model):
        return jnp.mean((model(x) - y) ** 2)

    loss(model)

    # Gradients
    grads = api.grad(loss)(model)
