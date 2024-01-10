r"""Tests for the inox.nn module."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
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
    leaves, treedef = jtu.tree_flatten(module)

    assert not all(isinstance(leaf, Array) for leaf in leaves)
    assert isinstance(treedef, Hashable)

    jtu.tree_unflatten(treedef, leaves)

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


@pytest.mark.parametrize('norm', ['layer', 'group'])
def test_MLP(norm: str):
    key = jax.random.key(0)
    x = jax.random.normal(key, (1024, 3))
    y = jnp.linalg.norm(x, axis=-1, keepdims=True)

    class MLP(Module):
        def __init__(self, key):
            keys = jax.random.split(key)

            self.l1 = Linear(3, 64, key=keys[0])
            self.l2 = Linear(64, 1, key=keys[1])
            self.relu = ReLU()
            self.norm = LayerNorm() if norm == 'group' else GroupNorm(4)

        def __call__(self, x):
            return self.l2(self.norm(self.relu(self.l1(x))))

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


def test_BatchNorm():
    key = jax.random.key(0)
    x = jax.random.uniform(key, (1024, 3))
    y = jnp.linalg.norm(x, axis=-1, keepdims=True)

    class MLP(Module):
        def __init__(self, key):
            keys = jax.random.split(key)

            self.l1 = Linear(3, 64, key=keys[0])
            self.l2 = Linear(64, 1, key=keys[1])
            self.relu = ReLU()
            self.norm = BatchNorm(64)

        def __call__(self, x, state):
            x = self.l1(x)
            x = self.relu(x)
            x, state = self.norm(x, state)
            x = self.l2(x)

            return x, state

    # __init__
    model = MLP(key)
    model, state = export_state(model)

    # __call__
    y, new = model(x, state)

    assert not any(jtu.tree_leaves(jax.tree_map(jnp.allclose, new, state)))

    with pytest.raises(TypeError):
        model(x, None)

    with pytest.raises(AssertionError):
        model(x[0], state)

    with pytest.raises(AssertionError):
        jax.vmap(model, in_axes=(0, None))(x, state)

    # JIT
    @api.jit
    def loss(model, state):
        z, state = model(x, state)

        return jnp.mean((z - y) ** 2), state

    l, state = loss(model, state)

    # Gradients
    grads, state = api.grad(loss, has_aux=True)(model, state)
