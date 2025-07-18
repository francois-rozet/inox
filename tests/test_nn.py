r"""Tests for the inox.nn module."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pickle
import pytest

from typing import Hashable

import inox
import inox.nn as nn

from inox.tree import is_array


def test_Module():
    module = nn.Module(
        a=0.0,
        b=nn.Parameter(jnp.ones(1)),
        c=[jnp.arange(2), "three"],
        d=False,
        e=nn.Module(f=np.zeros(5), g=range(6)),
        h=nn.ComplexParameter(7.0 + 8.0j),
        i=nn.Scope(j=nn.Reference("i.j", {9: None})),
    )

    module.i.k = module.i.j
    module.i.k[10] = module.i.j

    # Flatten
    leaves, treedef = jtu.tree_flatten(module)

    assert not all(map(is_array, leaves))
    assert isinstance(treedef, Hashable)

    copy = jtu.tree_unflatten(treedef, leaves)

    assert copy.i.j is copy.i.k
    assert copy.i.j is copy.i.k[10]

    ## Reference conflict
    copy.i.k = nn.Reference("i.j", "broken")

    with pytest.raises(AssertionError):
        jtu.tree_flatten(copy)

    # Partition
    static, arrays = module.partition()

    assert all(map(is_array, arrays.values()))
    assert not any(map(is_array, jtu.tree_leaves(static)))

    static(arrays)

    ## filters (type)
    static, params, others = module.partition(nn.Parameter)

    assert all(key.endswith((".value", ".real", ".imag")) for key in params)
    assert all(map(is_array, params.values()))
    assert all(map(is_array, others.values()))
    assert not any(map(is_array, jtu.tree_leaves(static)))

    static(params, others)

    with pytest.raises(KeyError):
        static(params)

    with pytest.raises(KeyError):
        static(params, others, {"none": None})

    ## filters (callable)
    module.e.frozen = True

    static, frozen, others = module.partition(lambda x: getattr(x, "frozen", False))

    assert all(".e" in key for key in frozen)
    assert all(".e" not in key for key in others)
    assert all(map(is_array, frozen.values()))
    assert all(map(is_array, others.values()))
    assert not any(map(is_array, jtu.tree_leaves(static)))

    # Pickle
    data = pickle.dumps(module)
    pickle.loads(data)

    # Print
    assert repr(module)


@pytest.mark.parametrize("norm", ["layer", "group"])
def test_MLP(norm: str):
    key = jax.random.key(0)
    x = jax.random.normal(key, (1024, 3))
    y = jnp.linalg.norm(x, axis=-1, keepdims=True)

    class MLP(nn.Module):
        def __init__(self, key):
            keys = jax.random.split(key)

            self.l1 = nn.Linear(3, 64, key=keys[0])
            self.l2 = nn.Linear(64, 1, key=keys[1])
            self.relu = nn.ReLU()
            self.norm = nn.LayerNorm() if norm == "group" else nn.GroupNorm(4)

        def __call__(self, x):
            return self.l2(self.norm(self.relu(self.l1(x))))

    # __init__
    model = MLP(key)

    # __call__
    z0 = model(x[0])
    z1 = model(x)
    z2 = jax.vmap(model)(x)

    assert z0.shape == y[0].shape
    assert z1.shape == y.shape
    assert z2.shape == y.shape
    assert jnp.allclose(z1, z2)

    # JIT
    @inox.jit
    def loss(model, x, y):
        return jnp.mean((model(x) - y) ** 2)

    loss(model, x, y)

    # Partition
    static, params, others = model.partition(nn.Parameter)

    assert all(key.endswith(".value") for key in params)
    assert all(map(is_array, params.values()))
    assert all(map(is_array, others.values()))
    assert not any(map(is_array, jtu.tree_leaves(static)))

    # Gradients
    def ell(params):
        return loss(static(params, others), x, y)

    grads = jax.grad(ell)(params)
    params = jtu.tree_map(jnp.add, params, grads)

    # Print
    assert repr(model)


def test_BatchNorm():
    key = jax.random.key(0)
    x = jax.random.uniform(key, (1024, 3))
    y = jnp.linalg.norm(x, axis=-1, keepdims=True)

    class MLP(nn.Module):
        def __init__(self, key):
            keys = jax.random.split(key)

            self.l1 = nn.Linear(3, 64, key=keys[0])
            self.l2 = nn.Linear(64, 1, key=keys[1])
            self.relu = nn.ReLU()
            self.norm = nn.BatchNorm(64)

        def __call__(self, x, state):
            x = self.l1(x)
            x = self.relu(x)
            x, state = self.norm(x, state)
            x = self.l2(x)

            return x, state

    # __init__
    model = MLP(key)
    model, state = nn.export_state(model)

    # __call__
    z, new = model(x, state)

    assert z.shape == y.shape
    assert not any(jtu.tree_leaves(jtu.tree_map(jnp.allclose, new, state)))

    with pytest.raises(TypeError):
        model(x, None)

    with pytest.raises(AssertionError):
        model(x[0], state)

    with pytest.raises(AssertionError):
        jax.vmap(model, in_axes=(0, None))(x, state)

    # JIT
    @inox.jit
    def loss(model, state, x, y):
        z, state = model(x, state)

        return jnp.mean((z - y) ** 2), state

    _, state = loss(model, state, x, y)

    # Partition
    static, params, others = model.partition(nn.Parameter)

    assert all(key.endswith(".value") for key in params)
    assert all(map(is_array, params.values()))
    assert all(map(is_array, others.values()))
    assert not any(map(is_array, jtu.tree_leaves(static)))

    # Gradients
    def ell(params):
        return loss(static(params, others), state, x, y)

    grads, state = jax.grad(ell, has_aux=True)(params)
    params = jtu.tree_map(jnp.add, params, grads)

    # Print
    assert repr(model)


def test_share():
    key = jax.random.key(0)
    x = jax.random.uniform(key, (1024, 3))
    y = jnp.linalg.norm(x, axis=-1, keepdims=True)

    class MLP(nn.Scope):
        def __init__(self, key):
            keys = jax.random.split(key, 3)

            self.l1 = nn.Linear(in_features=3, out_features=64, key=keys[0])
            self.l2 = nn.Linear(in_features=64, out_features=64, key=keys[1])
            self.l4 = nn.Linear(in_features=64, out_features=1, key=keys[2])
            self.relu = nn.ReLU()

            self.l2 = nn.Reference("l2", self.l2)
            self.l2.cycle = self.l2
            self.l3 = self.l2
            self.l4.weight = nn.Reference("l4.weight", self.l4.weight)
            self.void = self.l4.weight

        def __call__(self, x):
            x = self.l1(x)
            x = self.l2.cycle.cycle(self.relu(x))
            x = self.l3(self.relu(x))
            x = self.l4(self.relu(x))

            return x

    # __init__
    model = MLP(key)

    # __call__
    z0 = model(x[0])
    z1 = model(x)
    z2 = jax.vmap(model)(x)

    assert z0.shape == y[0].shape
    assert z1.shape == y.shape
    assert z2.shape == y.shape
    assert jnp.allclose(z1, z2)

    # JIT
    @inox.jit
    def loss(model, x, y):
        return jnp.mean((model(x) - y) ** 2)

    loss(model, x, y)

    # Partition
    static, params, others = model.partition(nn.Parameter)

    assert not any(".cycle" in key for key in params)
    assert not any(key.startswith(".l3") or key.startswith(".void") for key in params)
    assert all(key.endswith(".value") for key in params)
    assert all(map(is_array, params.values()))
    assert all(map(is_array, others.values()))
    assert not any(map(is_array, jtu.tree_leaves(static)))

    # Gradients
    def ell(params):
        return loss(static(params, others), x, y)

    grads = jax.grad(ell)(params)
    params = jtu.tree_map(jnp.add, params, grads)

    # Print
    assert repr(model)


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("causal", [False, True])
def test_mha(heads: int, causal: bool):
    key = jax.random.key(0)
    x = jax.random.normal(key, (3, 5, 64))

    if jax.__version__ < "0.5":
        pytest.skip("jax<=0.4 does not implement SDPA")

    # __init__
    model = nn.MultiheadAttention(
        in_features=64,
        heads=heads,
        causal=causal,
        key=key,
    )

    # __call__
    y0 = model(x[0])
    y1 = model(x)
    y2 = jax.vmap(model)(x)

    assert y0.shape == x[0].shape
    assert y1.shape == x.shape
    assert y2.shape == x.shape
    assert jnp.allclose(y1, y2)

    # JIT
    @inox.jit
    def loss(model, x):
        return jnp.mean(model(x) ** 2)

    loss(model, x)

    # Partition
    static, params, others = model.partition(nn.Parameter)

    assert all(key.endswith(".value") for key in params)
    assert all(map(is_array, params.values()))
    assert all(map(is_array, others.values()))
    assert not any(map(is_array, jtu.tree_leaves(static)))

    # Gradients
    def ell(params):
        return loss(static(params, others), x)

    grads = jax.grad(ell)(params)
    params = jtu.tree_map(jnp.add, params, grads)

    # Print
    assert repr(model)
