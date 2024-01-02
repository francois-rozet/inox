r"""Tests for the inox.tree_util module."""

import jax
import pickle
import pytest

from jax import Array
from typing import *

from inox.tree_util import *


@pytest.mark.parametrize('nested', [False, True])
def test_PyArray(nested: bool):
    key = jax.random.key(0)
    x = jax.random.normal(key, (2, 3))

    if nested:
        y = PyArray(PyArray(x))
    else:
        y = PyArray(x)

    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert jax.numpy.allclose(x, y)

    # Operations
    z = x + y
    z = y * y
    z = jax.numpy.mean(y)

    # Flatten
    leaves, treedef = jax.tree_util.tree_flatten(y)

    assert all(isinstance(leaf, Array) for leaf in leaves)
    assert isinstance(treedef, Hashable)

    z = jax.tree_util.tree_unflatten(treedef, leaves)

    assert jax.numpy.allclose(x, z)

    # Pickle
    data = pickle.dumps(y)
    z = pickle.loads(data)

    assert jax.numpy.allclose(x, z)

    # Print
    assert repr(y)


def test_Namespace():
    x = Namespace(a=1, b='2')

    assert x.a == 1
    assert x.b == '2'

    # Add
    x.c = True

    assert x.c == True

    # Modify
    x.a = Namespace(d=None)

    assert x.a.d is None

    # Delete
    del x.b

    assert not hasattr(x, 'b')

    # Flatten
    leaves, treedef = jax.tree_util.tree_flatten(x)

    assert isinstance(treedef, Hashable)

    jax.tree_util.tree_unflatten(treedef, leaves)

    # Pickle
    data = pickle.dumps(x)
    pickle.loads(data)

    # Print
    assert repr(x)


@pytest.mark.parametrize('nested', [False, True])
def test_Static(nested: bool):
    x = 'hashable'

    if nested:
        y = Static(Static(x))
    else:
        y = Static(x)

    # Flatten
    leaves, treedef = jax.tree_util.tree_flatten(y)

    assert not leaves
    assert isinstance(treedef, Hashable)

    z = jax.tree_util.tree_unflatten(treedef, leaves)

    assert hash(y) == hash(z)

    # Pickle
    data = pickle.dumps(y)
    z = pickle.loads(data)

    assert hash(y) == hash(z)

    # Print
    assert repr(y)


def test_Auto():
    x = Auto(
        a=jax.numpy.ones(()),
        b=2,
        c=[True, jax.numpy.arange(4)],
    )

    # Flatten
    leaves, treedef = jax.tree_util.tree_flatten(x)

    assert all(isinstance(leaf, Array) for leaf in leaves)
    assert isinstance(treedef, Hashable)

    jax.tree_util.tree_unflatten(treedef, leaves)

    # Pickle
    data = pickle.dumps(x)
    pickle.loads(data)

    # Print
    assert repr(x)
