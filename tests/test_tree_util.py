r"""Tests for the inox.tree_util module."""

import jax.numpy as jnp
import jax.tree_util as jtu
import pickle
import pytest

from inox.tree_util import *
from jax import Array
from typing import *


def tree_eq(x, y):
    def eq(a, b):
        if isinstance(a, Array) and isinstance(b, Array):
            return jnp.allclose(a, b)
        elif isinstance(a, Array) or isinstance(b, Array):
            return False
        else:
            return a == b

    return jtu.tree_all(jtu.tree_map(eq, x, y))


def test_Namespace():
    x = Namespace(a=1, b='2')

    assert x.a == 1
    assert x.b == '2'

    # Add
    x.c = True

    assert x.c is True

    # Modify
    x.a = Namespace(d=None)

    assert x.a.d is None

    # Delete
    del x.b

    assert not hasattr(x, 'b')

    # Flatten
    leaves, treedef = jtu.tree_flatten(x)

    assert isinstance(treedef, Hashable)

    y = jtu.tree_unflatten(treedef, leaves)

    assert tree_eq(x, y)

    # Pickle
    data = pickle.dumps(x)
    y = pickle.loads(data)

    assert tree_eq(x, y)

    # Print
    assert repr(x)


@pytest.mark.parametrize('nested', [False, True])
def test_Static(nested: bool):
    if nested:
        x = Static(Static('hashable'))
    else:
        x = Static('hashable')

    # Flatten
    leaves, treedef = jtu.tree_flatten(x)

    assert not leaves
    assert isinstance(treedef, Hashable)

    y = jtu.tree_unflatten(treedef, leaves)

    assert x == y

    # Pickle
    data = pickle.dumps(x)
    y = pickle.loads(data)

    assert x == y

    # Print
    assert repr(x)


def test_tree_mask():
    x = Namespace(
        a=jnp.ones(1),
        b=2,
        c=[jnp.arange(3), False],
        d=Namespace(e='five', f=jnp.eye(6)),
    )

    # tree_mask
    y = tree_mask(x)

    leaves, treedef = jtu.tree_flatten(y)

    assert all(isinstance(leaf, Array) for leaf in leaves)
    assert isinstance(treedef, Hashable)

    # tree_unmask
    z = tree_unmask(y)

    leaves, treedef = jtu.tree_flatten(z)

    assert not all(isinstance(leaf, Array) for leaf in leaves)
    assert isinstance(treedef, Hashable)

    assert tree_eq(x, z)


def test_tree_partition():
    x = Namespace(
        a=jnp.ones(1),
        b=2,
        c=[jnp.arange(3), False],
        d=Namespace(e='five', f=jnp.eye(6)),
    )

    # tree_partition
    treedef_a, leaves = tree_partition(x)

    assert not all(isinstance(leaf, Array) for leaf in leaves.values())
    assert isinstance(treedef_a, Hashable)

    treedef_b, arrays, others = tree_partition(x, Array)

    assert all(isinstance(leaf, Array) for leaf in arrays.values())
    assert not any(isinstance(leaf, Array) for leaf in others.values())
    assert isinstance(treedef_b, Hashable)

    assert treedef_a == treedef_b

    # tree_combine
    y = tree_combine(treedef_a, leaves)
    z = tree_combine(treedef_b, arrays, others)

    assert tree_eq(x, y)
    assert tree_eq(x, z)

    with pytest.raises(KeyError):
        tree_combine(treedef_a, {})

    with pytest.raises(KeyError):
        tree_combine(treedef_a, leaves, {'none': None})
