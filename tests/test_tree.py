r"""Tests for the inox.tree_util module."""

import jax.numpy as jnp
import jax.tree_util as jtu
import pickle
import pytest

from jax import Array
from typing import Hashable

from inox.tree import Namespace, Static, combine, mask_static, partition, unmask_static


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
    x = Namespace(a=1, b="2")

    assert x.a == 1
    assert x.b == "2"

    # Add
    x.c = True

    assert x.c is True

    # Modify
    x.a = Namespace(d=None)

    assert x.a.d is None

    # Delete
    del x.b

    assert not hasattr(x, "b")

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


@pytest.mark.parametrize("nested", [False, True])
def test_Static(nested: bool):
    if nested:
        x = Static(Static("hashable"))
    else:
        x = Static("hashable")

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


def test_mask_static():
    x = Namespace(
        a=jnp.ones(1),
        b=2,
        c=[jnp.arange(3), False],
        d=Namespace(e="five", f=jnp.eye(6)),
    )

    # mask_static
    y = mask_static(x)

    leaves, treedef = jtu.tree_flatten(y)

    assert all(isinstance(leaf, Array) for leaf in leaves)
    assert isinstance(treedef, Hashable)

    # unmask_static
    z = unmask_static(y)

    leaves, treedef = jtu.tree_flatten(z)

    assert not all(isinstance(leaf, Array) for leaf in leaves)
    assert isinstance(treedef, Hashable)

    assert tree_eq(x, z)


def test_partition():
    x = Namespace(
        a=jnp.ones(1),
        b=2,
        c=[jnp.arange(3), False],
        d=Namespace(e="five", f=jnp.eye(6)),
    )

    # partition
    treedef_a, leaves = partition(x)

    assert not all(isinstance(leaf, Array) for leaf in leaves.values())
    assert isinstance(treedef_a, Hashable)

    treedef_b, arrays, others = partition(x, Array)

    assert all(isinstance(leaf, Array) for leaf in arrays.values())
    assert not any(isinstance(leaf, Array) for leaf in others.values())
    assert isinstance(treedef_b, Hashable)

    assert treedef_a == treedef_b

    # combine
    y = combine(treedef_a, leaves)
    z = combine(treedef_b, arrays, others)

    assert tree_eq(x, y)
    assert tree_eq(x, z)

    with pytest.raises(KeyError):
        combine(treedef_a, {})

    with pytest.raises(KeyError):
        combine(treedef_a, leaves, {"none": None})
