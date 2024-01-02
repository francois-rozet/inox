r"""Tests for the inox.random module."""

import jax
import pickle
import pytest

from typing import *

from inox.random import *


@pytest.mark.parametrize('seed', [0, jax.random.key(0)])
def test_PRNG(seed):
    rng = PRNG(seed)
    shape = rng.state.shape

    # Split
    assert rng.split().shape == shape
    assert rng.split(3).shape == (3, *shape)

    assert not jax.numpy.allclose(rng.split(), rng.split())

    # Sample
    assert rng.normal().shape == ()
    assert rng.uniform(shape=(2, 3)).shape == (2, 3)

    assert not jax.numpy.allclose(rng.normal(), rng.normal())

    # Detect JIT
    with pytest.raises(AssertionError):
        jax.jit(lambda: rng.split())()

    # Flatten
    leaves, treedef = jax.tree_util.tree_flatten(rng)

    assert isinstance(treedef, Hashable)

    jax.tree_util.tree_unflatten(treedef, leaves)

    # Pickle
    data = pickle.dumps(rng)
    pickle.loads(data)

    # Print
    assert repr(rng)
