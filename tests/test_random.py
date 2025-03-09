r"""Tests for the inox.random module."""

import jax
import jax.numpy as jnp
import pickle
import pytest

from typing import Hashable

from inox.random import PRNG, get_rng, set_rng


@pytest.mark.parametrize("seed", [0, jax.random.key(0)])
def test_PRNG(seed):
    rng = PRNG(seed)
    shape = rng.state.shape

    # Split
    assert rng.split().shape == shape
    assert rng.split(3).shape == (3, *shape)

    assert not jnp.allclose(rng.split(), rng.split())

    # Sample
    assert rng.normal().shape == ()
    assert rng.uniform(shape=(2, 3)).shape == (2, 3)

    assert not jnp.allclose(rng.normal(), rng.normal())

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


@pytest.mark.parametrize("seed", [0, jax.random.key(0)])
def test_set_rng(seed):
    with set_rng(init=PRNG(seed), dropout=PRNG(seed)):
        a = get_rng("init").split()
        b = get_rng("dropout").split()

        assert jnp.allclose(a, b)

    with pytest.raises(AssertionError):
        get_rng("init")
