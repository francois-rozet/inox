r"""Tests for the inox.debug module."""

import jax
import pytest

from functools import partial
from inox.debug import *
from inox.tree_util import PyArray


@pytest.mark.parametrize('pyarray', [False, True])
def test_same_trace(pyarray: bool):
    x = jax.numpy.zeros(1)
    y = jax.numpy.ones(1)

    if pyarray:
        y = PyArray(y)

    assert same_trace(x, y)
    assert jax.jit(same_trace)(x, y)
    assert not jax.jit(partial(same_trace, x))(y)
