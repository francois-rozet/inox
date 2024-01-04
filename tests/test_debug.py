r"""Tests for the inox.debug module."""

import jax
import pytest

from functools import partial
from inox.debug import *


def test_same_trace():
    x = jax.numpy.zeros(1)
    y = jax.numpy.ones(1)

    assert same_trace(x, y)
    assert jax.jit(same_trace)(x, y)
    assert not jax.jit(partial(same_trace, x))(y)
