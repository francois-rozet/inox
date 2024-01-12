r"""Tests for the inox.api module."""

import jax
import pytest

from functools import partial
from inox.api import *
from inox.api import inner, outer
from inox.tree_util import Static


def test_automask():
    def is_static(x):
        assert isinstance(x, Static)
        return x

    def is_not_static(x):
        assert not isinstance(x, Static)
        return x

    # inner & outer
    assert is_static(inner(is_not_static)('leaf'))
    assert is_static(inner(is_not_static)(Static('leaf')))
    assert is_not_static(outer(is_static)('leaf'))
    assert is_not_static(outer(is_static)(Static('leaf')))

    def f(x): return x

    assert outer(inner(f)) is f
    assert inner(outer(f)) is f

    # automask
    assert is_not_static(automask(partial)(is_not_static)('leaf'))
    assert is_not_static(automask(partial)(is_not_static)(Static('leaf')))

    assert automask(lambda f: f)(f) is f
    assert automask(partial)(f) is not f
