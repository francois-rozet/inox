r"""Tests for the inox.api module."""

from functools import partial

from inox.api import automask, inner, outer
from inox.tree import Mask, Static


def test_automask():
    def is_static(x):
        assert type(x) is Static
        return x

    def is_masked(x):
        assert type(x) is Mask
        return x

    def is_not_masked(x):
        assert type(x) is not Mask
        return x

    # inner & outer
    assert is_masked(inner(is_not_masked)("leaf"))
    assert is_masked(inner(is_not_masked)(Mask("leaf")))
    assert is_static(inner(is_static)(Static("leaf")))
    assert is_not_masked(outer(is_masked)("leaf"))
    assert is_not_masked(outer(is_masked)(Mask("leaf")))
    assert is_static(outer(is_static)(Static("leaf")))

    def f(x):
        return x

    assert outer(inner(f)) is f
    assert inner(outer(f)) is f
    assert inner(f) is inner(f)

    # automask
    assert is_not_masked(automask(partial)(is_not_masked)("leaf"))
    assert is_not_masked(automask(partial)(is_not_masked)(Mask("leaf")))
    assert is_static(automask(partial)(is_static)(Static("leaf")))

    assert automask(lambda f: f)(f) is f
    assert automask(partial)(f) is not f
