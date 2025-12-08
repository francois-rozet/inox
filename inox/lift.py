r"""Lifted user-facing transformations.

The transformations provided in the :mod:`inox.lift` module are lifted versions of
native JAX transformations for which all non-array leaves (:py:`float`, :py:`str`,
functions, ...) are considered static, that is part of the tree structure.
"""

__all__ = [
    "automask",
    "jit",
    "grad",
    "value_and_grad",
    "jacfwd",
    "jacrev",
    "hessian",
    "checkpoint",
    "vmap",
    "pmap",
]

import jax

from functools import cache, wraps
from typing import Callable

from inox.tree import mask, unmask


@cache
def inner(fun: Callable):
    if fun is getattr(fun, "__inner__", None):
        return fun

    if fun is getattr(fun, "__outer__", None):
        return fun.__wrapped__

    @wraps(fun)
    def wrapped(*args, **kwargs):
        return mask(fun(*unmask(args), **unmask(kwargs)))

    wrapped.__inner__ = wrapped

    return wrapped


def outer(fun: Callable):
    if fun is getattr(fun, "__inner__", None):
        return fun.__wrapped__

    if fun is getattr(fun, "__outer__", None):
        return fun

    @wraps(fun)
    def wrapped(*args, **kwargs):
        return unmask(fun(*mask(args), **mask(kwargs)))

    wrapped.__outer__ = wrapped

    return wrapped


def automask(transform: Callable) -> Callable:
    r"""Lifts a transformation to consider all non-array leaves as static.

    For a function :py:`f` and a JAX transformation :py:`jax.tf`,

    .. code-block:: python

        y = automask(jax.tf)(f)(x)

    is equivalent to

    .. code-block:: python

        g = lambda x: mask(f(unmask(x)))
        y = unmask(jax.tf(g)(mask(x)))

    See also:
        :func:`inox.tree.mask` and :func:`inox.tree.unmask`

    Arguments:
        transform: The transformation to lift.

    Returns:
        The lifted transformation.
    """

    @wraps(transform)
    def wrapped(fun: Callable, *args, **kwargs) -> Callable:
        return outer(transform(inner(fun), *args, **kwargs))

    return wrapped


jit = automask(jax.jit)
grad = automask(jax.grad)
value_and_grad = automask(jax.value_and_grad)
jacfwd = automask(jax.jacfwd)
jacrev = automask(jax.jacrev)
hessian = automask(jax.hessian)
checkpoint = automask(jax.checkpoint)
vmap = automask(jax.vmap)
pmap = automask(jax.pmap)
