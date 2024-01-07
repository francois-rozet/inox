r"""Extended user-facing transformations and utilities

The transformations provided by :mod:`inox.api` are lifted versions of native JAX
transformations for which all non-array leaves (:py:`float`, :py:`str`, functions, ...)
are considered static, that is part of the tree structure.

Roughly, for a function :py:`f` and a JAX transformation :py:`jax.transform`,

.. code-block:: python

    y = inox.transform(f)(x)

is equivalent to

.. code-block:: python

    g = lambda x: inox.tree_mask(f(inox.tree_unmask(x)))
    y = inox.tree_unmask(jax.transform(g)(inox.tree_mask(x)))

Descriptions
------------
"""

__all__ = [
    'jit',
    'grad',
    'value_and_grad',
    'jacfwd',
    'jacrev',
    'hessian',
    'checkpoint',
    'vmap',
]

import jax

from functools import cache, wraps
from jax._src.api import api_boundary
from typing import *

from .tree_util import tree_mask, tree_unmask


@cache
def inner(fun: Callable):
    @wraps(fun)
    def wrapped(*args, **kwargs):
        return tree_mask(fun(*tree_unmask(args), **tree_unmask(kwargs)))

    return wrapped


def outer(fun: Callable):
    @wraps(fun)
    def wrapped(*args, **kwargs):
        return tree_unmask(fun(*tree_mask(args), **tree_mask(kwargs)))

    return wrapped


def masked(transform: Callable) -> Callable:
    @wraps(transform)
    @api_boundary
    def wrapped(fun: Callable, *args, **kwargs) -> Callable:
        return outer(transform(inner(fun), *args, **kwargs))

    return wrapped


jit = masked(jax.jit)
grad = masked(jax.grad)
value_and_grad = masked(jax.value_and_grad)
jacfwd = masked(jax.jacfwd)
jacrev = masked(jax.jacrev)
hessian = masked(jax.hessian)
checkpoint = masked(jax.checkpoint)
vmap = masked(jax.vmap)
