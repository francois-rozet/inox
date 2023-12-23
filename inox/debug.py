r"""Utilities for debugging"""

__all__ = [
    'same_trace',
]

import jax

from jax import Array
from jax.core import Tracer
from typing import *


def same_trace(a: Array, b: Array, ignore_primal: bool = False) -> bool:
    r"""Checks whether two arrays have the same trace source.

    Arguments:
        a: The first array.
        b: The second array.
        ignore_primal: Whether to ignore primal traces (:func:`jax.grad`).

    Example:
        >>> x, y = jax.numpy.zeros(2)
        >>> same_trace(x, y)
        True
        >>> jax.jit(lambda x, y: same_trace(x, y))(x, y)
        Array(True, dtype=bool)
        >>> jax.jit(lambda x: same_trace(x, y))(x)
        Array(False, dtype=bool)
    """

    if ignore_primal:
        while hasattr(a, 'primal'):
            a = a.primal

        while hasattr(b, 'primal'):
            b = b.primal

    if isinstance(a, Tracer) and isinstance(b, Tracer):
        return a._trace.main == b._trace.main
    elif isinstance(a, Tracer) or isinstance(b, Tracer):
        return False
    else:
        return True
