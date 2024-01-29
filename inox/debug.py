r"""Extended utilities for debugging"""

__all__ = [
    'same_trace',
]


from jax import Array
from jax.core import Tracer


def same_trace(x: Array, y: Array, ignore_primal: bool = False) -> bool:
    r"""Checks whether two arrays have the same trace source.

    Arguments:
        x: The first array.
        y: The second array.
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
        while hasattr(x, 'primal'):
            x = x.primal

        while hasattr(y, 'primal'):
            y = y.primal

    if isinstance(x, Tracer) and isinstance(y, Tracer):
        return x._trace.main == y._trace.main
    elif isinstance(x, Tracer) or isinstance(y, Tracer):
        return False
    else:
        return True
