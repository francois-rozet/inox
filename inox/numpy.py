r"""Extended NumPy interface"""

__all__ = [
    'flatten',
    'unflatten',
    'vectorize',
]

import jax

from functools import partial, wraps
from jax import Array
from typing import *


def flatten(x: Array, start: int = 0, stop: int = None) -> Array:
    r"""Flattens an axis range of an array.

    Arguments:
        x: An array.
        start: The start of the axis range to flatten.
        stop: The end of the axis range to flatten (excluded).
            If :py:`None`, :py:`x.ndim` is used instead.

    Returns:
        The flattened array.

    Example:
        >>> x = jax.numpy.zeros((2, 3, 5))
        >>> flatten(x, 0, 2).shape
        (6, 5)
    """

    if stop is None:
        stop = x.ndim

    return x.reshape(*x.shape[:start], -1, *x.shape[stop:])


def unflatten(x: Array, axis: int, shape: Sequence[int]) -> Array:
    r"""Reshapes an axis of an array.

    Arguments:
        x: An array.
        axis: The axis to reshape.
        shape: The shape of the reshaped axis.

    Returns:
        The array with the reshaped axis.

    Example:
        >>> x = jax.numpy.zeros((6, 5))
        >>> unflatten(x, 0, (2, 3)).shape
        (2, 3, 5)
    """

    return x.reshape(*x.shape[:axis], *shape, *x.shape[axis % x.ndim + 1:])


def vectorize(f: Callable, ndims: Union[int, Sequence[int]]):
    r"""Vectorizes a function with broadcasting.

    :func:`vectorize` is similar to :func:`jax.numpy.vectorize` except that it takes the
    number of core dimensions of arguments as signature instead of their shape.

    Arguments:
        f: A function to vectorize.
        ndims: The number of dimensions expected for each positional argument.

    Returns:
        The vectorized function.

    Example:
        >>> mvp = vectorize(jax.numpy.dot, (2, 1))
        >>> mvp(A, x)  # broadcasting matrix-vector product
    """

    if isinstance(ndims, int):
        ndims = [ndims]

    @wraps(f)
    def wrapped(*args, **kwargs):
        assert len(args) <= len(ndims)
        assert all(0 <= ndim <= arg.ndim for arg, ndim in zip(args, ndims))

        shapes = [arg.shape[:arg.ndim - ndim] for arg, ndim in zip(args, ndims)]
        broadcast = jax.numpy.broadcast_shapes(*shapes)
        squeezed = []

        for arg, shape in zip(args, shapes):
            axes = [i for i, size in enumerate(shape) if size == 1]
            squeezed.append(jax.numpy.squeeze(arg, axes))

        g = partial(f, **kwargs)

        for i, size in enumerate(reversed(broadcast), start=1):
            in_axes = [
                None if len(shape) < i or shape[-i] == 1 else 0
                for shape in shapes
            ]

            g = jax.vmap(g, in_axes=in_axes, axis_size=size)

        return g(*squeezed)

    return wrapped
