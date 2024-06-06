r"""Pooling layers"""

__all__ = [
    'AvgPool',
    'MaxPool',
]

import jax
import math

from functools import wraps
from jax import Array
from typing import Callable, Sequence, Tuple, Union

# isort: split
from .module import Module
from ..numpy import vectorize


class Pool(Module):
    r"""Abstract spatial pooling class."""

    def __init__(
        self,
        window_size: Sequence[int],
        stride: Union[int, Sequence[int]] = None,
        padding: Union[int, Sequence[Tuple[int, int]]] = 0,
    ):
        if stride is None:
            stride = window_size
        elif isinstance(stride, int):
            stride = [stride] * len(window_size)

        if isinstance(padding, int):
            padding = [(padding, padding)] * len(window_size)

        self.window_size = window_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: Array) -> Array:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*, H_1, \dots, H_n, C)`.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*, H_1', \dots, H_n', C)`,
            such that

            .. math:: H_i' = \left\lfloor \frac{H_i - k_i + p_i}{s_i} + 1 \right\rfloor

            where :math:`k_i`, :math:`s_i` and :math:`p_i` are respectively the window
            size, the stride coefficient and the total padding of the :math:`i`-th
            spatial axis.
        """

        return vectorize(jax.lax.reduce_window, ndims=self.ndim)(
            x,
            init_value=self.initial,
            computation=self.operator,
            window_dimensions=(*self.window_size, 1),
            window_strides=(*self.stride, 1),
            padding=(*self.padding, (0, 0)),
        )

    @property
    def ndim(self) -> int:
        return len(self.window_size) + 1


class AvgPool(Pool):
    r"""Creates an average spatial pooling layer.

    Arguments:
        window_size: The size of the pooling window in each spatial axis.
        stride: The stride coefficient in each spatial axis.
        padding: The padding applied to each end of each spatial axis.
    """

    @wraps(Pool.__call__)
    def __call__(self, x: Array) -> Array:
        return super().__call__(x) / math.prod(self.window_size)

    @property
    def operator(self) -> Callable[[Array, Array], Array]:
        return jax.lax.add

    @property
    def initial(self) -> float:
        return 0.0


class MaxPool(Pool):
    r"""Creates a maximum spatial pooling layer.

    Arguments:
        window_size: The size of the pooling window in each spatial axis.
        stride: The stride coefficient in each spatial axis.
        padding: The padding applied to each end of each spatial axis.
    """

    @wraps(Pool.__call__)
    def __call__(self, x: Array) -> Array:
        return super().__call__(x)

    @property
    def operator(self) -> Callable[[Array, Array], Array]:
        return jax.lax.max

    @property
    def initial(self) -> float:
        return -math.inf
