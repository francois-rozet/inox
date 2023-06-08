r"""Pooling layers"""

__all__ = [
    'AvgPool',
    'MaxPool',
]

import jax
import math

from functools import wraps
from jax import Array
from typing import *

from .module import *


class Pool(Module):
    r"""Abstract spatial pooling class."""

    def __init__(
        self,
        spatial: int,
        window_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = None,
        padding: Union[int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
    ):
        if isinstance(window_size, int):
            window_size = [window_size] * spatial

        if stride is None:
            stride = window_size
        elif isinstance(stride, int):
            stride = [stride] * spatial

        if isinstance(padding, int):
            padding = [(padding, padding)] * spatial
        else:
            padding = [
                (pad, pad) if isinstance(pad, int) else pad
                for pad in padding
            ]

        self.spatial = spatial
        self.window_size = window_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: Array) -> Array:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*, H_1, \dots, H_S, C)`.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*, H_1', \dots, H_S', C)`,
            such that

            .. math:: H_i' = \left\lfloor \frac{H_i - k_i + p_i}{s_i} + 1 \right\rfloor

            where :math:`k_i`, :math:`s_i` and :math:`p_i` are respectively the window
            size, the stride coefficient and the total padding of the :math:`i`-th
            spatial axis.
        """

        batch = x.shape[:-self.ndim]

        x = x.reshape(-1, *x.shape[-self.ndim:])
        x = jax.lax.reduce_window(
            operand=x,
            init_value=self.initial,
            computation=self.operator,
            window_dimensions=(1, *self.window_size, 1),
            window_strides=(1, *self.stride, 1),
            padding=((0, 0), *self.padding, (0, 0)),
        )
        x = x.reshape(*batch, *x.shape[-self.ndim:])

        return x

    @property
    def ndim(self) -> int:
        return self.spatial + 1


class AvgPool(Pool):
    r"""Creates an average spatial pooling layer.

    Arguments:
        spatial: The number of spatial axes :math:`S`.
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
        spatial: The number of spatial axes :math:`S`.
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
