r"""Pooling layers"""

__all__ = [
    'AvgPool',
    'MaxPool',
]

import jax
import math

from jax import Array
from typing import *

from .module import *


class Pool(Module):
    r""""""

    def __init__(
        self,
        spatial: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = None,
        padding: Union[int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
    ):
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if stride is None:
            stride = kernel_size
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
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: Array) -> Array:
        r""""""

        batch = x.shape[:-self.ndim]

        x = x.reshape(-1, *x.shape[-self.ndim:])
        x = jax.lax.reduce_window(
            operand=x,
            init_value=self.initial,
            computation=self.operator,
            window_dimensions=(1, *self.kernel_size, 1),
            window_strides=(1, *self.stride, 1),
            padding=((0, 0), *self.padding, (0, 0)),
        )
        x = x.reshape(*batch, *x.shape[-self.ndim:])

        return x

    @property
    def ndim(self) -> int:
        return self.spatial + 1


class AvgPool(Pool):
    r""""""

    def __call__(self, x: Array) -> Array:
        r""""""

        return super().__call__(x) / math.prod(self.kernel_size)

    @staticmethod
    def operator(x: Array, y: Array) -> Array:
        return jax.lax.add(x, y)

    @property
    def initial(self) -> float:
        return 0.0


class MaxPool(Pool):
    r""""""

    @staticmethod
    def operator(x: Array, y: Array) -> Array:
        return jax.lax.max(x, y)

    @property
    def initial(self) -> float:
        return -math.inf
