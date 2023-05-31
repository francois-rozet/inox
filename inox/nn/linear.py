r"""Linear layers"""

__all__ = [
    'Linear',
    'Conv',
    'ConvTransposed',
]

import jax
import math

from jax import Array
from jax.random import KeyArray
from typing import *

from .module import *


class Linear(Module):
    r""""""

    def __init__(
        self,
        key: KeyArray,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        keys = jax.random.split(key, 2)
        lim = 1 / in_features ** 0.5

        self.weight = jax.random.uniform(
            key=keys[0],
            shape=(in_features, out_features),
            minval=-lim,
            maxval=lim,
        )

        if bias:
            self.bias = jax.random.uniform(
                key=keys[1],
                shape=(out_features,),
                minval=-lim,
                maxval=lim,
            )
        else:
            self.bias = None

    def __call__(self, x: Array) -> Array:
        r""""""

        if self.bias is None:
            return x @ self.weight
        else:
            return x @ self.weight + self.bias


class Conv(Module):
    r""""""

    def __init__(
        self,
        key: KeyArray,
        spatial: int,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        dilation: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        groups: int = 1,
    ):
        in_channels = in_channels // groups

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        if isinstance(dilation, int):
            dilation = [dilation] * spatial

        if isinstance(padding, int):
            padding = [(padding, padding)] * spatial
        else:
            padding = tuple(
                (pad, pad) if isinstance(pad, int) else pad
                for pad in padding
            )

        keys = jax.random.split(key, 2)
        lim = 1 / (math.prod(kernel_size) * in_channels) ** 0.5

        self.kernel = jax.random.uniform(
            key=keys[0],
            shape=(*kernel_size, in_channels, out_channels),
            minval=-lim,
            maxval=lim,
        )

        if bias:
            self.bias = jax.random.uniform(
                key=keys[1],
                shape=(out_channels,),
                minval=-lim,
                maxval=lim,
            )
        else:
            self.bias = None

        self.spatial = spatial
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups

    def __call__(self, x: Array) -> Array:
        r""""""

        batch = x.shape[:-self.ndim]

        x = x.reshape(-1, *x.shape[-self.ndim:])
        x = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.kernel,
            dimension_numbers=self.dimensions,
            window_strides=self.stride,
            rhs_dilation=self.dilation,
            padding=self.padding,
            feature_group_count=self.groups,
        )
        x = x.reshape(*batch, *x.shape[-self.ndim:])

        if self.bias is None:
            return x
        else:
            return x + self.bias

    @property
    def ndim(self) -> int:
        return self.spatial + 1

    @property
    def dimensions(self) -> jax.lax.ConvDimensionNumbers:
        return jax.lax.ConvDimensionNumbers(
            (0, self.ndim, *range(1, self.ndim)),
            (self.ndim, self.ndim - 1, *range(0, self.ndim - 1)),
            (0, self.ndim, *range(1, self.ndim)),
        )


class ConvTransposed(Conv):
    r""""""

    def __call__(self, x: Array) -> Array:
        r""""""

        batch = x.shape[:-self.ndim]

        x = x.reshape(-1, *x.shape[-self.ndim:])
        x = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.kernel,
            dimension_numbers=self.dimensions,
            window_strides=[1] * self.spatial,
            padding=self.transposed_padding,
            lhs_dilation=self.stride,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        x = x.reshape(*batch, *x.shape[-self.ndim:])

        if self.bias is None:
            return x
        else:
            return x + self.bias

    @property
    def transposed_padding(self) -> Sequence[Tuple[int, int]]:
        return [
            (d * (k - 1) - p[0], d * (k - 1) - p[1])
            for k, d, p in zip(
                self.kernel_size,
                self.dilation,
                self.padding,
            )
        ]
