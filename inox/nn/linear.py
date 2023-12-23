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

from .module import Module


class Linear(Module):
    r"""Creates a linear layer.

    .. math:: y = W \cdot x + b

    Arguments:
        key: A PRNG key for initialization.
        in_features: The number of input features :math:`C`.
        out_features: The number of output features  :math:`C'`.
        bias: Whether the layer learns an additive bias :math:`b` or not.
    """

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

    @jax.jit
    def __call__(self, x: Array) -> Array:
        r"""
        Arguments:
            x: The input vector :math:`x`, with shape :math:`(*, C)`.

        Returns:
            The output vector :math:`y`, with shape :math:`(*, C')`.
        """

        if self.bias is None:
            return x @ self.weight
        else:
            return x @ self.weight + self.bias


class Conv(Module):
    r"""Creates a convolution layer.

    .. math:: y = W * x + b

    References:
        | A guide to convolution arithmetic for deep learning (Dumoulin et al., 2016)
        | https://arxiv.org/abs/1603.07285

    Arguments:
        key: A PRNG key for initialization.
        spatial: The number of spatial axes :math:`S`.
        in_channels: The number of input channels :math:`C`.
        out_channels: The number of output channels  :math:`C'`.
        bias: Whether the layer learns an additive bias :math:`b` or not.
        kernel_size: The size of the kernel :math:`W` in each spatial axis.
        stride: The stride coefficient in each spatial axis.
        dilation: The dilation coefficient in each spatial axis.
        padding: The padding applied to each end of each spatial axis.
        groups: The number of channel groups :math:`G`.
            Both :math:`C` and :math:`C'` must be divisible by :math:`G`.
    """

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

    @jax.jit
    def __call__(self, x: Array) -> Array:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*, H_1, \dots, H_S, C)`.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*, H_1', \dots, H_S', C')`,
            such that

            .. math:: H_i' =
                \left\lfloor \frac{H_i - d_i \times (k_i - 1) + p_i}{s_i} + 1 \right\rfloor

            where :math:`k_i`, :math:`s_i`, :math:`d_i` and :math:`p_i` are respectively
            the kernel size, the stride coefficient, the dilation coefficient and the
            total padding of the :math:`i`-th spatial axis.
        """

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
    r"""Creates a transposed convolution layer.

    This layer can be seen as the gradient of :class:`Conv` with respect to its input.
    It is also known as a "deconvolution", altough it does not actually compute the
    inverse of a convolution.

    References:
        | A guide to convolution arithmetic for deep learning (Dumoulin et al., 2016)
        | https://arxiv.org/abs/1603.07285

    Arguments:
        key: A PRNG key for initialization.
        spatial: The number of spatial axes :math:`S`.
        in_channels: The number of input channels :math:`C`.
        out_channels: The number of output channels  :math:`C'`.
        bias: Whether the layer learns an additive bias :math:`b` or not.
        kernel_size: The size of the kernel :math:`W` in each spatial axis.
        stride: The stride coefficient in each spatial axis.
        dilation: The dilation coefficient in each spatial axis.
        padding: The padding applied to each end of each spatial axis.
        groups: The number of channel groups :math:`G`.
            Both :math:`C` and :math:`C'` must be divisible by :math:`G`.
    """

    @jax.jit
    def __call__(self, x: Array) -> Array:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*, H_1, \dots, H_S, C)`.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*, H_1', \dots, H_S', C')`,
            such that

            .. math:: H_i' = (H_i - 1) \times s_i + d_i \times (k_i - 1) - p_i + 1

            where :math:`k_i`, :math:`s_i`, :math:`d_i` and :math:`p_i` are respectively
            the kernel size, the stride coefficient, the dilation coefficient and the
            total padding of the :math:`i`-th spatial axis.
        """

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
