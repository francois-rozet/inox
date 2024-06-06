r"""Linear layers"""

__all__ = [
    'Linear',
    'Conv',
    'ConvTransposed',
]

import jax
import jax.numpy as jnp
import math

from jax import Array
from typing import Sequence, Tuple, Union

# isort: split
from .module import Module, Parameter
from ..numpy import vectorize
from ..random import get_rng


class Linear(Module):
    r"""Creates a linear layer.

    .. math:: y = W x + b

    Arguments:
        in_features: The number of input features :math:`C`.
        out_features: The number of output features :math:`C'`.
        bias: Whether the layer learns an additive bias :math:`b` or not.
        key: A PRNG key for initialization. If :py:`None`,
            :func:`inox.random.get_rng` is used instead.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        key: Array = None,
    ):
        if key is None:
            key = get_rng().split()

        scale = 1 / math.sqrt(in_features)

        self.weight = Parameter(
            jax.random.uniform(
                key,
                shape=(in_features, out_features),
                minval=-scale,
                maxval=scale,
            )
        )

        if bias:
            self.bias = Parameter(jnp.zeros(out_features))
        else:
            self.bias = None

    def __call__(self, x: Array) -> Array:
        r"""
        Arguments:
            x: The input vector :math:`x`, with shape :math:`(*, C)`.

        Returns:
            The output vector :math:`y`, with shape :math:`(*, C')`.
        """

        if self.bias is None:
            return x @ self.weight()
        else:
            return x @ self.weight() + self.bias()


class Conv(Module):
    r"""Creates a convolution layer.

    .. math:: y = W * x + b

    References:
        | A guide to convolution arithmetic for deep learning (Dumoulin et al., 2016)
        | https://arxiv.org/abs/1603.07285

    Arguments:
        in_channels: The number of input channels :math:`C`.
        out_channels: The number of output channels :math:`C'`.
        kernel_size: The size of the kernel :math:`W` in each spatial axis.
        bias: Whether the layer learns an additive bias :math:`b` or not.
        stride: The stride coefficient in each spatial axis.
        dilation: The dilation coefficient in each spatial axis.
        padding: The padding applied to each end of each spatial axis.
        groups: The number of channel groups :math:`G`.
            Both :math:`C` and :math:`C'` must be divisible by :math:`G`.
        key: A PRNG key for initialization. If :py:`None`,
            :func:`inox.random.get_rng` is used instead.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        bias: bool = True,
        stride: Union[int, Sequence[int]] = 1,
        dilation: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[Tuple[int, int]]] = 0,
        groups: int = 1,
        key: Array = None,
    ):
        if key is None:
            key = get_rng().split()

        if isinstance(stride, int):
            stride = [stride] * len(kernel_size)

        if isinstance(dilation, int):
            dilation = [dilation] * len(kernel_size)

        if isinstance(padding, int):
            padding = [(padding, padding)] * len(kernel_size)

        in_channels = in_channels // groups
        scale = 1 / math.sqrt(math.prod(kernel_size) * in_channels)

        self.kernel = Parameter(
            jax.random.uniform(
                key,
                shape=(*kernel_size, in_channels, out_channels),
                minval=-scale,
                maxval=scale,
            )
        )

        if bias:
            self.bias = Parameter(jnp.zeros(out_channels))
        else:
            self.bias = None

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups

    def __call__(self, x: Array) -> Array:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*, H_1, \dots, H_n, C)`.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*, H_1', \dots, H_n', C')`,
            such that

            .. math:: H_i' =
                \left\lfloor \frac{H_i - d_i \times (k_i - 1) + p_i}{s_i} + 1 \right\rfloor

            where :math:`k_i`, :math:`s_i`, :math:`d_i` and :math:`p_i` are respectively
            the kernel size, the stride coefficient, the dilation coefficient and the
            total padding of the :math:`i`-th spatial axis.
        """

        x = jnp.expand_dims(x, axis=0)
        x = vectorize(jax.lax.conv_general_dilated, ndims=self.ndim + 1)(
            x,
            rhs=self.kernel(),
            dimension_numbers=self.dimensions,
            window_strides=self.stride,
            rhs_dilation=self.dilation,
            padding=self.padding,
            feature_group_count=self.groups,
        )
        x = jnp.squeeze(x, axis=0)

        if self.bias is None:
            return x
        else:
            return x + self.bias()

    @property
    def ndim(self) -> int:
        return len(self.kernel_size) + 1

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
        in_channels: The number of input channels :math:`C`.
        out_channels: The number of output channels  :math:`C'`.
        kernel_size: The size of the kernel :math:`W` in each spatial axis.
        bias: Whether the layer learns an additive bias :math:`b` or not.
        stride: The stride coefficient in each spatial axis.
        dilation: The dilation coefficient in each spatial axis.
        padding: The padding applied to each end of each spatial axis.
        groups: The number of channel groups :math:`G`.
            Both :math:`C` and :math:`C'` must be divisible by :math:`G`.
        key: A PRNG key for initialization. If :py:`None`,
            :func:`inox.random.get_rng` is used instead.
    """

    def __call__(self, x: Array) -> Array:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*, H_1, \dots, H_n, C)`.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*, H_1', \dots, H_n', C')`,
            such that

            .. math:: H_i' = (H_i - 1) \times s_i + d_i \times (k_i - 1) - p_i + 1

            where :math:`k_i`, :math:`s_i`, :math:`d_i` and :math:`p_i` are respectively
            the kernel size, the stride coefficient, the dilation coefficient and the
            total padding of the :math:`i`-th spatial axis.
        """

        x = jnp.expand_dims(x, axis=0)
        x = vectorize(jax.lax.conv_general_dilated, ndims=self.ndim + 1)(
            x,
            rhs=self.kernel(),
            dimension_numbers=self.dimensions,
            window_strides=[1] * (self.ndim - 1),
            padding=self.transposed_padding,
            lhs_dilation=self.stride,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        x = jnp.squeeze(x, axis=0)

        if self.bias is None:
            return x
        else:
            return x + self.bias()

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
