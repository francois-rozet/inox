r"""Spectral layers"""

__all__ = [
    'SpectralConv',
]

import jax
import jax.numpy as jnp
import math

from jax import Array
from typing import Sequence

# isort: local
from .module import ComplexParameter, Module
from ..random import get_rng


class SpectralConv(Module):
    r"""Creates a spectral convolution layer.

    .. math:: y = \mathcal{F}^{-1}(W \mathcal{F}(x) + b)

    where :math:`\mathcal{F}` is the discrete Fourier transform.

    References:
        | Fourier Neural Operator for Parametric Partial Differential Equations (Li et al., 2020)
        | https://arxiv.org/abs/2010.08895

    Arguments:
        in_channels: The number of input channels :math:`C`.
        out_channels: The number of output channels :math:`C'`.
        modes: The number of spectral modes of the kernel :math:`W` in each axis.
        bias: Whether the layer learns an additive bias :math:`b` or not.
        key: A PRNG key for initialization. If :py:`None`,
            :func:`inox.random.get_rng` is used instead.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
        bias: bool = True,
        key: Array = None,
    ):
        if key is None:
            key = get_rng().split()

        kernel_size = [2 * m + 1 for m in modes]

        self.kernel = ComplexParameter(
            jax.random.normal(
                key,
                shape=(*kernel_size, in_channels, out_channels),
                dtype=complex,
            )
            / math.sqrt(in_channels)
        )

        if bias:
            self.bias = ComplexParameter(jnp.zeros((*kernel_size, out_channels)))
        else:
            self.bias = None

        self.modes = modes

    def __call__(self, x: Array) -> Array:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*, H_1, \dots, H_n, C)`.
                Floating point arrays are promoted to complex arrays.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*, H_1, \dots, H_n, C')`.
        """

        assert x.ndim >= self.ndim

        index = (..., *(slice(2 * m + 1) for m in self.modes), slice(None))

        x = jnp.fft.fftn(x, norm='forward', axes=self.axes)
        x = jnp.roll(x, shift=self.modes, axis=self.axes)

        y = jnp.einsum('...i,...ij->...j', x[index], self.kernel())

        if self.bias is not None:
            y = y + self.bias()

        y = jnp.zeros(x.shape[:-1] + y.shape[-1:], dtype=y.dtype).at[index].set(y)
        y = jnp.roll(y, shift=(-m for m in self.modes), axis=self.axes)
        y = jnp.fft.ifftn(y, norm='forward', axes=self.axes)

        return y

    @property
    def ndim(self) -> int:
        return len(self.modes) + 1

    @property
    def axes(self) -> Sequence[int]:
        return range(-self.ndim, -1)
