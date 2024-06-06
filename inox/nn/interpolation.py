r"""Interpolation layers"""

__all__ = [
    'Resample',
]

import jax
import math

from functools import partial
from jax import Array
from typing import Sequence

# isort: split
from .module import Module
from ..numpy import vectorize


class Resample(Module):
    r"""Creates a spatial resampling layer.

    Arguments:
        factor: The resampling factor in each spatial axis.
        kwargs: Keyword arguments passed to :func:`jax.image.resize`.
    """

    def __init__(self, factor: Sequence[float], **kwargs):
        self.factor = factor
        self.kwargs = kwargs

    def __call__(self, x: Array) -> Array:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*, H_1, \dots, H_n, C)`.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*, H_1', \dots, H_n', C)`,
            such that

            .. math:: H_i' = \left\lfloor r_i \times H_i \right\rfloor

            where :math:`r_i` is the resampling factor of the :math:`i`-th spatial axis.
        """

        assert x.ndim >= self.ndim

        shape = x.shape[-self.ndim :]
        shape = [math.floor(r * h) for r, h in zip(self.factor, shape)]

        resize = partial(jax.image.resize, shape=shape, **self.kwargs)
        resize = jax.vmap(resize, in_axes=-1, out_axes=-1)
        resize = vectorize(resize, ndims=self.ndim)

        return resize(x)

    @property
    def ndim(self) -> int:
        return len(self.factor) + 1
