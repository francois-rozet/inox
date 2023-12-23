r"""Padding layers"""

__all__ = [
    'Pad',
]

import jax

from jax import Array
from typing import *

from .module import Module


class Pad(Module):
    r"""Creates a spatial padding layer.

    This module is a thin wrapper around :func:`jax.numpy.pad`.

    Arguments:
        padding: The padding applied to each end of each spatial axis.
        mode: The padding mode in :py:`{'constant', 'edge', 'reflect', 'wrap'}`.
        value: The padding value if :py:`mode='constant`.
    """

    def __init__(
        self,
        padding: Sequence[Tuple[int, int]],
        mode: str = 'constant',
        value: float = 0.0,
    ):
        self.padding = padding
        self.mode = mode
        self.value = value

    @jax.jit
    def __call__(self, x: Array) -> Array:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*, H_1, \dots, H_n, C)`.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*, H_1', \dots, H_n', C)`,
            such that

            .. math:: H_i' = H_i + p_i

            where :math:`p_i` is the total padding of the :math:`i`-th spatial axis.
        """

        batch = x.shape[:-self.ndim]

        x = x.reshape(-1, *x.shape[-self.ndim:])
        x = jax.numpy.pad(
            array=x,
            pad_width=((0, 0), *self.padding, (0, 0)),
            mode=self.mode,
            constant_values=self.value,
        )
        x = x.reshape(*batch, *x.shape[-self.ndim:])

        return x

    @property
    def ndim(self) -> int:
        return len(self.padding) + 1
