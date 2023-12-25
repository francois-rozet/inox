r"""Dropout layers"""

__all__ = [
    'Dropout',
]

import jax

from jax import Array
from jax.random import KeyArray
from typing import *

from .module import Module
from ..random import get_key


class Dropout(Module):
    r"""Creates a dropout layer.

    At training,

    .. math:: y = \frac{m \odot x}{1 - p}

    where the binary mask :math:`m` is drawn from a Bernoulli distribution such that
    :math:`P(m_i = 0) = p`. This has proven to be an effective technique for
    regularization and preventing overfitting. At evaluation, the layer simply computes
    the identity :math:`y = x`.

    References:
        | A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al., 2014)
        | https://jmlr.org/papers/v15/srivastava14a

    Arguments:
        p: The masking probability :math:`p \in [0, 1]`.
    """

    training: bool = True

    def __init__(self, p: float):
        self.p = p

    def __call__(self, x: Array, key: KeyArray = None) -> Array:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*)`.
            key: A PRNG key. If :py:`None`, :func:`inox.random.get_key` is used instead.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*)`.
        """

        if self.training and key is None:
            key = get_key()

        return self._call_(x, key)

    @jax.jit
    def _call_(self, x: Array, key: KeyArray = None) -> Array:
        if self.training:
            mask = jax.random.bernoulli(key, 1 - self.p, shape=x.shape)

            return jax.numpy.where(mask, x / (1 - self.p), 0)
        else:
            return x
