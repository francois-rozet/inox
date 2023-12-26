r"""Dropout layers"""

__all__ = [
    'Dropout',
    'TrainingDropout',
]

import jax

from jax import Array
from jax.random import KeyArray
from typing import *

from .module import Module
from ..random import get_rng


class Dropout(Module):
    r"""Creates a dropout layer.

    .. math:: y = \frac{m \odot x}{1 - p}

    where the binary mask :math:`m` is drawn from a Bernoulli distribution such that
    :math:`P(m_i = 0) = p`. This has proven to be an effective technique for
    regularization and preventing overfitting.

    References:
        | A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al., 2014)
        | https://jmlr.org/papers/v15/srivastava14a

    Arguments:
        p: The dropout rate :math:`p \in [0, 1]`.
    """

    def __init__(self, p: float):
        self.p = p

    @jax.jit
    def __call__(self, x: Array, key: KeyArray) -> Array:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*)`.
            key: A PRNG key.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*)`.
        """

        mask = jax.random.bernoulli(key, 1 - self.p, shape=x.shape)

        return jax.numpy.where(mask, x / (1 - self.p), 0)


class TrainingDropout(Dropout):
    r"""Creates a training-bound dropout layer.

    When :py:`self.training = False`,

    .. math:: y = x

    See also:
        :class:`Dropout`

    Arguments:
        p: The dropout rate :math:`p \in [0, 1]`.
    """

    training: bool = True

    def __call__(self, x: Array, key: KeyArray = None) -> Array:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*)`.
            key: A PRNG key. If :py:`None`, :func:`inox.random.get_rng` is used instead.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*)`.
        """

        if self.training:
            if key is None:
                key = get_rng().split()

            return super().__call__(x, key)
        else:
            return x
