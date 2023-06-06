r"""Dropout layers"""

__all__ = [
    'Dropout',
]

import jax

from jax import Array
from jax.random import KeyArray
from typing import *

from .module import *
from ..random import *


class Dropout(Module):
    r"""Creates a dropout layer.

    At training,

    .. math:: y = \frac{b x}{q}

    where :math:`b \in \{0, 1\}` is drawn from a Bernoulli distribution such that
    :math:`P(b = 0) = p` and :math:`P(b = 1) = 1 - p = q`. This has proven to be an effective
    technique for regularization and preventing overfitting. At evaluation, the layer
    simply computes the identity :math:`y = x`.

    References:
        | A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al., 2014)
        | https://jmlr.org/papers/v15/srivastava14a

    Arguments:
        key: A PRNG key for initialization.
        p: The masking probability :math:`p \in [0, 1]`.
    """

    def __init__(self, key: KeyArray, p: float = 0.5):
        self.q = 1 - p
        self.training = True
        self.state = Buffer(rng=Generator(key))

    def __call__(self, x: Array) -> Array:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*)`.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*)`.
        """

        if self.training:
            state = self.state
            b = state.rng.bernoulli(shape=x.shape, p=self.q)
            self.state = state

            return jax.numpy.where(b, x / self.q, 0)
        else:
            return x
