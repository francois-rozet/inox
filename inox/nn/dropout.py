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


class Dropout(Buffer):
    r""""""

    def __init__(self, key: KeyArray, p: float = 0.5):
        self.q = 1 - p
        self.rng = Generator(key)
        self.training = True

    def __call__(self, x: Array) -> Array:
        r""""""

        if self.training:
            b = self.rng.bernoulli(shape=x.shape, p=self.q)
            return jax.numpy.where(b, x / self.q, 0)
        else:
            return x
