r"""Linear layers"""

__all__ = [
    'Linear',
]

import jax

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
        bound = 1 / in_features ** 0.5

        self.weight = jax.random.uniform(
            key=keys[0],
            shape=(in_features, out_features),
            minval=-bound,
            maxval=bound,
        )

        if bias:
            self.bias = jax.random.uniform(
                key=keys[1],
                shape=(out_features,),
                minval=-bound,
                maxval=bound,
            )
        else:
            self.bias = None

    def __call__(self, x: Array) -> Array:
        r""""""

        if self.bias is None:
            return x @ self.weight
        else:
            return x @ self.weight + self.bias
