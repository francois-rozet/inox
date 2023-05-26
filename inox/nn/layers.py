r"""Basic layers"""

__all__ = [
    'Linear',
    'ReLU',
    'MLP'
]

import jax
import math

from jax import Array
from jax.random import PRNGKey
from typing import *

from .base import *


class Linear(Module):
    r""""""

    weight: Array
    bias: Array

    def __init__(self, key: PRNGKey, in_size: int, out_size: int):
        k1, k2 = jax.random.split(key, 2)
        b = 1 / math.sqrt(in_size)

        self.weight = jax.random.uniform(k1, (in_size, out_size), minval=-b, maxval=b)
        self.bias = jax.random.uniform(k2, (out_size,), minval=-b, maxval=b)

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.weight + self.bias


class ReLU(Module):
    r""""""

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.relu(x)


class MLP(Sequential):
    r""""""

    def __init__(
        self,
        key: PRNGKey,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[], Module] = ReLU,
    ):
        layers = []

        for before, after in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            key, subkey = jax.random.split(key, 2)

            layers.extend((
                Linear(subkey, before, after),
                activation(),
            ))

        layers.pop()

        super().__init__(*layers)
