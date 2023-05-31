r"""Special networks"""

__all__ = [
    'MLP',
]

import jax

from jax import Array
from jax.random import KeyArray
from typing import *

from .activation import *
from .container import *
from .linear import *
from .module import *


class MLP(Sequential):
    r""""""

    def __init__(
        self,
        key: KeyArray,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[], Module] = ReLU,
    ):
        keys = jax.random.split(key, len(hidden_features) + 1)
        layers = []

        for key, before, after in zip(
            keys,
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            layers.append(Linear(key, before, after))
            layers.append(activation())

        super().__init__(*layers[:-1])

    def __call__(self, x: Array) -> Array:
        r""""""

        return super().__call__(x)
