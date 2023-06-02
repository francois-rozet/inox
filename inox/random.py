r"""Extended utilities for pseudo-random number generation"""

__all__ = [
    'Generator',
]

import jax
import math

from jax.random import KeyArray
from typing import *

from .tree_util import *


class Generator(Namespace):
    r""""""

    def __init__(self, seed: Union[int, KeyArray]):
        if isinstance(seed, int):
            self.key = jax.random.PRNGKey(seed)
        else:
            self.key = seed

    def __call__(self, num: int = None) -> KeyArray:
        r""""""

        if num is None:
            keys, self.key = jax.random.split(self.key)
        else:
            keys = jax.random.split(self.key, num=num + 1)
            keys, self.key = keys[:-1], keys[-1]

        return keys

    def __getattr__(self, name: str) -> Any:
        attr = getattr(jax.random, name)

        if callable(attr):
            return lambda *args, **kwargs: attr(self(), *args, **kwargs)
        else:
            return attr
