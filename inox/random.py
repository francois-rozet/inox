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
    r"""Creates a pseudo-random number generator (PRNG).

    This class is a thin wrapper around the :mod:`jax.random` module, and allows to
    generate new PRNG keys or sample from distributions without having to split keys
    with :func:`jax.random.split` by hand.

    Arguments:
        seed: A integer seed or PRNG key.

    Example:
        >>> rng = Generator(42)
        >>> rng()  # generates a key
        Array([2465931498, 3679230171], dtype=uint32)
        >>> rng(3)  # generates a vector of 3 keys
        Array([[ 956272045, 3465119146],
               [1903583750,  988321301],
               [3226638877, 2833683589]], dtype=uint32)
        >>> rng.normal((5,))
        Array([-0.08789567,  0.00974573], dtype=float32)
    """

    def __init__(self, seed: Union[int, KeyArray]):
        if isinstance(seed, int):
            self.key = jax.random.PRNGKey(seed)
        else:
            self.key = seed

    def __call__(self, num: int = None) -> KeyArray:
        r"""
        Arguments:
            num: The number of keys to generate.

        Returns:
            A new key if :py:`num=None` and a vector of keys otherwise.
        """

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

    def tree_repr(self, **kwargs) -> str:
        return f'{self.__class__.__name__}(key={self.key})'
