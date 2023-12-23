r"""Extended utilities for pseudo-random number generation"""

__all__ = [
    'set_seed',
    'get_key',
    'Generator',
]

import jax

from contextlib import contextmanager
from jax.random import KeyArray
from typing import *

from .debug import same_trace
from .tree_util import Namespace


RNG_STATE: KeyArray = None


@contextmanager
def set_seed(seed: KeyArray):
    r"""Context manager that sets the PRNG state.

    See also:
        :func:`get_key`

    Arguments:
        seed: A PRNG seed.

    Example:
        >>> seed = jax.random.key(0)
        >>> with set_seed(seed):
        >>> ... a = jax.random.normal(get_key())
        >>> ... b = jax.random.uniform(get_key())
    """

    global RNG_STATE

    try:
        state, RNG_STATE = RNG_STATE, seed
        yield
    finally:
        RNG_STATE = state


def get_key() -> KeyArray:
    r"""Gets a new key from the current PRNG state.

    See also:
        :func:`set_seed`
    """

    global RNG_STATE
    assert RNG_STATE is not None, "no PRNG seed is set. See 'inox.random.set_seed' for more information."

    old = RNG_STATE
    new, key = jax.random.split(old)

    assert same_trace(old, new), "a PRNG leak was detected. Ensure that 'inox.random.set_seed' and 'inox.random.get_key' are called within the same compilation trace."

    RNG_STATE = new

    return key


class Generator(Namespace):
    r"""Creates a pseudo-random number generator (PRNG).

    This class is a thin wrapper around the :mod:`jax.random` module, and allows to
    generate new PRNG keys or sample from distributions without having to split keys
    with :func:`jax.random.split` by hand.

    Arguments:
        seed: A integer seed or PRNG key.
        kwargs: Keyword arguments passed to :func:`jax.random.key`.

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

    def __init__(self, seed: Union[int, KeyArray], **kwargs):
        if isinstance(seed, int):
            self.state = jax.random.key(seed, **kwargs)
        else:
            self.state = seed

    def __call__(self, num: int = None) -> KeyArray:
        r"""
        Arguments:
            num: The number of keys to generate.

        Returns:
            A new key if :py:`num=None` and a vector of keys otherwise.
        """

        if num is None:
            keys, self.state = jax.random.split(self.state)
        else:
            keys = jax.random.split(self.state, num=num + 1)
            keys, self.state = keys[:-1], keys[-1]

        return keys

    def __getattr__(self, name: str) -> Any:
        attr = getattr(jax.random, name)

        if callable(attr):
            return lambda *args, **kwargs: attr(self(), *args, **kwargs)
        else:
            return attr
