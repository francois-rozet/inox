r"""Extended utilities for random number generation"""

__all__ = [
    'PRNG',
    'set_rng',
    'get_rng',
]

import jax

from contextlib import contextmanager
from jax import Array
from typing import Any, Union

# isort: split
from .debug import same_trace
from .tree_util import Namespace


class PRNG(Namespace):
    r"""Creates a pseudo-random number generator (PRNG).

    This class is a thin wrapper around the :mod:`jax.random` module, and allows to
    generate new PRNG keys or sample from distributions without having to split keys
    with :func:`jax.random.split` by hand.

    Arguments:
        seed: An integer seed or PRNG key.
        kwargs: Keyword arguments passed to :func:`jax.random.PRNGKey`.

    Example:
        >>> rng = PRNG(42)
        >>> rng.split()  # generates a key
        Array([2465931498, 3679230171], dtype=uint32)
        >>> rng.split(3)  # generates a vector of 3 keys
        Array([[ 956272045, 3465119146],
               [1903583750,  988321301],
               [3226638877, 2833683589]], dtype=uint32)
        >>> rng.normal((5,))
        Array([ 0.5694761 , -1.4582146 ,  0.2309113 , -0.03029377,  0.11095619], dtype=float32)
    """

    def __init__(self, seed: Union[int, Array], **kwargs):
        if isinstance(seed, int):
            self.state = jax.random.PRNGKey(seed, **kwargs)
        else:
            self.state = seed

    def __getattr__(self, name: str) -> Any:
        attr = getattr(jax.random, name)

        if callable(attr):
            return lambda *args, **kwargs: attr(self.split(), *args, **kwargs)
        else:
            return attr

    def split(self, num: int = None) -> Array:
        r"""
        Arguments:
            num: The number of keys to generate.

        Returns:
            A new key if :py:`num=None` and a vector of keys otherwise.
        """

        if num is None:
            keys = jax.random.split(self.state, num=2)
        else:
            keys = jax.random.split(self.state, num=num + 1)

        assert same_trace(
            self.state, keys
        ), "the PRNG was initialized and used within different JIT traces."

        if num is None:
            key, self.state = keys
        else:
            key, self.state = keys[:-1], keys[-1]

        return key


INOX_RNG: PRNG = None


@contextmanager
def set_rng(rng: PRNG):
    r"""Sets the PRNG within a context.

    See also:
        :class:`PRNG` and :func:`get_rng`

    Arguments:
        rng: A PRNG instance.

    Example:
        >>> with set_rng(PRNG(0)):
        >>> ... a = get_rng().split()
        >>> ... b = get_rng().normal((2, 3))
    """

    global INOX_RNG

    try:
        old, INOX_RNG = INOX_RNG, rng
        yield
    finally:
        INOX_RNG = old


def get_rng() -> PRNG:
    r"""Returns the context-bound PRNG.

    See also:
        :class:`PRNG` and :func:`set_rng`
    """

    global INOX_RNG
    assert INOX_RNG is not None, "no PRNG is set in this context."
    return INOX_RNG
