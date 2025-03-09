r"""Extended utilities for random number generation."""

__all__ = [
    "PRNG",
    "set_rng",
    "get_rng",
]

import jax

from contextlib import contextmanager
from functools import partial
from jax import Array
from typing import Any, Dict, Union

from .debug import same_trace
from .tree import Namespace


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
        Array([1832780943,  270669613], dtype=uint32)
        >>> rng.split(3)  # generates a vector of 3 keys
        Array([[3187376881,  129218101],
               [2350016172, 1168365246],
               [ 257214496,  567757975]], dtype=uint32)
        >>> rng.normal((5,))
        Array([ 0.6611632 , -1.0414096 ,  0.5554834 , -1.8841821 ,  0.36664668],      dtype=float32)
    """

    def __init__(self, seed: Union[int, Array], **kwargs):
        if isinstance(seed, int):
            self.state = jax.random.PRNGKey(seed, **kwargs)
        else:
            self.state = seed

    def __getattr__(self, name: str) -> Any:
        attr = getattr(jax.random, name)

        if callable(attr):
            return partial(attr, self.split())
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

        assert same_trace(self.state, keys), (
            "the RNG state was initialized and used within different JIT traces."
        )

        if num is None:
            key, self.state = keys
        else:
            key, self.state = keys[:-1], keys[-1]

        return key


INOX_RNG: Dict[str, PRNG] = {}


@contextmanager
def set_rng(**rngs: Dict[str, PRNG]):
    r"""Sets named RNG states within a context.

    See also:
        :func:`get_rng`

    Arguments:
        rngs: Named :class:`PRNG` instances.

    Example:
        >>> with set_rng(init=PRNG(0), dropout=PRNG(42)):
        ...     keys = get_rng("init").split(3)
        ...     mask = get_rng("dropout").bernoulli(shape=(2, 3))
    """

    old = {}
    nil = object()

    try:
        for name, rng in rngs.items():
            old[name], INOX_RNG[name] = INOX_RNG.get(name, nil), rng
        yield
    finally:
        for name, rng in old.items():
            if rng is nil:
                del INOX_RNG[name]
            else:
                INOX_RNG[name] = rng


def get_rng(name: str) -> PRNG:
    r"""Returns a context-bound RNG state given its name.

    See also:
        :func:`set_rng`

    Arguments:
        name: The RNG state's name.
    """

    assert name in INOX_RNG, f"no RNG state with name '{name}' is set in this context."

    return INOX_RNG[name]
