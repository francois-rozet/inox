r"""Einstein-like tensor operations

References:
    | Einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation (Rogozhnikov et al., 2022)
    | https://openreview.net/forum?id=oapKSVM2bcj
"""

__all__ = [
    'Rearrange',
    'Reduce',
]

import einops
import jax

from jax import Array
from typing import *

from .module import *


class Rearrange(Module):
    r"""Creates an axis rearrangement layer.

    This module is a thin wrapper around :func:`einops.rearrange`.

    Arguments:
        pattern: The axis rearrangement pattern. For example, the pattern
            :py:`'A B C -> C (A B)'` moves and flattens the two first dimensions.
        lengths: The lengths of the axes.
    """

    def __init__(self, pattern: str, **lengths: int):
        self.pattern = pattern
        self.lengths = lengths

    def __call__(self, x: Array) -> Array:
        return einops.rearrange(x, self.pattern, **self.lengths)


class Reduce(Module):
    r"""Creates an axis reduction layer.

    This module is a thin wrapper around :func:`einops.reduce`.

    Arguments:
        pattern: The axis rearrangement pattern. For example, the pattern
            :py:`'A B C -> C (A B)'` moves and flattens the two first dimensions.
        reduction: The type of reduction (:py:`'sum'`, :py:`'mean'`, :py:`'max'`, ...).
        lengths: The lengths of the axes.
    """

    def __init__(self, pattern: str, reduction: str = 'sum', **lengths: int):
        self.pattern = pattern
        self.reduction = reduction
        self.lengths = lengths

    def __call__(self, x: Array) -> Array:
        return einops.reduce(x, self.pattern, self.reduction, **self.lengths)
