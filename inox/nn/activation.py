r"""Activation functions"""

__all__ = [
    'Identity',
    'Tanh',
    'Sigmoid',
    'SiLU',
    'Softplus',
    'Softmax',
    'ReLU',
    'LeakyReLU',
    'ELU',
    'CELU',
    'GELU',
    'SELU',
    'SiLU',
]

import jax

from jax import Array
from typing import *

from .module import *


class Activation(Module):
    r""""""

    def __init__(self):
        pass


class Identity(Activation):
    r""""""

    def __call__(self, x: Array) -> Array:
        return x


class Tanh(Activation):
    r""""""

    def __call__(self, x: Array) -> Array:
        return jax.numpy.tanh(x)


class Sigmoid(Activation):
    r""""""

    def __call__(self, x: Array) -> Array:
        return jax.nn.sigmoid(x)


class Softplus(Activation):
    r""""""

    def __call__(self, x: Array) -> Array:
        return jax.nn.softplus(x)


class Softmax(Activation):
    r""""""

    def __init__(self, axis: Union[int, Sequence[int]] = -1):
        self.axis = axis

    def __call__(self, x: Array) -> Array:
        return jax.nn.softmax(x, axis=self.axis)

    def tree_repr(self, **kwargs) -> str:
        return f'{self.__class__.__name__}(axis={self.axis})'


class ReLU(Activation):
    r""""""

    def __call__(self, x: Array) -> Array:
        return jax.nn.relu(x)


class LeakyReLU(Activation):
    r""""""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def __call__(self, x: Array) -> Array:
        return jax.nn.leaky_relu(x, self.alpha)

    def tree_repr(self, **kwargs) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha})'


class ELU(Activation):
    r""""""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, x: Array) -> Array:
        return jax.nn.elu(x)

    def tree_repr(self, **kwargs) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha})'


class CELU(Activation):
    r""""""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, x: Array) -> Array:
        return jax.nn.celu(x)

    def tree_repr(self, **kwargs) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha})'


class GELU(Activation):
    r""""""

    def __init__(self, approximate: bool = True):
        self.approximate = approximate

    def __call__(self, x: Array) -> Array:
        return jax.nn.gelu(x, self.approximate)

    def tree_repr(self, **kwargs) -> str:
        return f'{self.__class__.__name__}(approximate={self.approximate})'


class SELU(Activation):
    r""""""

    def __call__(self, x: Array) -> Array:
        return jax.nn.selu(x)


class SiLU(Activation):
    r""""""

    def __call__(self, x: Array) -> Array:
        return jax.nn.silu(x)
