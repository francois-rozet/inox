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
import jax.numpy as jnp

from jax import Array
from typing import Sequence, Union

# isort: split
from .module import Module


class Activation(Module):
    r"""Abstract activation class."""

    def __init__(self):
        pass


class Identity(Activation):
    r"""Creates an identity activation function.

    .. math:: y = x
    """

    def __call__(self, x: Array) -> Array:
        return x


class Tanh(Activation):
    r"""Creates an identity activation function.

    .. math:: y = \tanh(x)
    """

    def __call__(self, x: Array) -> Array:
        return jnp.tanh(x)


class Sigmoid(Activation):
    r"""Creates a sigmoid activation function.

    .. math:: y = \sigma(x) = \frac{1}{1 + \exp(-x)}
    """

    def __call__(self, x: Array) -> Array:
        return jax.nn.sigmoid(x)


class Softplus(Activation):
    r"""Creates a softplus activation function.

    .. math:: y = \log(1 + \exp(x))
    """

    def __call__(self, x: Array) -> Array:
        return jax.nn.softplus(x)


class Softmax(Activation):
    r"""Creates a softmax activation function.

    .. math:: y_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    Arguments:
        axis: The axis(es) over which the sum is performed.
    """

    def __init__(self, axis: Union[int, Sequence[int]] = -1):
        self.axis = axis

    def __call__(self, x: Array) -> Array:
        return jax.nn.softmax(x, axis=self.axis)

    def tree_repr(self, **kwargs) -> str:
        return f'{self.__class__.__name__}(axis={self.axis})'


class ReLU(Activation):
    r"""Creates a rectified linear unit (ReLU) activation function.

    .. math:: y = \max(x, 0)
    """

    def __call__(self, x: Array) -> Array:
        return jax.nn.relu(x)


class LeakyReLU(Activation):
    r"""Creates a leaky-ReLU activation function.

    .. math:: y = \begin{cases}
            \alpha x & \text{if } x \leq 0 \\
            x & \text{otherwise}
        \end{cases}

    Arguments:
        alpha: The negative slope :math:`\alpha`.
    """

    def __init__(self, alpha: Union[float, Array] = 0.01):
        self.alpha = jnp.asarray(alpha)

    def __call__(self, x: Array) -> Array:
        return jax.nn.leaky_relu(x, self.alpha)

    def tree_repr(self, **kwargs) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha})'


class ELU(Activation):
    r"""Creates an exponential linear unit (ELU) activation function.

    .. math:: y = \begin{cases}
            \alpha (\exp(x) - 1) & \text{if } x \leq 0 \\
            x & \text{otherwise}
        \end{cases}

    References:
        | Fast and Accurate Deep Network Learning by Exponential Linear Units (Clevert et al., 2015)
        | https://arxiv.org/abs/1511.07289

    Arguments:
        alpha: The coefficient :math:`\alpha`.
    """

    def __init__(self, alpha: Union[float, Array] = 1.0):
        self.alpha = jnp.asarray(alpha)

    def __call__(self, x: Array) -> Array:
        return jax.nn.elu(x)

    def tree_repr(self, **kwargs) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha})'


class CELU(ELU):
    r"""Creates a continuously-differentiable ELU (CELU) activation function.

    .. math:: y = \max(x, 0) + \alpha \min(0, \exp(x / \alpha) - 1)

    References:
        | Continuously Differentiable Exponential Linear Units (Barron, 2017)
        | https://arxiv.org/abs/1704.07483

    Arguments:
        alpha: The coefficient :math:`\alpha`.
    """

    def __call__(self, x: Array) -> Array:
        return jax.nn.celu(x)


class GELU(Activation):
    r"""Creates a Gaussian error linear unit (GELU) activation function.

    .. math:: y = \frac{x}{2}
        \left(1 + \mathrm{erf}\left(\frac{x}{\sqrt{2}}\right)\right)

    When :py:`approximate=True`, it is approximated as

    .. math:: y = \frac{x}{2}
        \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)\right)\right)

    References:
        | Gaussian Error Linear Units (Hendrycks et al., 2017)
        | https://arxiv.org/abs/1606.08415v4

    Arguments:
        approximate: Whether to use the approximate or exact formulation.
    """

    def __init__(self, approximate: bool = True):
        self.approximate = approximate

    def __call__(self, x: Array) -> Array:
        return jax.nn.gelu(x, self.approximate)

    def tree_repr(self, **kwargs) -> str:
        return f'{self.__class__.__name__}(approximate={self.approximate})'


class SELU(Activation):
    r"""Creates a self-normalizing ELU (SELU) activation function.

    .. math:: y = \lambda \begin{cases}
            \alpha (\exp(x) - 1) & \text{if } x \leq 0 \\
            x & \text{otherwise}
        \end{cases}

    where :math:`\lambda \approx 1.0507` and :math:`\alpha \approx 1.6732`.

    References:
        | Self-Normalizing Neural Networks (Klambauer et al., 2017)
        | https://arxiv.org/abs/1706.02515
    """

    def __call__(self, x: Array) -> Array:
        return jax.nn.selu(x)


class SiLU(Activation):
    r"""Creates a sigmoid linear unit (SiLU) activation function.

    .. math:: y = x \sigma(x)

    References:
        | Gaussian Error Linear Units (Hendrycks et al., 2017)
        | https://arxiv.org/abs/1606.08415v4

        | Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning (Elfwing et al., 2017)
        | https://arxiv.org/abs/1702.03118
    """

    def __call__(self, x: Array) -> Array:
        return jax.nn.silu(x)
