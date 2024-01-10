r"""Normalization layers"""

__all__ = [
    'BatchNorm',
    'LayerNorm',
]

import jax
import jax.numpy as jnp

from jax import Array
from typing import *

from .module import Module
from .state import StateEntry, update_state


class Statistics(NamedTuple):
    mean: Array
    var: Array


class BatchNorm(Module):
    r"""Creates a batch-normalization layer.

    .. math:: y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathbb{V}[x] + \epsilon}}

    The mean and variance are calculated over the batch and spatial axes. During
    training, the layer keeps running estimates of the mean and variance, which are then
    used for normalization during evaluation. The update rule for a running average
    statistic :math:`\hat{s}` is

    .. math:: \hat{s} \gets \alpha \hat{s} + (1 - \alpha) s

    where :math:`s` is the statistic calculated for the current batch.

    References:
        | Accelerating Deep Network Training by Reducing Internal Covariate Shift (Ioffe et al., 2015)
        | https://arxiv.org/abs/1502.03167

    Arguments:
        channels: The number of channels :math:`C`.
        epsilon: A numerical stability term :math:`\epsilon`.
        momentum: The momentum :math:`\alpha \in [0, 1]` for the running estimates.
    """

    training: bool = True

    def __init__(
        self,
        channels: int,
        epsilon: float = 1e-05,
        momentum: float = 0.9,
    ):
        self.epsilon = epsilon
        self.momentum = momentum

        self.stats = StateEntry(
            Statistics(
                mean=jnp.zeros(channels),
                var=jnp.ones(channels),
            )
        )

    def __call__(self, x: Array, state: Dict) -> Tuple[Array, Dict]:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(N, *, C)`.
            state: The state dictionary.

        Returns:
            The output tensor :math:`y`, with shape :math:`(N, *, C)`, and the
            (updated) state dictionary.
        """

        if self.training:
            assert x.ndim > 1, "the input tensor is not batched."

            y = x.reshape(-1, x.shape[-1])
            mean = jnp.mean(y, axis=0)
            var = jnp.var(y, axis=0)

            stats = state[self.stats]
            stats = Statistics(
                mean=self.ema(stats.mean, jax.lax.stop_gradient(mean)),
                var=self.ema(stats.var, jax.lax.stop_gradient(var)),
            )

            state = update_state(state, {self.stats: stats})
        else:
            mean, var = state[self.stats]

        y = (x - mean) / jnp.sqrt(var + self.epsilon)

        return y, state

    def ema(self, x: Array, y: Array) -> Array:
        return self.momentum * x + (1 - self.momentum) * y


class LayerNorm(Module):
    r"""Creates a layer-normalization layer.

    .. math:: y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathbb{V}[x] + \epsilon}}

    References:
        | Layer Normalization (Ba et al., 2016)
        | https://arxiv.org/abs/1607.06450

    Arguments:
        axis: The axis(es) over which the mean and variance are calculated.
        epsilon: A numerical stability term :math:`\epsilon`.
    """

    def __init__(
        self,
        axis: Union[int, Sequence[int]] = -1,
        epsilon: float = 1e-05,
    ):
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, x: Array) -> Array:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*, C)`.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*, C)`.
        """

        mean = jnp.mean(x, axis=self.axis, keepdims=True)
        var = jnp.var(x, axis=self.axis, keepdims=True)

        return (x - mean) / jnp.sqrt(var + self.epsilon)
