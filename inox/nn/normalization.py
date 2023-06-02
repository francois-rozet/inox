r"""Normalization layers"""

__all__ = [
    'BatchNorm',
    'LayerNorm',
]

import jax
import jax.numpy as jnp

from jax import Array
from typing import *

from .module import *


class BatchNorm(Module):
    r""""""

    def __init__(
        self,
        channels: int,
        epsilon: float = 1e-05,
        momentum: float = 0.9,
    ):
        self.running = Buffer(
            mean=jnp.zeros((channels,)),
            var=jnp.ones((channels,)),
        )

        self.epsilon = epsilon
        self.momentum = momentum
        self.training = True

    def __call__(self, x: Array) -> Array:
        r""""""

        if self.training:
            y = x.reshape(-1, x.shape[-1])

            mean = jnp.mean(y, axis=0)
            var = jnp.var(y, axis=0)

            update = lambda x, y: self.momentum * x + (1 - self.momentum) * y

            self.running.mean = update(self.running.mean, jax.lax.stop_gradient(mean))
            self.running.var = update(self.running.var, jax.lax.stop_gradient(var))
        else:
            mean = self.running.mean
            var = self.running.var

        return (x - mean) / jnp.sqrt(var + self.epsilon)


class LayerNorm(Module):
    r""""""

    def __init__(
        self,
        axis: Union[int, Sequence[int]] = -1,
        epsilon: float = 1e-05,
    ):
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, x: Array) -> Array:
        r""""""

        mean = jnp.mean(x, axis=self.axis, keepdims=True)
        var = jnp.var(x, axis=self.axis, keepdims=True)

        return (x - mean) / jnp.sqrt(var + self.epsilon)
