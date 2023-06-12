r"""Recurrent layers"""

__all__ = [
    'Recurrent',
    'GRUCell',
    'LSTMCell',
]

import jax
import jax.numpy as jnp

from jax import Array
from jax.random import KeyArray
from typing import *

from .module import *
from .linear import Linear


class Recurrent(Module):
    r"""Creates a recurrent layer.

    .. math:: (h_i, y_i) \gets f(h_{i-1}, x_i)

    Warning:
        The recurrence function :math:`f` must be functionally pure.

    Arguments:
        f: The recurrence function :math:`f`.
        h: The initial hidden state :math:`h_0`.
        reverse: Whether to apply the recurrence in reverse or not.
    """

    def __init__(
        self,
        f: Callable[[Any, Any], Tuple[Any, Any]],
        h: Any,
        reverse: bool = False,
    ):
        self.f = f
        self.h = h
        self.reverse = reverse

    def __call__(self, xs: Any) -> Any:
        r"""
        Arguments:
            xs: A sequence of inputs :math:`x_i`, stacked on the leading axis.
                When inputs are vectors, :py:`xs` has shape :math:`(L, C)`.

        Returns:
            A sequence of outputs :math:`y_i`, stacked on the leading axis. When outputs
            are vectors, :py:`ys` has shape :math:`(L, C')`.
        """

        _, ys = jax.lax.scan(
            f=self.f,
            init=self.h,
            xs=xs,
            reverse=self.reverse,
        )

        return ys


class GRUCell(Module):
    r"""Creates a gated recurrent unit (GRU) cell.

    References:
        | Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (Cho et al., 2014)
        | https://arxiv.org/abs/1406.1078

    Arguments:
        key: A PRNG key for initialization.
        in_features: The number of input features :math:`C`.
        hid_features: The number of hidden features :math:`H`.
        bias: Whether the cell learns additive biases or not.
    """

    def __init__(
        self,
        key: KeyArray,
        in_features: int,
        hid_features: int,
        bias: bool = True,
    ):
        keys = jax.random.split(key, 2)

        self.lin_h = Linear(keys[0], hid_features, 3 * hid_features, bias)
        self.lin_x = Linear(keys[1], in_features, 3 * hid_features, bias)

    def __call__(self, h: Array, x: Array) -> Tuple[Array, Array]:
        r"""
        Arguments:
            h: The hidden state :math:`h_{i-1}`, with shape :math:`(*, H)`.
            x: The input vector :math:`x_i`, with shape :math:`(*, C)`.

        Returns:
            The hidden state :math:`(h_i, h_i)`.
        """

        rh, zh, gh = jnp.split(self.lin_h(h), 3, axis=-1)
        rx, zx, gx = jnp.split(self.lin_x(x), 3, axis=-1)

        r = jax.nn.sigmoid(rx + rh)
        z = jax.nn.sigmoid(zx + zh)
        g = jax.nn.tanh(gx + r * gh)

        h = (1 - z) * g + z * h

        return h, h


class LSTMCell(Module):
    r"""Creates a long short-term memory (LSTM) cell.

    References:
        | Long Short-Term Memory (Hochreiter et al., 1997)
        | https://ieeexplore.ieee.org/abstract/document/6795963

    Arguments:
        key: A PRNG key for initialization.
        in_features: The number of input features :math:`C`.
        hid_features: The number of hidden features :math:`H`.
        bias: Whether the cell learns additive biases or not.
    """

    def __init__(
        self,
        key: KeyArray,
        in_features: int,
        hid_features: int,
        bias: bool = True,
    ):
        keys = jax.random.split(key, 2)

        self.lin_h = Linear(keys[0], hid_features, 4 * hid_features, bias)
        self.lin_x = Linear(keys[1], in_features, 4 * hid_features, bias)

    def __call__(
        self,
        h: Tuple[Array, Array],
        x: Array,
    ) -> Tuple[Tuple[Array, Array], Array]:
        r"""
        Arguments:
            h: The hidden and cell states :math:`(h_{i-1}, c_{i-1})`,
                each with shape :math:`(*, H)`.
            x: The input vector :math:`x_i`, with shape :math:`(*, C)`.

        Returns:
            The hidden and cell states :math:`((h_i, c_i), h_i)`.
        """

        h, c = h
        i, f, g, o = jnp.split(self.lin_h(h) + self.lin_x(x), 4, axis=-1)

        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jax.nn.tanh(g)
        o = jax.nn.sigmoid(o)

        c = f * c + i * g
        h = o * jax.nn.tanh(c)

        return (h, c), h
