r"""Recurrent layers"""

__all__ = [
    'Cell',
    'Recurrent',
    'BRCell',
    'MGUCell',
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


class Cell(Module):
    r"""Abstract cell class.

    A cell defines a recurrence function :math:`f` of the form

    .. math:: (h_i, y_i) = f(h_{i-1}, x_i)

    and an initial hidden state :math:`h_0`.

    Warning:
        The recurrence function :math:`f` should be functionally pure.
    """

    def __call__(self, h: Any, x: Any) -> Tuple[Any, Any]:
        r"""
        Arguments:
            h: The previous hidden state :math:`h_{i-1}`.
            x: The input :math:`x_i`.

        Returns:
            The hidden state and output :math:`(h_i, y_i)`.
        """

        raise NotImplementedError()

    def init(self) -> Any:
        r"""
        Returns:
            The initial hidden state :math:`h_0`.
        """

        raise NotImplementedError()


class Recurrent(Module):
    r"""Creates a recurrent layer.

    Arguments:
        cell: A recurrent cell.
        reverse: Whether to apply the recurrence in reverse or not.
    """

    def __init__(
        self,
        cell: Cell,
        reverse: bool = False,
    ):
        self.cell = cell
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
            f=self.cell,
            init=self.cell.init(),
            xs=xs,
            reverse=self.reverse,
        )

        return ys


class GRUCell(Cell):
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

        self.in_features = in_features
        self.hid_features = hid_features

    def __call__(self, h: Array, x: Array) -> Tuple[Array, Array]:
        r"""
        Arguments:
            h: The previous hidden state :math:`h_{i-1}`, with shape :math:`(*, H)`.
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

    def init(self) -> Array:
        r"""
        Returns:
            The initial hidden state :math:`h_0 = 0`, with shape :math:`(H)`.
        """

        return jnp.zeros(self.hid_features)


class BRCell(Module):
    r"""Creates a bistable recurrent cell (BRC).

    References:
        | A bio-inspired bistable recurrent cell allows for long-lasting memory (Vecoven et al., 2021)
        | https://arxiv.org/abs/2006.05252

    Arguments:
        key: A PRNG key for initialization.
        in_features: The number of input features :math:`C`.
        hid_features: The number of hidden features :math:`H`.
        bias: Whether the cell learns additive biases or not.
        modulated: Whether to use neuromodulation or not.
    """

    def __init__(
        self,
        key: KeyArray,
        in_features: int,
        hid_features: int,
        bias: bool = True,
        modulated: bool = True,
    ):
        keys = jax.random.split(key, 3)

        self.modulated = modulated

        self.lin_x = Linear(keys[0], in_features, 3 * hid_features, bias)

        if self.modulated:
            self.lin_h = Linear(keys[1], hid_features, 2 * hid_features, bias)
        else:
            self.wa = jax.random.normal(keys[1], (hid_features,))
            self.wc = jax.random.normal(keys[2], (hid_features,))

        self.in_features = in_features
        self.hid_features = hid_features

    def __call__(self, h: Array, x: Array) -> Tuple[Array, Array]:
        r"""
        Arguments:
            h: The previous hidden state :math:`h_{i-1}`, with shape :math:`(*, H)`.
            x: The input vector :math:`x_i`, with shape :math:`(*, C)`.

        Returns:
            The hidden state :math:`(h_i, h_i)`.
        """

        if self.modulated:
            ah, ch = jnp.split(self.lin_h(h), 2, axis=-1)
        else:
            ah = self.wa * h
            ch = self.wc * h

        ax, cx, gx = jnp.split(self.lin_x(x), 3, axis=-1)

        a = 1.0 + jax.nn.tanh(ax + ah)
        c = jax.nn.sigmoid(cx + ch)
        g = jax.nn.tanh(gx + a * h)

        h = (1 - c) * g + c * h

        return h, h

    def init(self) -> Array:
        r"""
        Returns:
            The initial hidden state :math:`h_0 = 0`, with shape :math:`(H)`.
        """

        return jnp.zeros(self.hid_features)


class MGUCell(Module):
    r"""Creates a minimal gated unit (MGU) cell.

    References:
        | Minimal Gated Unit for Recurrent Neural Networks (Zhou et al., 2016)
        | https://arxiv.org/pdf/1603.09420

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

        self.lin_fh = Linear(keys[0], hid_features, 1 * hid_features, bias)
        self.lin_hh = Linear(keys[0], hid_features, 1 * hid_features, bias)
        self.lin_x = Linear(keys[1], in_features, 2 * hid_features, bias)

        self.in_features = in_features
        self.hid_features = hid_features

    def __call__(self, h: Array, x: Array) -> Tuple[Array, Array]:
        r"""
        Arguments:
            h: The previous hidden state :math:`h_{i-1}`, with shape :math:`(*, H)`.
            x: The input vector :math:`x_i`, with shape :math:`(*, C)`.

        Returns:
            The hidden state :math:`(h_i, h_i)`.
        """

        fh = self.lin_fh(h)
        fx, gx = jnp.split(self.lin_x(x), 2, axis=-1)
        f = jax.nn.sigmoid(fx + fh)
        gh = self.lin_hh(f * h)
        g = jax.nn.tanh(gx + gh)
        h = (1 - f) * g + f * h

        return h, h

    def init(self) -> Array:
        r"""
        Returns:
            The initial hidden state :math:`h_0 = 0`, with shape :math:`(H)`.
        """

        return jnp.zeros(self.hid_features)


class LSTMCell(Cell):
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

        self.in_features = in_features
        self.hid_features = hid_features

    def __call__(
        self,
        h: Tuple[Array, Array],
        x: Array,
    ) -> Tuple[Tuple[Array, Array], Array]:
        r"""
        Arguments:
            h: The previous hidden and cell states :math:`(h_{i-1}, c_{i-1})`,
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

    def init(self) -> Tuple[Array, Array]:
        r"""
        Returns:
            The initial hidden and cell states :math:`h_0 = c_0 = 0`,
            each with shape :math:`(H)`.
        """

        return jnp.zeros(self.hid_features), jnp.zeros(self.hid_features)
