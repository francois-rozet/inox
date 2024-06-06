r"""Recurrent layers"""

__all__ = [
    'Cell',
    'Recurrent',
    'GRUCell',
    'LSTMCell',
]

import jax
import jax.numpy as jnp

from jax import Array
from typing import Any, Tuple

# isort: split
from .linear import Linear
from .module import Module
from ..random import get_rng


class Cell(Module):
    r"""Abstract cell class.

    A cell defines a recurrence function :math:`f` of the form

    .. math:: (h_i, y_i) = f(h_{i-1}, x_i)

    and an initial hidden state :math:`h_0`.

    Warning:
        The recurrence function :math:`f` should have no side effects.
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
        in_features: The number of input features :math:`C`.
        hid_features: The number of hidden features :math:`H`.
        bias: Whether the cell learns additive biases or not.
        key: A PRNG key for initialization. If :py:`None`,
            :func:`inox.random.get_rng` is used instead.
    """

    def __init__(
        self,
        in_features: int,
        hid_features: int,
        bias: bool = True,
        key: Array = None,
    ):
        if key is None:
            keys = get_rng().split(2)
        else:
            keys = jax.random.split(key, 2)

        self.lin_h = Linear(hid_features, 3 * hid_features, bias, key=keys[0])
        self.lin_x = Linear(in_features, 3 * hid_features, bias, key=keys[1])

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


class LSTMCell(Cell):
    r"""Creates a long short-term memory (LSTM) cell.

    References:
        | Long Short-Term Memory (Hochreiter et al., 1997)
        | https://ieeexplore.ieee.org/abstract/document/6795963

    Arguments:
        in_features: The number of input features :math:`C`.
        hid_features: The number of hidden features :math:`H`.
        bias: Whether the cell learns additive biases or not.
        key: A PRNG key for initialization. If :py:`None`,
            :func:`inox.random.get_rng` is used instead.
    """

    def __init__(
        self,
        in_features: int,
        hid_features: int,
        bias: bool = True,
        key: Array = None,
    ):
        if key is None:
            keys = get_rng().split(2)
        else:
            keys = jax.random.split(key, 2)

        self.lin_h = Linear(hid_features, 4 * hid_features, bias, key=keys[0])
        self.lin_x = Linear(in_features, 4 * hid_features, bias, key=keys[1])

        self.hid_features = hid_features

    def __call__(
        self,
        hc: Tuple[Array, Array],
        x: Array,
    ) -> Tuple[Tuple[Array, Array], Array]:
        r"""
        Arguments:
            hc: The previous hidden and cell states :math:`(h_{i-1}, c_{i-1})`,
                each with shape :math:`(*, H)`.
            x: The input vector :math:`x_i`, with shape :math:`(*, C)`.

        Returns:
            The hidden and cell states :math:`((h_i, c_i), h_i)`.
        """

        h, c = hc
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
