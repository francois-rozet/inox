r"""Attention layers"""

__all__ = [
    'MultiheadAttention',
]

import jax
import math

from einops import rearrange
from jax import Array
from jax.random import KeyArray
from typing import *

from .linear import Linear
from .module import Module
from ..random import get_key


@jax.jit
def attention(
    q: Array,
    k: Array,
    v: Array,
    mask: Array = None,
) -> Array:
    r"""Computes the scaled dot-product attention.

    Arguments:
        q: The query tensor :math:`Q`, with shape :math:`(*, S, C)`.
        k: The key tensor :math:`K`, with shape :math:`(*, T, C)`.
        v: The value tensor :math:`V`, with shape :math:`(*, T, C')`.
        mask: A boolean attention mask, with shape :math:`(*, S, T)`.
            A :py:`False` value indicates that the corresponding attention weight
            is set to :math:`-\infty`.

    Returns:
        The output vector :math:`y`, with shape :math:`(*, S, C')`.
    """

    C = q.shape[-1]

    weight = jax.numpy.einsum('...ik,...jk->...ij', q, k)
    weight = weight / math.sqrt(C)

    if mask is not None:
        weight = jax.numpy.where(mask, weight, -1e9)

    attn = jax.nn.softmax(weight, axis=-1)

    return jax.numpy.einsum('...ij,...jk->...ik', attn, v)


class MultiheadAttention(Module):
    r"""Creates a multihead attention layer.

    .. math:: Y = \sum_i
        \mathrm{attention}(X_q W_q^i + b_q^i, X_k W_k^i + b_k^i, X_v W_v^i) W_y^i

    where

    .. math:: \mathrm{attention}(Q, K, V) =
        \mathrm{softmax}\left( \frac{Q K^T}{\sqrt{H}} \right) V

    denotes the scaled dot-product attention.

    References:
        | Attention Is All You Need (Vaswani et al., 2023)
        | https://arxiv.org/abs/1706.03762

    Arguments:
        heads: The number of attention heads.
        in_features: The number of input features :math:`C`.
        hid_features: The number of hidden features :math:`H` per head.
        out_features: The number of output features :math:`C'`.
            If :py:`None`, :math:`C' = C`.
        bias: Whether the layer learns additive biases :math:`(b_q, b_k)` or not.
        causal: Whether the attention mask is causal or not. If :py:`True`, the
            :math:`i`-th query is only allowed to attend the :math:`j`-th key if
            :math:`j - i \leq T - S`.
        dropout: The dropout probability on attention weights.
    """

    training: bool = True

    def __init__(
        self,
        key: KeyArray,
        heads: int,
        in_features: int,
        hid_features: int,
        out_features: int = None,
        bias: bool = True,
        causal: bool = False,
        dropout: float = 0.0,
    ):
        keys = jax.random.split(key, 4)

        if out_features is None:
            out_features = in_features

        self.lin_q = Linear(keys[0], in_features, hid_features * heads, bias=bias)
        self.lin_k = Linear(keys[1], in_features, hid_features * heads, bias=bias)
        self.lin_v = Linear(keys[2], in_features, hid_features * heads, bias=False)
        self.lin_y = Linear(keys[3], hid_features * heads, out_features, bias=False)

        self.heads = heads
        self.causal = causal
        self.dropout = dropout

    def __call__(
        self,
        xq: Array,
        xk: Array = None,
        xv: Array = None,
        mask: Array = None,
        key: KeyArray = None,
    ) -> Array:
        r"""
        Arguments:
            xq: The query tensor :math:`X_q`, with shape :math:`(*, S, C)`.
            xk: The key tensor :math:`X_k`, with shape :math:`(*, T, C)`.
                If :py:`None`, :math:`X_k = X_q`.
            xv: The value tensor :math:`X_v`, with shape :math:`(*, T, C)`.
                If :py:`None`, :math:`X_v = X_k`.
            mask: A boolean attention mask, with shape :math:`(*, S, T)`.
                A :py:`False` value indicates that the corresponding attention weight
                is set to :math:`-\infty`.
            key: A PRNG key. If :py:`None`, :func:`inox.random.get_key` is used instead.

        Returns:
            The output tensor :math:`Y`, with shape :math:`(*, S, C')`.
        """

        if self.training and self.dropout > 0.0 and key is None:
            key = get_key()

        return self._call_(xq, xk, xv, mask, key)

    @jax.jit
    def _call_(
        self,
        xq: Array,
        xk: Array = None,
        xv: Array = None,
        mask: Array = None,
        key: KeyArray = None,
    ) -> Array:
        if xk is None:
            xk = xq

        if xv is None:
            xv = xk

        S, T = xq.shape[-2], xk.shape[-2]

        # Project
        q = self.lin_q(xq)
        k = self.lin_k(xk)
        v = self.lin_v(xv)

        q, k, v = [
            rearrange(x, '... L (N H) -> ... N L H', H=self.heads)
            for x in (q, k, v)
        ]

        # Mask
        if self.causal:
            if mask is None:
                mask = jax.numpy.ones((S, T), dtype=bool)

            mask = jax.numpy.tril(mask, T - S)

        if self.training and self.dropout > 0.0:
            shape = jax.numpy.broadcast_shapes(
                (*q.shape[:-2], S, 1),
                (*k.shape[:-2], 1, T),
                (S, T) if mask is None else mask.shape,
            )

            keep = jax.random.bernoulli(key, p=1 - self.dropout, shape=shape)

            if mask is None:
                mask = keep
            else:
                mask = jax.numpy.logical_and(mask, keep)

        # Attention
        y = attention(q, k, v, mask)
        y = rearrange(y, '... N L H -> ... L (N H)')
        y = self.lin_y(y)

        return y
