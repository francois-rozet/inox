r"""Attention layers"""

__all__ = [
    "MultiheadAttention",
]

import jax
import jax.numpy as jnp

from einops import rearrange
from jax import Array
from typing import Union

from .linear import Linear
from .module import Module
from ..random import get_rng


class MultiheadAttention(Module):
    r"""Creates a multihead attention layer.

    .. math:: Y = \sum_i
        \mathrm{attention}(X_q W_q^i + b_q^i, X_k W_k^i + b_k^i, X_v W_v^i + b_v^i) W_y^i

    where

    .. math:: \mathrm{attention}(Q, K, V) =
        \mathrm{softmax}\left( \frac{Q K^T}{\sqrt{H}} \right) V

    denotes the scaled dot-product attention.

    References:
        | Attention Is All You Need (Vaswani et al., 2023)
        | https://arxiv.org/abs/1706.03762

    Arguments:
        in_features: The number of input features :math:`C`.
        out_features: The number of output features :math:`C'`.
            If :py:`None`, :math:`C' = C`.
        hid_features: The number of hidden features :math:`H` per head.
            If :py:`None`, :math:`H = \frac{C}{N}`.
        heads: The number of attention heads :math:`N`.
        bias: Whether the layer learns additive biases :math:`(b_q, b_k, b_v)` or not.
        causal: Whether the attention mask is causal or not. If :py:`True`, the
            :math:`i`-th query is only allowed to attend the :math:`j`-th key if
            :math:`j - i \leq T - S`.
        dropout: The dropout rate on attention weights.
        key: A PRNG key for initialization. If :py:`None`,
            :py:`inox.random.get_rng("init")` is used instead.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        hid_features: int = None,
        heads: int = 1,
        bias: bool = True,
        causal: bool = False,
        dropout: Union[float, Array] = 0.0,
        key: Array = None,
    ):
        if key is None:
            keys = get_rng("init").split(4)
        else:
            keys = jax.random.split(key, 4)

        if out_features is None:
            out_features = in_features

        if hid_features is None:
            hid_features = in_features // heads

        self.lin_q = Linear(in_features, hid_features * heads, bias, key=keys[0])
        self.lin_k = Linear(in_features, hid_features * heads, bias, key=keys[1])
        self.lin_v = Linear(in_features, hid_features * heads, bias, key=keys[2])
        self.lin_y = Linear(hid_features * heads, out_features, False, key=keys[3])

        self.heads = heads
        self.causal = causal
        self.dropout = jnp.asarray(dropout)

    def __call__(
        self,
        xq: Array,
        xk: Array = None,
        xv: Array = None,
        mask: Array = None,
        key: Array = None,
    ) -> Array:
        r"""
        Arguments:
            xq: The query tensor :math:`X_q`, with shape :math:`(*, S, C)`.
            xk: The key tensor :math:`X_k`, with shape :math:`(*, T, C)`.
                If :py:`None`, :math:`X_k = X_q`.
            xv: The value tensor :math:`X_v`, with shape :math:`(*, T, C)`.
                If :py:`None`, :math:`X_v = X_k`.
            mask: A boolean attention mask, with shape :math:`(*, N, S, T)`.
                A :py:`False` value indicates that the corresponding attention weight
                is set to :math:`-\infty`.
            key: A PRNG key. If :py:`None`, dropout is not applied.

        Returns:
            The output tensor :math:`Y`, with shape :math:`(*, S, C')`.
        """

        if xk is None:
            xk = xq

        if xv is None:
            xv = xk

        S, T = xq.shape[-2], xk.shape[-2]

        # Project
        q = self.lin_q(xq)
        k = self.lin_k(xk)
        v = self.lin_v(xv)

        q, k, v = [rearrange(x, "... L (N H) -> ... L N H", N=self.heads) for x in (q, k, v)]

        # Mask
        if key is not None:
            shape = jnp.broadcast_shapes(
                (*q.shape[:-3], self.heads, S, 1),
                (*k.shape[:-3], self.heads, 1, T),
                () if mask is None else mask.shape,
            )

            keep = jax.random.bernoulli(key, p=1 - self.dropout, shape=shape)

            if mask is None:
                mask = keep
            else:
                mask = jnp.logical_and(mask, keep)

        # Attention
        if mask is None:
            y = jax.numpy.vectorize(
                lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=self.causal),
                signature="(S,N,H),(T,N,H),(T,N,H)->(S,N,H)",
            )(q, k, v)
        else:
            y = jax.numpy.vectorize(
                lambda q, k, v, mask: jax.nn.dot_product_attention(
                    q, k, v, mask=mask, is_causal=self.causal
                ),
                signature="(S,N,H),(T,N,H),(T,N,H),(N,S,T)->(S,N,H)",
            )(q, k, v, mask)

        y = rearrange(y, "... N H -> ... (N H)")
        y = self.lin_y(y)

        return y
