r"""State space model (SSM) layers"""

__all__ = [
    'S4',
    'S4Cell',
]

import jax
import jax.numpy as jnp
import math
import numpy as np

from jax import Array
from jax.random import KeyArray
from typing import *

from .module import *
from .recurrent import Cell


class S4(Module):
    r"""TODO

    .. math::
        h'(t) & = A h(t) + B x(t) \\
        y(t) & = C h(t)

    References:
        | Efficiently Modeling Long Sequences with Structured State Spaces (Gu et al., 2021)
        | https://arxiv.org/abs/2111.00396

        | The Annotated S4 (Rush et al., 2023)
        | https://srush.github.io/annotated-s4

    Arguments:
        key: A PRNG key for initialization.
        hid_features: The number of hidden features :math:`H`.
    """

    def __init__(self, key: KeyArray, hid_features: int):
        keys = jax.random.split(key, 3)

        A, P = S4.DPLR_HiPPO(hid_features)

        self.A_re = jnp.log(-A.real)
        self.A_im = A.imag
        self.P = P
        self.B = jax.random.normal(keys[0], (hid_features,), dtype=complex)
        self.C = jax.random.normal(keys[1], (hid_features,), dtype=complex)
        self.log_dt = jax.random.uniform(
            keys[2],
            (),
            minval=math.log(1e-3),
            maxval=math.log(1e-1),
        )

    def __call__(self, xs: Array) -> Array:
        r"""TODO

        .. math:: y_i = \sum_{j = 1}^{i} \bar{C} \bar{A}{i-j} \bar{B} x_j

        Arguments:
            xs: The sequence of inputs :math:`x_i`, with shape :math:`(L)`.

        Returns:
            The sequence of outputs :math:`y_i`, with shape :math:`(L)`.
        """

        L = xs.shape[0]
        k = self.kernel(L)

        return jax.scipy.signal.fftconvolve(xs, k)[:L]

    @staticmethod
    def DPLR_HiPPO(n: int) -> Tuple[Array, Array]:
        r"""Returns the diagonal plus low-rank (DPLR) form of the HiPPO matrix.

        .. math:: A = \Lambda - PP^*
        """

        P = np.sqrt(np.arange(n) + 1 / 2)
        S = np.outer(P, P.conj())
        S = np.tril(S) - np.triu(S)

        # Diagonal A
        A_real = -np.ones(n) / 2
        A_imag, V = np.linalg.eigh(-1j * S)
        A = A_real + 1j * A_imag

        # Project P
        P = V.T.conj() @ P

        return jnp.asarray(A), jnp.asarray(P)

    def discrete(self) -> Tuple[Array, Array, Array]:
        r"""Returns :math:`\bar{A}`, :math:`\bar{B}` and :math:`\bar{C}`."""

        A = -jnp.exp(self.A_re) + 1j * self.A_im
        P, B, C = self.P, self.B, self.C
        dt = jnp.exp(self.log_dt)

        D = jnp.diag(1 / (2 / dt - A))
        PQ = jnp.outer(P, P.conj())
        A0 = jnp.diag(2 / dt + A) - PQ
        A1 = D - D @ PQ / (1 + jnp.dot(P.conj(), D @ P)) @ D

        Ab = A1 @ A0
        Bb = 2 * A1 @ B

        return Ab, Bb, C

    def kernel(self, L: int) -> Array:
        r"""Returns the kernel :math:`\bar{k}` of length :math:`L`.

        .. math:: \bar{k} = (\bar{C} \bar{B}, \bar{C} \bar{A} \bar{B},
            \dots, \bar{C} \bar{A}^{L-1} \bar{B})
        """

        A = -jnp.exp(self.A_re) + 1j * self.A_im
        P, B, C = self.P, self.B, self.C
        dt = jnp.exp(self.log_dt)

        # \tilde{C}
        Ab, _, _ = self.discrete()
        Ct = C - jnp.linalg.matrix_power(Ab, L).T @ C

        # Roots of unity
        z = jnp.exp(-2j * math.pi / L * jnp.arange(L))

        # Cauchy
        u = 2 / dt * (1 - z) / (1 + z)
        v = jnp.stack((
            Ct * B,
            Ct * P,
            P.conj() * B,
            P.conj() * P,
        ))

        k00, k01, k10, k11 = jnp.sum(v[:, None] / (u[:, None] - A), axis=-1)

        # Kernel
        k = 2 / (1 + z) * (k00 - k01 / (1 + k11) * k10)
        k = jnp.fft.ifft(k).real

        return k

    def cell(self) -> Cell:
        r"""Returns an equivalent recurrent cell."""

        return S4Cell(*self.discrete())


class S4Cell(Cell):
    r"""TODO

    .. math::
        h_i & = A h_{i-1} + B x_i \\
        y_i & = C h_i

    Arguments:
        A: with shape :math:`(H, H)`.
        B: with shape :math:`(H)`.
        C: with shape :math:`(H)`.
    """

    def __init__(self, A: Array, B: Array, C: Array):
        self.A = A
        self.B = B
        self.C = C

    def __call__(self, h: Array, x: Array) -> Tuple[Array, Array]:
        r"""
        Arguments:
            h: The previous hidden state :math:`h_{i-1}`, with shape :math:`(H)`.
            x: The input :math:`x_i`, with shape :math:`()`.

        Returns:
            The hidden state and output :math:`(h_i, y_i)`, with shape
            :math:`(H)` and :math:`()`.
        """

        h = self.A @ h + self.B * x
        y = jnp.dot(self.C, h)

        return h, y.real

    def init(self) -> Array:
        return jnp.zeros_like(self.B)
