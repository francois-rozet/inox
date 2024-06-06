r"""State space model (SSM) layers"""

__all__ = [
    'SISO',
    'S4',
]

import jax
import jax.numpy as jnp
import math
import numpy as np

from jax import Array
from typing import Tuple

# isort: split
from .module import ComplexParameter, Module, Parameter
from ..random import get_rng


class SISO(Module):
    r"""Abstract single-input single-output (SISO) state space model class.

    A SISO state space model defines a system of equations of the form

    .. math::
        \dot{x}(t) & = A x(t) + B u(t) \\
        y(t) & = C x(t)

    where :math:`u(t), y(t) \in \mathbb{C}` are input and output signals and :math:`x(t)
    \in \mathbb{C}^{H}` is a latent/hidden state. In practice, the input and output
    signals are sampled every :math:`\Delta` time units, leading to sequences
    :math:`(x_1, x_2, \dots)` and :math:`(y_1, y_2, \dots)` whose dynamics are governed
    by the discrete-time form of the system

    .. math::
        x_i & = \bar{A} x_{i-1} + \bar{B} u_i \\
        y_i & = \bar{C} x_i

    where :math:`\bar{A} = \exp(\Delta A)`, :math:`\bar{B} = A^{-1} (\bar{A} - I) B` and
    :math:`\bar{C} = C`. Assuming :math:`x_0 = 0`, the dynamics can also be represented
    as a discrete-time convolution

    .. math:: y_{1:L} = \bar{k}_{1:L} * u_{1:L}

    where :math:`\bar{k}_i = \bar{C} \bar{A}^{i-1} \bar{B} \in \mathbb{C}`.

    Wikipedia:
        https://wikipedia.org/wiki/State-space_representation
    """

    def __call__(self, u: Array) -> Array:
        r"""
        Arguments:
            u: The input signal :math:`u_{1:L}`, with shape :math:`(*, L)`.
                Floating point arrays are promoted to complex arrays.

        Returns:
            The output signal :math:`y_{1:L}`, with shape :math:`(*, L)`.
        """

        L = u.shape[-1]
        k = self.kernel(L)

        return jax.scipy.signal.fftconvolve(k, u, axes=-1)[..., :L]

    def discrete(self) -> Tuple[Array, Array, Array]:
        r"""
        Returns:
            The matrices :math:`\bar{A}`, :math:`\bar{B}` and :math:`\bar{C}`,
            respectively with shape :math:`(H, H)`, :math:`(H)` and :math:`(H)`.
        """

        raise NotImplementedError()

    def kernel(self, length: int) -> Array:
        r"""
        Arguments:
            length: The kernel length :math:`L`.

        Returns:
            The kernel :math:`\bar{k}_{1:L}`, with shape :math:`(L)`.
        """

        raise NotImplementedError()


class Eigenvalue(ComplexParameter):
    r"""Wrapper to indicate an optimizable eigenvalue array."""

    def __init__(self, real: Array, imag: Array):
        self.log_real = jnp.log(-real)
        self.imag = imag

    @property
    def real(self) -> Array:
        return -jnp.exp(self.log_real)


class S4(SISO):
    r"""Creates an S4 state space model.

    References:
        | Efficiently Modeling Long Sequences with Structured State Spaces (Gu et al., 2021)
        | https://arxiv.org/abs/2111.00396

        | The Annotated S4 (Rush et al., 2023)
        | https://srush.github.io/annotated-s4

    Arguments:
        hid_features: The number of hidden features :math:`H`.
        key: A PRNG key for initialization. If :py:`None`,
            :func:`inox.random.get_rng` is used instead.

    Example:
        >>> ssm = S4(hid_features=64, key=key)
        >>> u = jax.numpy.linspace(0.0, 1.0, 1024)
        >>> y = ssm(u)
    """

    def __init__(self, hid_features: int, key: Array = None):
        if key is None:
            keys = get_rng().split(3)
        else:
            keys = jax.random.split(key, 3)

        A, P = S4.DPLR_HiPPO(hid_features)

        self.A = Eigenvalue(A)
        self.P = ComplexParameter(P)
        self.B = ComplexParameter(
            jax.random.normal(
                keys[0],
                shape=(hid_features,),
                dtype=complex,
            )
        )
        self.C = ComplexParameter(
            jax.random.normal(
                keys[1],
                shape=(hid_features,),
                dtype=complex,
            )
            / math.sqrt(hid_features)
        )
        self.log_dt = Parameter(
            jax.random.uniform(
                keys[2],
                shape=(),
                minval=math.log(1e-3),
                maxval=math.log(1e-1),
            )
        )

    @staticmethod
    def DPLR_HiPPO(n: int) -> Tuple[Array, Array]:
        r"""Returns the diagonal plus low-rank (DPLR) form of the HiPPO matrix.

        .. math:: A = \Lambda - PP^*

        Arguments:
            n: The size :math:`n` of the HiPPO matrix.
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

        return A, P

    def discrete(self) -> Tuple[Array, Array, Array]:
        A, P, B, C = self.A(), self.P(), self.B(), self.C()
        dt = jnp.exp(self.log_dt())

        D = jnp.diag(1 / (2 / dt - A))
        PQ = jnp.outer(P, P.conj())
        A0 = jnp.diag(2 / dt + A) - PQ
        A1 = D - D @ PQ / (1 + jnp.dot(P.conj(), D @ P)) @ D

        Ab = A1 @ A0
        Bb = 2 * A1 @ B

        return Ab, Bb, C

    def kernel(self, length: int) -> Array:
        A, P, B, C = self.A(), self.P(), self.B(), self.C()
        dt = jnp.exp(self.log_dt())

        # \tilde{C}
        Ab, _, _ = self.discrete()
        Ct = C - jnp.linalg.matrix_power(Ab, length).T @ C

        # Roots of unity
        z = jnp.exp(-2j * math.pi / length * jnp.arange(length))

        # Cauchy
        w = 2 / dt * (1 - z) / (1 + z)
        v = jnp.stack((
            Ct * B,
            Ct * P,
            P.conj() * B,
            P.conj() * P,
        ))

        k00, k01, k10, k11 = jnp.sum(v[:, None] / (w[:, None] - A), axis=-1)

        # Kernel
        k = 2 / (1 + z) * (k00 - k01 / (1 + k11) * k10)
        k = jnp.fft.ifft(k)

        return k
