r"""Upstream patches"""

__all__ = []

from jax._src import core
from jax._src.lax import lax


# core.Primitive.bind comply with 'valid_jaxtype'
bind_old = core.Primitive.bind

def bind_new(self, *args, **kwargs):
    args = [
        x.__jax_array__() if hasattr(x, '__jax_array__') else x
        for x in args
    ]

    return bind_old(self, *args, **kwargs)

core.Primitive.bind = bind_new


# lax.asarray comply with 'valid_jaxtype'
asarray_old = lax.asarray

def asarray_new(x):
    if hasattr(x, '__jax_array__'):
        x = x.__jax_array__()

    return asarray_old(x)

lax.asarray = asarray_new
