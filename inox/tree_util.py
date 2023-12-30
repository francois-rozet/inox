r"""Extended utilities for tree-like data structures"""

__all__ = [
    'PyArray',
    'Namespace',
    'Static',
    'Auto',
    'tree_repr',
]

import jax
import jax._src.tree_util as jtu
import numpy as np

from jax import Array
from textwrap import indent
from typing import *
from warnings import warn


PyTree = TypeVar('PyTree', bound=Any)
PyTreeDef = TypeVar('PyTreeDef')


def is_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) or isinstance(x, Array)


def is_static(x: Any) -> bool:
    return isinstance(x, Static)


def is_auto(x: Any) -> bool:
    return isinstance(x, Auto)


class PyTreeMeta(type):
    r"""PyTree meta-class."""

    def __new__(cls, *args, **kwargs) -> type:
        cls = super().__new__(cls, *args, **kwargs)

        if hasattr(cls, 'tree_flatten_with_keys'):
            jtu.register_pytree_with_keys_class(cls)
        else:
            jtu.register_pytree_node_class(cls)

        return cls


class PyArray(metaclass=PyTreeMeta):
    r"""Wraps an array as a PyTree.

    Subclassing :class:`PyArray` allows to associate metadata (name, role, ...) to
    arrays, while preserving the array interface (`.shape`, `.dtype`, ...) and
    supporting :mod:`jax.numpy` operations.

    Arguments:
        value: An array value.

    Example:
        >>> x = jax.numpy.arange(5)
        >>> x = PyArray(x); x
        int32[5]
        >>> jax.numpy.mean(x)
        Array(2., dtype=float32)
    """

    def __init__(self, value: Array):
        self.value = value

    def __array__(self, dtype=None):
        return self.value.__array__(dtype)

    def __array_module__(self, types):
        return self.value.__array_module__(types)

    def __jax_array__(self) -> Array:
        if hasattr(self.value, '__jax_array__'):
            return self.value.__jax_array__()
        else:
            return self.value

    def __getattr__(self, attr: str) -> Any: return getattr(self.value, attr)
    def __getitem__(self, key: Hashable): return self.value[key]
    def __len__(self) -> int: return len(self.value)
    def __iter__(self) -> Iterable: return iter(self.value)
    def __reversed__(self) -> Iterable: return reversed(self.value)

    def __neg__(self): return self.value.__neg__()
    def __pos__(self): return self.value.__pos__()
    def __abs__(self): return self.value.__abs__()
    def __invert__(self): return self.value.__invert__()
    def __round__(self, ndigits=None): return self.value.__round__(ndigits)
    def __eq__(self, other): return self.value.__eq__(other)
    def __ne__(self, other): return self.value.__ne__(other)
    def __lt__(self, other): return self.value.__lt__(other)
    def __le__(self, other): return self.value.__le__(other)
    def __gt__(self, other): return self.value.__gt__(other)
    def __ge__(self, other): return self.value.__ge__(other)
    def __add__(self, other): return self.value.__add__(other)
    def __radd__(self, other): return self.value.__radd__(other)
    def __sub__(self, other): return self.value.__sub__(other)
    def __rsub__(self, other): return self.value.__rsub__(other)
    def __mul__(self, other): return self.value.__mul__(other)
    def __rmul__(self, other): return self.value.__rmul__(other)
    def __div__(self, other): return self.value.__div__(other)
    def __rdiv__(self, other): return self.value.__rdiv__(other)
    def __truediv__(self, other): return self.value.__truediv__(other)
    def __rtruediv__(self, other): return self.value.__rtruediv__(other)
    def __floordiv__(self, other): return self.value.__floordiv__(other)
    def __rfloordiv__(self, other): return self.value.__rfloordiv__(other)
    def __divmod__(self, other): return self.value.__divmod__(other)
    def __rdivmod__(self, other): return self.value.__rdivmod__(other)
    def __mod__(self, other): return self.value.__mod__(other)
    def __rmod__(self, other): return self.value.__rmod__(other)
    def __pow__(self, other): return self.value.__pow__(other)
    def __rpow__(self, other): return self.value.__rpow__(other)
    def __matmul__(self, other): return self.value.__matmul__(other)
    def __rmatmul__(self, other): return self.value.__rmatmul__(other)
    def __and__(self, other): return self.value.__and__(other)
    def __rand__(self, other): return self.value.__rand__(other)
    def __or__(self, other): return self.value.__or__(other)
    def __ror__(self, other): return self.value.__ror__(other)
    def __xor__(self, other): return self.value.__xor__(other)
    def __rxor__(self, other): return self.value.__rxor__(other)
    def __lshift__(self, other): return self.value.__lshift__(other)
    def __rlshift__(self, other): return self.value.__rlshift__(other)
    def __rshift__(self, other): return self.value.__rshift__(other)
    def __rrshift__(self, other): return self.value.__rrshift__(other)

    def __repr__(self) -> str:
        return self.tree_repr()

    def tree_repr(self, **kwargs) -> str:
        return tree_repr(self.value, **kwargs)

    def tree_flatten(self):
        return [self.value], None

    def tree_flatten_with_keys(self):
        return [('', self.value)], None

    @classmethod
    def tree_unflatten(cls, _, leaves):
        self = object.__new__(cls)
        self.value = leaves[0]

        return self


class Namespace(metaclass=PyTreeMeta):
    r"""PyTree class for name-value mappings.

    Arguments:
        kwargs: A name-value mapping.

    Example:
        >>> ns = Namespace(a=1, b='2'); ns
        Namespace(
          a = 1,
          b = '2'
        )
        >>> ns.c = [3, False]; ns
        Namespace(
          a = 1,
          b = '2',
          c = [3, False]
        )
        >>> jax.tree_util.tree_leaves(ns)
        [1, '2', 3, False]
    """

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def __repr__(self) -> str:
        return tree_repr(self)

    def tree_repr(self, **kwargs) -> str:
        lines = (
            f'{name} = {tree_repr(getattr(self, name), **kwargs)}'
            for name in sorted(self.__dict__.keys())
        )

        lines = ',\n'.join(lines)

        if lines:
            lines = '\n' + indent(lines, '  ') + '\n'

        return f'{self.__class__.__name__}({lines})'

    def tree_flatten(self):
        if self.__dict__:
            names, values = zip(*sorted(self.__dict__.items()))
        else:
            names, values = (), ()

        return values, names

    def tree_flatten_with_keys(self):
        values, names = self.tree_flatten()
        keys = map(jtu.GetAttrKey, names)

        return list(zip(keys, values)), names

    @classmethod
    def tree_unflatten(cls, names, values):
        self = object.__new__(cls)
        self.__dict__ = dict(zip(names, values))

        return self


class Static(metaclass=PyTreeMeta):
    r"""Wraps an hashable value as a leafless PyTree.

    Arguments:
        value: An hashable value to wrap.

    Example:
        >>> x = Static((0, 'one', None))
        >>> x.value
        (0, 'one', None)
        >>> jax.tree_util.tree_leaves(x)
        []
        >>> jax.tree_util.tree_structure(x)
        PyTreeDef(CustomNode(Static[(0, 'one', None)], []))
    """

    def __init__(self, value: Hashable):
        if not isinstance(value, Hashable):
            warn(f"'{type(value).__name__}' object is not hashable.")

        self.value = value

    def __hash__(self) -> int:
        return hash((type(self), self.value))

    def __repr__(self) -> str:
        return self.tree_repr()

    def tree_repr(self, **kwargs) -> str:
        return f'{self.__class__.__name__}({tree_repr(self.value, **kwargs)})'

    def tree_flatten(self):
        return (), self.value

    @classmethod
    def tree_unflatten(cls, value, _):
        self = object.__new__(cls)
        self.value = value

        return self


class Auto(Namespace):
    r"""Subclass of :class:`Namespace` that automatically detects non-array leaves
    and considers them as static.

    Important:
        :py:`object()` leaves are never considered static.

    Arguments:
        kwargs: A name-value mapping.

    Example:
        >>> auto = Auto(a=1, b=jnp.array(2.0)); auto
        Auto(
          a = 1,
          b = float32[]
        )
        >>> auto.c = ['3', jnp.arange(4)]; auto
        Auto(
          a = 1,
          b = float32[],
          c = ['3', int32[4]]
        )
        >>> jax.tree_util.tree_leaves(auto)  # only arrays
        [Array(2., dtype=float32, weak_type=True), Array([0, 1, 2, 3], dtype=int32)]
    """

    def tree_flatten(self):
        values, names = super().tree_flatten()

        values = jtu.tree_map(
            f=lambda x: x if type(x) is object or is_array(x) or is_auto(x) else Static(x),
            tree=values,
            is_leaf=is_auto,
        )

        return values, names

    @classmethod
    def tree_unflatten(cls, names, values):
        values = jtu.tree_map(
            f=lambda x: x.value if is_static(x) else x,
            tree=values,
            is_leaf=lambda x: is_auto(x) or is_static(x),
        )

        return super().tree_unflatten(names, values)


def tree_repr(
    x: PyTree,
    /,
    linewidth: int = 88,
    typeonly: bool = True,
    **kwargs,
) -> str:
    r"""Creates a pretty representation of a tree.

    Arguments:
        x: The tree to represent.
        linewidth: The maximum line width before elements of tuples, lists and dicts
            are represented on separate lines.
        typeonly: Whether to represent the type of arrays instead of their elements.

    Returns:
        The representation string.

    Example:
        >>> tree = [1, 'two', (True, False), list(range(5)), {'6': jnp.arange(7)}]
        >>> print(tree_repr(tree))
        [
          1,
          'two',
          (True, False),
          [0, 1, 2, 3, 4, 5],
          {'6': int32[7]}
        ]
    """

    kwargs.update(
        linewidth=linewidth,
        typeonly=typeonly,
    )

    if hasattr(x, 'tree_repr'):
        return x.tree_repr(**kwargs)
    elif isinstance(x, tuple):
        bra, ket = '(', ')'
        lines = [tree_repr(y, **kwargs) for y in x]
    elif isinstance(x, list):
        bra, ket = '[', ']'
        lines = [tree_repr(y, **kwargs) for y in x]
    elif isinstance(x, dict):
        bra, ket = '{', '}'
        lines = [
            f'{tree_repr(key)}: {tree_repr(value)}'
            for key, value in x.items()
        ]
    elif is_array(x):
        if typeonly:
            return f'{x.dtype}{list(x.shape)}'
        else:
            return repr(x)
    else:
        return repr(x).strip(' \n')

    if any('\n' in line for line in lines):
        lines = ',\n'.join(lines)
    elif sum(len(line) + 2 for line in lines) > linewidth:
        lines = ',\n'.join(lines)
    else:
        lines = ', '.join(lines)

    if '\n' in lines:
        lines = '\n' + indent(lines, '  ') + '\n'

    return f'{bra}{lines}{ket}'
