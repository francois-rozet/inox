r"""Extended utilities for tree-like data structures"""

__all__ = [
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

    __slots__ = ('value')

    def __init__(self, value: Hashable):
        if not isinstance(value, Hashable):
            warn(f"'{type(value).__name__}' object is not hashable.")

        self.value = value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.value)})'

    def tree_flatten(self):
        return (), self.value

    @classmethod
    def tree_unflatten(cls, value, _):
        return cls(value)


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
