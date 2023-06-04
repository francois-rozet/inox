r"""Extended utilities for tree-like data structures"""

__all__ = [
    'Namespace',
    'Static',
    'tree_partition',
    'tree_merge',
    'tree_repr',
]

import jax
import jax._src.tree_util as jtu

from textwrap import indent
from typing import *


PyTree = TypeVar('PyTree', bound=Any)
PyTreeDef = TypeVar('PyTreeDef')


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
        >>> ns := Namespace(a=1, b='2'); ns
        Namespace(
          a = 1,
          b = '2'
        )
        >>> ns.c = [3]; ns
        Namespace(
          a = 1,
          b = '2',
          c = [3]
        )
        >>> jax.tree_util.tree_leaves(ns)
        [1, '2', 3]
    """

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __repr__(self) -> str:
        return tree_repr(self)

    def tree_repr(self, **kwargs) -> str:
        lines = (
            f'{name} = {tree_repr(value, **kwargs)}'
            for name, value in sorted(self.__dict__.items())
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
    r"""Wraps an hashable object as a leafless PyTree.

    Arguments:
        x: An hashable object to wrap.

    Example:
        >>> x = Static((0, 'one', None))
        >>> jax.tree_util.tree_leaves(x)
        []
        >>> jax.tree_util.tree_structure(x)
        PyTreeDef(CustomNode(Static[(0, 'one', None)], []))
    """

    def __init__(self, x: Hashable):
        self.x = x

    def __call__(self) -> Hashable:
        r"""
        Returns:
            The wrapped object.
        """

        return self.x

    def __repr__(self) -> str:
        return repr(self.x)

    def tree_flatten(self):
        return (), self.x

    @classmethod
    def tree_unflatten(cls, x, _):
        return cls(x)


def tree_partition(
    f: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Callable[[Any], bool] = None,
) -> Tuple[List[Any], List[Any], PyTreeDef]:
    r"""Flattens a tree and splits the leaves into two partitions.

    See also:
        :func:`tree_merge`

    Arguments:
        f: A predicate choosing the partition of each leaf.
        tree: The tree to flatten.
        is_leaf: A predicate for what to consider as a leaf.

    Returns:
        The two partitions and the structure definition of the tree.

    Example:
        >>> f = lambda x: isinstance(x, str)
        >>> tree = [1, 'two', (True, 'False')]
        >>> left, right, treedef = tree_partition(f, tree)
        >>> left
        [None, 'two', None, 'False']
        >>> right
        [1, None, True, None]
    """

    leaves, treedef = jtu.tree_flatten(tree, is_leaf)
    left = [x if f(x) else None for x in leaves]
    right = [None if f(x) else x for x in leaves]

    return left, right, treedef


def tree_merge(
    treedef: PyTreeDef,
    left: List[Any],
    right: List[Any],
) -> PyTree:
    r"""Merges two partitions as a single tree.

    See also:
        :func:`tree_partition`

    Arguments:
        treedef: The PyTree structure definition.
        left: The left partition.
        right: The right partition.

    Returns:
        The resulting tree.

    Example:
        >>> left = [None if x is None else x.upper() for x in left]
        >>> tree_merge(treedef, left, right)
        [1, 'TWO', (True, 'FALSE')]
    """

    leaves = [x if y is None else y for x, y in zip(left, right)]
    tree = jtu.tree_unflatten(treedef, leaves)

    return tree


def tree_repr(
    x: Any,
    /,
    linewidth: int = 88,
    shapeonly: bool = True,
    threshold: int = 6,
    **kwargs,
) -> str:
    r"""Creates a pretty tree representation of an object.

    Arguments:
        x: An object to represent.
        linewidth: The maximum line width before elements of tuples, lists and dicts
            are represented on separate lines.
        shapeonly: Whether to represent the shape of arrays instead of their elements.
        threshold: The maximum number of elements before tuples, lists and dicts are
            summarized.

    Returns:
        The representation string.

    Example:
        >>> tree = [1, 'two', (True, False), list(range(56)), {7: jax.numpy.arange(8), None: '10'}]
        >>> print(tree_repr(tree))
        [
          1,
          'two',
          (True, False),
          [0, 1, 2, ..., 53, 54, 55],
          {7: Array(shape=(8,), dtype=int32), None: '10'}
        ]
    """

    kwargs.update(
        linewidth=linewidth,
        shapeonly=shapeonly,
        threshold=threshold,
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
    elif isinstance(x, jax.Array) and shapeonly:
        return f'Array(shape={x.shape}, dtype={x.dtype})'
    else:
        return repr(x).strip(' \n')

    if len(lines) > threshold:
        lines = lines[:threshold // 2] + ['...'] + lines[-threshold // 2:]

    if any('\n' in line for line in lines):
        lines = ',\n'.join(lines)
    elif sum(len(line) + 2 for line in lines) > linewidth:
        lines = ',\n'.join(lines)
    else:
        lines = ', '.join(lines)

    if '\n' in lines:
        lines = '\n' + indent(lines, '  ') + '\n'

    return f'{bra}{lines}{ket}'
