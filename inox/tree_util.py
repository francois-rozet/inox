r"""Extended utilities for tree-like data structures"""

__all__ = [
    'Namespace',
    'Static',
    'tree_mask',
    'tree_unmask',
    'tree_partition',
    'tree_combine',
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
        >>> tree = Namespace(a=1, b='2'); tree
        Namespace(
          a = 1,
          b = '2'
        )
        >>> tree.c = [3, False]; tree
        Namespace(
          a = 1,
          b = '2',
          c = [3, False]
        )
        >>> jax.tree_util.tree_leaves(tree)
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
    r"""Wraps an hashable value as a leafless tree.

    Arguments:
        value: An hashable value to wrap.

    Example:
        >>> tree = Static((0, 'one', None))
        >>> tree.value
        (0, 'one', None)
        >>> jax.tree_util.tree_leaves(tree)
        []
        >>> jax.tree_util.tree_structure(tree)
        PyTreeDef(CustomNode(Static[(0, 'one', None)], []))
    """

    def __init__(self, value: Hashable):
        if not isinstance(value, Hashable):
            warn(f"'{type(value).__name__}' object is not hashable.")

        self.value = value

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other) and self.value == other.value

    def __hash__(self) -> int:
        return hash((type(self), self.value))

    def __repr__(self) -> str:
        return self.tree_repr()

    def tree_repr(self, **kwargs) -> str:
        return f'{self.__class__.__name__}({tree_repr(self.value, **kwargs)})'

    def tree_flatten(self):
        return (), self.value

    def tree_flatten_with_keys(self):
        return self.tree_flatten()

    @classmethod
    def tree_unflatten(cls, value, _):
        self = object.__new__(cls)
        self.value = value

        return self


def tree_mask(
    tree: PyTree,
    is_static: Callable[[Any], bool] = None,
) -> PyTree:
    r"""Masks the static leaves of a tree.

    See also:
        :func:`tree_unmask`

    Arguments:
        tree: The tree to mask.
        is_static: A predicate for what to consider static. If :py:`None`,
            all non-array leaves are considered static.

    Returns:
        The masked tree.

    Example:
        >>> tree = [1, jax.numpy.arange(2), 'three']
        >>> jax.tree_util.tree_leaves(tree)
        [1, Array([0, 1], dtype=int32), 'three']
        >>> tree = tree_mask(tree); tree
        [Static(1), Array([0, 1], dtype=int32), Static('three')]
        >>> jax.tree_util.tree_leaves(tree)
        [Array([0, 1], dtype=int32)]
    """

    if is_static is None:
        is_static = lambda x: not is_array(x)

    return jtu.tree_map(
        f=lambda x: Static(x) if is_static(x) else x,
        tree=tree,
    )


def tree_unmask(tree: PyTree) -> PyTree:
    r"""Unmasks the static leaves of a masked tree.

    See also:
        :func:`tree_mask`

    Arguments:
        tree: The masked tree to unmask.

    Returns:
        The unmasked tree.

    Example:
        >>> tree = [Static(1), jax.numpy.arange(2), Static('three')]
        >>> tree_unmask(tree)
        [1, Array([0, 1], dtype=int32), 'three']
    """

    is_static = lambda x: isinstance(x, Static)

    return jtu.tree_map(
        f=lambda x: x.value if is_static(x) else x,
        tree=tree,
        is_leaf=is_static,
    )


def tree_partition(
    tree: PyTree,
    *filters: Union[type, Callable[[Any], bool]],
    is_leaf: Callable[[Any], bool] = None,
) -> Tuple[PyTreeDef, Dict[str, Any]]:
    r"""Flattens a tree and partitions the leaves.

    The leaves are partitioned into a set of path-leaf mappings. Each mapping contains
    the leaves of the subset of nodes satisfying the corresponding filtering constraint.
    The last mapping is dedicated to leaves that do not satisfy any constraint.

    See also:
        :func:`tree_combine`

    Arguments:
        tree: The tree to flatten.
        filters: A set of filtering constraints. Types are transformed into
            :py:`isinstance` constraints.
        is_leaf: A predicate for what to consider as a leaf.

    Returns:
        The tree definition and leaf partitions.

    Example:
        >>> tree = Namespace(a=1, b=jax.numpy.arange(2), c=['three', False])
        >>> treedef, leaves = tree_partition(tree)
        >>> leaves
        {'.a': 1, '.b': Array([0, 1], dtype=int32), '.c[0]': 'three', '.c[1]': False}
        >>> treedef, arrays, others = tree_partition(tree, jax.Array)
        >>> arrays
        {'.b': Array([0, 1], dtype=int32)}
        >>> others
        {'.a': 1, '.c[0]': 'three', '.c[1]': False}
    """

    treedef = jtu.tree_structure(tree, is_leaf)
    leaves = [{} for _ in filters] + [{}]

    filters = [
        (lambda x: isinstance(x, filtr))
        if isinstance(filtr, type) else filtr
        for filtr in filters
    ]

    if is_leaf is None:
        is_node = lambda x: any(filtr(x) for filtr in filters)
    else:
        is_node = lambda x: any(filtr(x) for filtr in filters) or is_leaf(x)

    for path, node in jtu.tree_leaves_with_path(tree, is_node):
        for i, filtr in enumerate(filters):
            if filtr(node):
                break
        else:
            i = -1

        for subpath, leaf in jtu.tree_leaves_with_path(node, is_leaf):
            leaves[i][jtu.keystr(path + subpath)] = leaf

    return treedef, *leaves


def tree_combine(
    treedef: PyTreeDef,
    *leaves: Dict[str, Any],
) -> PyTree:
    r"""Reconstructs a tree from the tree definition and leaf partitions.

    See also:
        :func:`tree_partition`

    Arguments:
        treedef: The tree definition.
        leaves: The set of leaf partitions.

    Returns:
        The reconstructed tree.

    Example:
        >>> tree = Namespace(a=1, b=jax.numpy.arange(2), c=['three', False])
        >>> treedef, arrays, others = tree_partition(tree, jax.Array)
        >>> others = {key: str(leaf).upper() for key, leaf in others.items()}
        >>> tree_combine(treedef, arrays, others)
        Namespace(
          a = '1',
          b = int32[2],
          c = ['THREE', 'FALSE']
        )
    """

    missing = []
    leaves = {
        key: leaf
        for partition in leaves
        for key, leaf in partition.items()
    }

    def f(path, leaf):
        key = jtu.keystr(path)

        if key in leaves:
            leaf = leaves.pop(key)
        else:
            missing.append(key)

        return leaf

    tree = jtu.tree_unflatten(treedef, [object()] * treedef.num_leaves)
    tree = jtu.tree_map_with_path(f, tree)

    if missing:
        keys = ', '.join(f'"{key}"' for key in missing)

        raise KeyError(f"Missing key(s) in leaves: {keys}.")

    if leaves:
        keys = ', '.join(f'"{key}"' for key in leaves)

        raise KeyError(f"Unexpected key(s) in leaves: {keys}.")

    return tree


def tree_repr(
    tree: PyTree,
    linewidth: int = 88,
    typeonly: bool = True,
    **kwargs,
) -> str:
    r"""Creates a pretty representation of a tree.

    Arguments:
        tree: The tree to represent.
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

    if hasattr(tree, 'tree_repr'):
        return tree.tree_repr(**kwargs)
    elif isinstance(tree, tuple):
        bra, ket = '(', ')'
        lines = [tree_repr(x, **kwargs) for x in tree]
    elif isinstance(tree, list):
        bra, ket = '[', ']'
        lines = [tree_repr(x, **kwargs) for x in tree]
    elif isinstance(tree, dict):
        bra, ket = '{', '}'
        lines = [
            f'{tree_repr(key)}: {tree_repr(value)}'
            for key, value in tree.items()
        ]
    elif is_array(tree) and typeonly:
        return f'{tree.dtype}{list(tree.shape)}'
    else:
        return repr(tree).strip(' \n')

    if any('\n' in line for line in lines):
        lines = ',\n'.join(lines)
    elif sum(len(line) + 2 for line in lines) > linewidth:
        lines = ',\n'.join(lines)
    else:
        lines = ', '.join(lines)

    if '\n' in lines:
        lines = '\n' + indent(lines, '  ') + '\n'

    return f'{bra}{lines}{ket}'
