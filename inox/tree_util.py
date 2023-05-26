r"""Extended utilities for tree-like data structures"""

__all__ = [
    'tree_partition',
    'tree_merge',
    'tree_repr',
]

import jax

from textwrap import indent
from typing import *


PyTree = TypeVar('PyTree')
PyTreeDef = TypeVar('PyTreeDef')


def tree_partition(
    f: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Callable[[Any], bool] = None,
) -> Tuple[List[Any], List[Any], PyTreeDef]:
    r""""""

    leaves, treedef = jax.tree_util.tree_flatten(tree, is_leaf)
    left = [x if f(x) else None for x in leaves]
    right = [None if f(x) else x for x in leaves]

    return left, right, treedef


def tree_merge(
    left: List[Any],
    right: List[Any],
    treedef: PyTreeDef,
) -> PyTree:
    r""""""

    leaves = [x if y is None else y for x, y in zip(left, right)]
    tree = jax.tree_util.tree_unflatten(treedef, leaves)

    return tree


def tree_repr(x: object) -> str:
    r""""""

    if isinstance(x, tuple):
        bra, ket = '(', ')'
        lines = list(map(tree_repr, x))
    elif isinstance(x, list):
        bra, ket = '[', ']'
        lines = list(map(tree_repr, x))
    elif isinstance(x, dict):
        bra, ket = '{', '}'
        lines = [
            f'{tree_repr(key)}: {tree_repr(value)}'
            for key, value in x.items()
        ]
    else:
        return repr(x).strip(' \n')

    if any('\n' in line for line in lines):
        lines = ',\n'.join(lines)
    else:
        lines = ', '.join(lines)

    if '\n' in lines:
        lines = '\n' + indent(lines, '  ') + '\n'

    return f'{bra}{lines}{ket}'
