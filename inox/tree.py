r"""Extended utilities for tree-like data structures."""

__all__ = [
    "Namespace",
    "Partial",
    "Static",
    "mask_static",
    "unmask_static",
    "partition",
    "combine",
    "prepr",
]

import dataclasses
import jax._src.tree_util as jtu
import numpy as np

from jax import Array
from textwrap import indent
from typing import Any, Callable, Dict, Hashable, Tuple, TypeVar, Union
from warnings import warn

PyTree = TypeVar("PyTree", bound=Any)
PyTreeDef = TypeVar("PyTreeDef")


def is_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) or isinstance(x, Array)


def is_atom(x: Any) -> bool:
    return jtu.pytree.all_leaves(jtu.none_leaf_registry, (x,))


class PyTreeMeta(type):
    r"""PyTree meta-class."""

    def __new__(cls, *args, **kwargs) -> type:
        cls = super().__new__(cls, *args, **kwargs)

        if hasattr(cls, "tree_flatten_with_keys"):
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
        >>> jax.tree.leaves(tree)
        [1, '2', 3, False]
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self) -> str:
        return prepr(self)

    def tree_repr(self, **kwargs) -> str:
        private = kwargs.get("private", False)

        lines = (
            f"{name} = {prepr(getattr(self, name), **kwargs)}"
            for name in sorted(self.__dict__.keys())
            if not name.startswith("_") or private
        )

        lines = ",\n".join(lines)

        if lines:
            lines = "\n" + indent(lines, "  ") + "\n"

        return f"{self.__class__.__name__}({lines})"

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


class Partial(Namespace):
    r"""A version of :class:`functools.partial` that is a PyTree.

    Arguments:
        func: A function.
        args: Positional arguments for future calls.
        kwds: Keyword arguments for future calls.

    Examples:
        >>> increment = Partial(jax.numpy.add, 1)
        >>> increment(2)
        Array(3, dtype=int32, weak_type=True)

        >>> println = Partial(print, sep='\n')
        >>> println('Hello', 'World!')
        Hello
        World!
    """

    def __init__(self, func: Callable, *args: Any, **kwds: Any):
        self.func = func
        self.args = args
        self.kwds = kwds

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*self.args, *args, **self.kwds, **kwds)


class Static(metaclass=PyTreeMeta):
    r"""Wraps an hashable value as a leafless PyTree.

    Arguments:
        value: An hashable value to wrap.

    Example:
        >>> tree = Static((0, 'one', None))
        >>> tree.value
        (0, 'one', None)
        >>> jax.tree.leaves(tree)
        []
        >>> jax.tree.structure(tree)
        PyTreeDef(CustomNode(Static[(0, 'one', None)], []))
    """

    def __init__(self, value: Hashable):
        if not isinstance(value, Hashable):
            warn(f"considering a non-hashable object ('{type(value).__name__}') static could lead to frequent JIT recompilations.", stacklevel=2)  # fmt: off
        elif callable(value):
            if "<lambda>" in value.__qualname__ or "<locals>" in value.__qualname__:
                warn(f"considering a local function ('{value.__qualname__}') static could lead to frequent JIT recompilations.", stacklevel=2)  # fmt: off

        self.value = value

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other) and self.value == other.value

    def __hash__(self) -> int:
        return hash((type(self), self.value))

    def __repr__(self) -> str:
        return self.tree_repr()

    def tree_repr(self, **kwargs) -> str:
        return f"{self.__class__.__name__}({prepr(self.value, **kwargs)})"

    def tree_flatten(self):
        return (), self.value

    def tree_flatten_with_keys(self):
        return self.tree_flatten()

    @classmethod
    def tree_unflatten(cls, value, _):
        self = object.__new__(cls)
        self.value = value

        return self


class Mask(Static):
    pass


def mask_static(
    tree: PyTree,
    is_static: Callable[[Any], bool] = None,
) -> PyTree:
    r"""Masks the static leaves of a tree.

    The structure of the tree remains unchanged, but leaves that are considered static
    are masked, which hides them from :func:`jax.tree.leaves` and :func:`jax.tree.map`.
    Applying :func:`inox.tree.mask_static` more than once leads to the same tree.

    See also:
        :func:`inox.tree.unmask_static`

    Arguments:
        tree: The tree to mask.
        is_static: A predicate for what to consider static. If :py:`None`,
            all non-array leaves are considered static.

    Returns:
        The masked tree.

    Example:
        >>> tree = [1, jax.numpy.arange(2), 'three']
        >>> jax.tree.leaves(tree)
        [1, Array([0, 1], dtype=int32), 'three']
        >>> tree = inox.tree.mask_static(tree); tree
        [Mask(1), Array([0, 1], dtype=int32), Mask('three')]
        >>> jax.tree.leaves(tree)
        [Array([0, 1], dtype=int32)]
        >>> inox.tree.unmask_static(tree)
        [1, Array([0, 1], dtype=int32), 'three']
    """

    if is_static is None:
        is_static = lambda x: not is_array(x)

    return jtu.tree_map(
        f=lambda x: Mask(x) if is_static(x) else x,
        tree=tree,
    )


def unmask_static(tree: PyTree) -> PyTree:
    r"""Unmasks the static leaves of a masked tree.

    See also:
        :func:`inox.tree.mask_static`

    Arguments:
        tree: The masked tree to unmask.

    Returns:
        The unmasked tree.
    """

    is_masked = lambda x: type(x) is Mask

    return jtu.tree_map(
        f=lambda x: x.value if is_masked(x) else x,
        tree=tree,
        is_leaf=is_masked,
    )


def partition(
    tree: PyTree,
    *filters: Union[type, Callable[[Any], bool]],
    is_leaf: Callable[[Any], bool] = None,
) -> Tuple[PyTreeDef, Dict[str, Any]]:
    r"""Flattens a tree and partitions the leaves.

    The leaves are partitioned into a set of path-leaf mappings. The mapping in which a
    leaf is contained is chosen according to its oldest (closest to the root) ancestor
    that satisfies a constraint. If a node satisfies several constraints, the first one
    is selected. The last mapping is dedicated to leaves that do not satisfy any
    constraint.

    See also:
        :func:`inox.tree.combine`

    Arguments:
        tree: The tree to flatten.
        filters: A set of filtering constraints. Types are transformed into
            :py:`isinstance` constraints.
        is_leaf: A predicate for what to consider as a leaf.

    Returns:
        The tree definition and leaf partitions.

    Example:
        >>> tree = Namespace(a=1, b=jax.numpy.arange(2), c=['three', False])
        >>> treedef, leaves = inox.tree.partition(tree)
        >>> leaves
        {'.a': 1, '.b': Array([0, 1], dtype=int32), '.c[0]': 'three', '.c[1]': False}
        >>> treedef, arrays, others = inox.tree.partition(tree, jax.Array)
        >>> arrays
        {'.b': Array([0, 1], dtype=int32)}
        >>> others
        {'.a': 1, '.c[0]': 'three', '.c[1]': False}
    """

    treedef = jtu.tree_structure(tree, is_leaf)
    leaves = [{} for _ in filters] + [{}]

    def factory(filtr):
        if isinstance(filtr, type):
            return lambda x: isinstance(x, filtr)
        else:
            return filtr

    filters = list(map(factory, filters))

    if is_leaf is None:
        is_node = lambda x: any(filtr(x) for filtr in filters)
    else:
        is_node = lambda x: any(filtr(x) for filtr in filters) or is_leaf(x)

    def f(node_path, node):
        for i, filtr in enumerate(filters):  # noqa: B007
            if filtr(node):
                break
        else:
            i = len(filters)

        def g(leaf_path, leaf):
            key = jtu.keystr(node_path + leaf_path)
            leaves[i][key] = leaf

        jtu.tree_map_with_path(f=g, tree=node, is_leaf=is_leaf)

    jtu.tree_map_with_path(f=f, tree=tree, is_leaf=is_node)

    return treedef, *leaves


def combine(
    treedef: PyTreeDef,
    *leaves: Dict[str, Any],
) -> PyTree:
    r"""Reconstructs a tree from the tree definition and leaf partitions.

    See also:
        :func:`inox.tree.partition`

    Arguments:
        treedef: The tree definition.
        leaves: The set of leaf partitions.

    Returns:
        The reconstructed tree.

    Example:
        >>> tree = Namespace(a=1, b=jax.numpy.arange(2), c=['three', False])
        >>> treedef, arrays, others = inox.tree.partition(tree, jax.Array)
        >>> others = {key: str(leaf).upper() for key, leaf in others.items()}
        >>> inox.tree.combine(treedef, arrays, others)
        Namespace(
          a = '1',
          b = int32[2],
          c = ['THREE', 'FALSE']
        )
    """

    missing = []
    leaves = {key: leaf for partition in leaves for key, leaf in partition.items()}

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
        keys = ", ".join(f'"{key}"' for key in missing)

        raise KeyError(f"Missing key(s) in leaves: {keys}.")

    if leaves:
        keys = ", ".join(f'"{key}"' for key in leaves)

        raise KeyError(f"Unexpected key(s) in leaves: {keys}.")

    return tree


def prepr(
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
        >>> tree = [1, 'two', (True, False), list(range(5)), {'6': jax.numpy.arange(7)}]
        >>> tree.append(tree)
        >>> print(inox.tree.prepr(tree))
        [1, 'two', (True, False), [0, 1, 2, 3, 4], {'6': int32[7]}, [...]]
    """

    kwargs.update(
        linewidth=linewidth,
        typeonly=typeonly,
    )

    ancestors = kwargs.setdefault("ancestors", set())

    if id(tree) in ancestors:
        if hasattr(tree, "tree_repr"):
            return f"{type(tree).__name__}(...)"
        elif dataclasses._is_dataclass_instance(tree):
            return f"{type(tree).__name__}(...)"
        elif isinstance(tree, tuple):
            if hasattr(tree, "_fields"):
                return f"{type(tree).__name__}(...)"
            else:
                return "(...)"
        elif isinstance(tree, list):
            return "[...]"
        elif isinstance(tree, dict):
            return "{...}"

    ancestors.add(id(tree))

    if hasattr(tree, "tree_repr"):
        return tree.tree_repr(**kwargs)
    elif dataclasses._is_dataclass_instance(tree):
        bra, ket = f"{type(tree).__name__}(", ")"
        lines = [
            f"{field.name}={prepr(getattr(tree, field.name), **kwargs)}"
            for field in dataclasses.fields(tree)
            if field.repr
        ]
    elif isinstance(tree, tuple):
        if hasattr(tree, "_fields"):
            bra, ket = f"{type(tree).__name__}(", ")"
            lines = [
                f"{field}={prepr(value, **kwargs)}" for field, value in tree._asdict().items()
            ]
        else:
            bra, ket = "(", ")"
            lines = [prepr(x, **kwargs) for x in tree]
    elif isinstance(tree, list):
        bra, ket = "[", "]"
        lines = [prepr(x, **kwargs) for x in tree]
    elif isinstance(tree, dict):
        bra, ket = "{", "}"
        lines = [f"{repr(key)}: {prepr(value, **kwargs)}" for key, value in tree.items()]
    elif is_array(tree) and typeonly:
        return f"{tree.dtype}{list(tree.shape)}"
    else:
        return repr(tree).strip(" \n")

    ancestors.remove(id(tree))

    if any("\n" in line for line in lines):
        lines = ",\n".join(lines)
    elif sum(len(line) + 2 for line in lines) > linewidth:
        lines = ",\n".join(lines)
    else:
        lines = ", ".join(lines)

    if "\n" in lines:
        lines = "\n" + indent(lines, "  ") + "\n"

    return f"{bra}{lines}{ket}"
