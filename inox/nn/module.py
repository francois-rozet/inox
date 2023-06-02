r"""Base modules"""

from __future__ import annotations

__all__ = [
    'Module',
    'Buffer',
]

import jax
import jax.tree_util as jtu

from jax import Array
from typing import *

from ..tree_util import *


def is_array(x: Any) -> bool:
    return isinstance(x, Array)


def is_module(x: Any) -> bool:
    return isinstance(x, Module)


def is_buffer(x: Any) -> bool:
    return isinstance(x, Buffer)


class Module(Namespace):
    r""""""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def partition(
        self,
        include: Callable[[Any], bool] = None,
        exclude: Callable[[Any], bool] = None,
    ) -> Tuple[List[Any], List[Any], Callable[[List[Any], List[Any]], Module]]:
        r""""""

        if exclude is None:
            if include is None:
                exclude = is_buffer
            else:
                exclude = lambda x: False

        if include is None:
            include = is_array

        included, excluded, treedef = tree_partition(
            f=lambda x: include(x) and not exclude(x),
            tree=self,
            is_leaf=lambda x: include(x) or exclude(x),
        )

        return included, excluded, jtu.Partial(tree_merge, treedef)

    def replace(self, **kwargs):
        r""""""

        for name, value in kwargs.items():
            if hasattr(self, name):
                setattr(self, name, value)

        leaves = jtu.tree_leaves(self.__dict__, is_leaf=is_module)

        for leaf in leaves:
            if is_module(leaf):
                leaf.replace(**kwargs)

    def tree_flatten(self):
        children, static, treedef = tree_partition(
            f=lambda x: is_array(x) or is_module(x),
            tree=Namespace(self.__dict__),
            is_leaf=is_module,
        )

        return children, (treedef, tuple(static))

    def tree_flatten_with_keys(self):
        children, auxilary = self.tree_flatten()
        keys = tree_paths(
            tree=Namespace(self.__dict__),
            is_leaf=is_module,
        )

        return list(zip(keys, children)), auxilary

    @classmethod
    def tree_unflatten(cls, auxilary, children):
        namespace = tree_merge(*auxilary, children)

        self = object.__new__(cls)
        self.__dict__ = namespace.__dict__

        return self


class Buffer(Module):
    r""""""

    pass
