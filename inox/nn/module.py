r"""Base modules"""

from __future__ import annotations

__all__ = [
    'Module',
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


class Module(Namespace):
    r""""""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def partition(
        self,
        include: Callable[[Any], bool] = None,
        exclude: Callable[[Any], bool] = None,
    ) -> Tuple[List[Any], Callable[[List[Any]], Module]]:
        r""""""

        if include is None:
            include = is_array

        if exclude is None:
            exclude = lambda x: False

        included, excluded, treedef = tree_partition(
            f=lambda x: include(x) and not exclude(x),
            tree=self,
            is_leaf=lambda x: include(x) or exclude(x),
        )

        if all(x is None for x in excluded):
            build = jtu.Partial(jtu.tree_unflatten, treedef)
        else:
            build = jtu.Partial(tree_merge, treedef, excluded)

        return included, build

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
