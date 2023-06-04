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


def is_static(x: Any) -> bool:
    return isinstance(x, Static)


def is_buffer(x: Any) -> bool:
    return isinstance(x, Buffer)


class Module(Namespace):
    r""""""

    def __getattribute__(self, name: str):
        if name in super().__dict__:
            return jtu.tree_map(
                f=lambda x: x() if is_static(x) else x,
                tree=super().__dict__[name],
                is_leaf=lambda x: is_module(x) or is_static(x),
            )
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any):
        super().__setattr__(name, value)

        if name in super().__dict__:
            super().__setattr__(
                name,
                jtu.tree_map(
                    f=lambda x: x if is_array(x) or is_module(x) else Static(x),
                    tree=value,
                    is_leaf=lambda x: is_module(x),
                )
            )

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


class Buffer(Module):
    r""""""

    pass
