r"""Base building blocks"""

from __future__ import annotations

__all__ = [
    'Module',
    'Wrap',
    'Sequential',
]

import jax
import jax.tree_util as jtu

from jax import Array
from textwrap import indent
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


class Wrap(Module):
    r""""""

    def __init__(self, wrapped: Any, /, **kwargs):
        super().__init__(**kwargs)

        self.wrapped = wrapped

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)

    def __getitem__(self, key: Any) -> Any:
        return self.wrapped[key]

    def tree_repr(self, **kwargs) -> str:
        return f'{self.__class__.__name__}({tree_repr(self.wrapped, **kwargs)})'


class Sequential(Module):
    r""""""

    def __init__(self, *layers: Module):
        self.layers = layers

    def __call__(self, x: Any) -> Any:
        for layer in self.layers:
            x = layer(x)
        return x

    def tree_repr(self, **kwargs) -> str:
        lines = (tree_repr(layer, **kwargs) for layer in self.layers)
        lines = ',\n'.join(lines)

        if lines:
            lines = '\n' + indent(lines, '  ') + '\n'

        return f'{self.__class__.__name__}({lines})'
