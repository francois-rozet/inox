r"""Base building blocks"""

from __future__ import annotations

__all__ = [
    'Module',
    'Wrap',
    'Sequential',
]

import dataclasses
import jax

from functools import partial
from jax import Array
from textwrap import indent
from typing import *

from ..tree_util import *


class DataTree(type):
    r""""""

    def __new__(cls, *args, **kwargs):
        cls = super().__new__(cls, *args, **kwargs)
        cls = dataclasses.dataclass(cls, init=False, repr=False)

        jax.tree_util.register_pytree_node_class(cls)

        return cls


class ArrayRepr(object):
    r""""""

    def __init__(self, x: Array):
        self.shape = x.shape
        self.dtype = x.dtype

    def __repr__(self) -> str:
        return f'Array(shape={self.shape}, dtype={self.dtype})'


class Module(metaclass=DataTree):
    r""""""

    meta: Any = dataclasses.field(default=None, repr=False)

    def __init__(self):
        pass

    def __repr__(self) -> str:
        self = jax.tree_util.tree_map(ArrayRepr, self)

        names = [field.name for field in dataclasses.fields(self) if field.repr]
        values = [getattr(self, name) for name in names]

        lines = (
            f'{name} = {tree_repr(value)}'
            for name, value in zip(names, values)
        )

        lines = ',\n'.join(lines)

        if '\n' in lines:
            lines = '\n' + indent(lines, '  ') + '\n'

        return f'{self.__class__.__name__}({lines})'

    def tree_flatten(self):
        names = [field.name for field in dataclasses.fields(self)]
        values = [getattr(self, name) for name in names]

        children, static, treedef = tree_partition(
            f=lambda x: isinstance(x, Array) or isinstance(x, Module),
            tree=values,
            is_leaf=lambda x: isinstance(x, Module),
        )

        return children, (static, treedef)

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        self = object.__new__(cls)

        static, treedef = auxiliary
        values = tree_merge(children, static, treedef)
        names = [field.name for field in dataclasses.fields(self)]

        for name, value in zip(names, values):
            setattr(self, name, value)

        return self

    def partition(
        self,
        exclude: Callable[[Module], bool] = None,
    ) -> Tuple[List[Array], Callable[[List[Array]], Module]]:
        r""""""

        if exclude is None:
            f = lambda x: False
        else:
            f = lambda x: isinstance(x, Module) and exclude(x)

        excluded, state, treedef = tree_partition(f, self, f)
        build = partial(tree_merge, right=excluded, treedef=treedef)

        return state, build


class Wrap(Module):
    r""""""

    wrapped: Any

    def __init__(self, wrapped: Any):
        self.wrapped = wrapped

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({tree_repr(self.wrapped)})'

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)

    def __getitem__(self, key: Any) -> Any:
        return self.wrapped[key]


class Sequential(Module):
    r""""""

    layers: Sequence[Module]

    def __init__(self, *layers: Module):
        self.layers = layers

    def __repr__(self) -> str:
        lines = [
            f'({i}) {tree_repr(layer)}'
            for i, layer in enumerate(self.layers)
        ]

        lines = '\n'.join(lines)

        if '\n' in lines:
            lines = '\n' + indent(lines, '  ') + '\n'

        return f'{self.__class__.__name__}({lines})'

    def __call__(self, x: Any) -> Any:
        for layer in self.layers:
            x = layer(x)
        return x
