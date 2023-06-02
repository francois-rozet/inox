r"""Functional transformations"""

__all__ = [
    'vmap',
]

import jax
import jax.tree_util as jtu

from typing import *

from .tree_util import *


def flatten_axes(tree: Any, axes: Any):
    flat_axes = []

    def extend(axis, x):
        flat_axes.extend([axis] * len(jtu.tree_leaves(x)))

    jtu.tree_map(extend, axes, tree, is_leaf=lambda x: x is None)

    return flat_axes


class vmap(Namespace):
    r""""""

    def __init__(
        self,
        fun: Callable,
        /,
        in_axes: Any = 0,
        out_axes: Any = 0,
        **kwargs,
    ):
        self.fun = fun
        self.in_axes = in_axes
        self.out_axes = out_axes
        self.kwargs = kwargs

    def __call__(self, *ins) -> Any:
        flat_in_axes = flatten_axes(ins, self.in_axes)
        flat_ins, treedef_ins = jtu.tree_flatten(ins)

        flat_out_axes = []
        store = []

        def flat_fun(flat_ins):
            ins = jtu.tree_unflatten(treedef_ins, flat_ins)
            outs = self.fun(*ins)

            flat_out_axes.extend(flatten_axes(outs, self.out_axes))
            flat_outs, treedef_outs = jtu.tree_flatten(outs)

            store.append(treedef_outs)

            return flat_outs

        flat_outs = jax.vmap(flat_fun, (flat_in_axes,), flat_out_axes, **self.kwargs)(flat_ins)
        treedef_outs = store.pop()

        return jtu.tree_unflatten(treedef_outs, flat_outs)
