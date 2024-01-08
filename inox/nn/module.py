r"""Base modules

In Inox, a module is a PyTree whose branches are its attributes. A branch can be any
PyTree-compatible object (:py:`bool`, :py:`str`, :py:`list`, :py:`dict`, ...), including
other modules. Parametric functions, such as neural networks, should subclass
:class:`Module` and indicate their parameters with :class:`Parameter`.

.. code-block:: python

    import jax
    import jax.random as jrd
    import inox
    import inox.nn as nn

    class Linear(nn.Module):
        def __init__(self, in_features, out_features, key):
            keys = jrd.split(key, 2)

            self.weight = Parameter(jrd.normal(keys[0], (in_features, out_features)))
            self.bias = Parameter(jrd.normal(keys[1], (out_features,)))

        def __call__(self, x):
            return x @ self.weight() + self.bias()

    class Classifier(nn.Module):
        def __init__(self, in_features, num_classes, key):
            keys = jrd.split(key, 3)

            self.l1 = Linear(in_features, 64, key=keys[0])
            self.l2 = Linear(64, 64, key=keys[1])
            self.l3 = Linear(64, num_classes, key=keys[2])
            self.relu = nn.ReLU()

            self.return_logits = True

        def __call__(self, x):
            x = self.l1(x)
            x = self.l2(self.relu(x))
            x = self.l3(self.relu(x))

            if self.return_logits:
                return x
            else:
                return jax.nn.softmax(x)

    key = jax.random.key(0)
    model = Classifier(16, 3, key)
"""

from __future__ import annotations

__all__ = [
    'Module',
    'ModuleDef',
    'Parameter',
]

import jax
import jax.tree_util as jtu

from jax import Array
from typing import *

from ..tree_util import *
from ..tree_util import PyTreeDef


def is_module(x: Any) -> bool:
    return isinstance(x, Module)


class Module(Namespace):
    r"""Base class for all modules.

    Arguments:
        kwargs: A name-value mapping.
    """

    def train(self, mode: bool = True):
        r"""Toggles between training and evaluation modes.

        This method is primarily useful for (sub)modules that behave differently at
        training and evaluation, such as :class:`inox.nn.dropout.TrainingDropout` and
        :class:`inox.nn.normalization.BatchNorm`.

        Arguments:
            mode: Whether to turn training mode on or off.

        Example:
            >>> model.train(False)  # turns off dropout
        """

        for leaf in jtu.tree_leaves(vars(self), is_module):
            if is_module(leaf):
                leaf.train(mode)

        if hasattr(self, 'training'):
            self.training = mode

    def partition(
        self,
        *filters: Union[type, Callable[[Any], bool]],
    ) -> Tuple[ModuleDef, Dict[str, Array]]:
        r"""Splits the static definition of the module from its arrays.

        The arrays are partitioned into a set of path-array mappings. Each mapping
        contains the arrays of the subset of nodes satisfying the corresponding
        filtering constraint. The last mapping is dedicated to arrays that do not
        satisfy any constraint.

        See also:
            :class:`inox.tree_util.tree_mask` and :class:`inox.tree_util.tree_partition`

        Arguments:
            filters: A set of filtering constraints. Types are transformed into
                :py:`isinstance` constraints.

        Returns:
            The module static definition and array partitions.

        Examples:
            >>> static, arrays = model.partition()
            >>> clone = static(arrays)

            >>> static, params, others = model.partition(nn.Parameter)
            >>> params, opt_state = optimizer.update(grads, opt_state, params)
            >>> model = static(params, others)

            >>> model.path[2].layer.frozen = True
            >>> filtr = lambda x: getattr(x, 'frozen', False)
            >>> static, frozen, others = model.partition(filtr)
        """

        tree = tree_mask(self)
        treedef, *arrays = tree_partition(tree, *filters)

        return ModuleDef(treedef), *arrays


class ModuleDef(Static):
    r"""Abstraction for the static definition of a module.

    See also:
        :meth:`Module.partition`

    Arguments:
        treedef: A module tree definition.
    """

    def __init__(self, treedef: PyTreeDef):
        self.value = treedef

    def __call__(self, *arrays: Dict[str, Array]) -> Module:
        r"""
        Arguments:
            arrays: A set of array partitions.

        Returns:
            A new instance of the module.
        """

        tree = tree_combine(self.treedef, *arrays)
        tree = tree_unmask(tree)

        return tree

    @property
    def treedef(self) -> PyTreeDef:
        return self.value


class Parameter(NamedTuple):
    r"""Wrapper to indicate an optimizable array.

    All arrays that require gradient updates in a :class:`Module` should be wrapped in a
    :class:`Parameter` instance.

    Arguments:
        value: An array.

    Example:
        >>> weight = Parameter(jax.numpy.ones((3, 5))); weight
        Parameter(float32[3, 5])
        >>> def linear(x):
        ...     return x @ weight()
    """

    value: Array

    def __call__(self) -> Array:
        r"""
        Returns:
            The wrapped array.
        """

        return self.value

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.value, attr)

    def __repr__(self) -> str:
        return self.tree_repr()

    def tree_repr(self, **kwargs) -> str:
        return f'{self.__class__.__name__}({tree_repr(self.value, **kwargs)})'
