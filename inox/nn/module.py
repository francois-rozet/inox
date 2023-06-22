r"""Base modules"""

from __future__ import annotations

__all__ = [
    'Module',
    'Buffer',
]

import jax
import jax.tree_util as jtu
import numpy as np

from jax import Array
from typing import *

from ..tree_util import *


def is_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) or isinstance(x, Array)


def is_module(x: Any) -> bool:
    return isinstance(x, Module)


def is_static(x: Any) -> bool:
    return isinstance(x, Static)


def is_buffer(x: Any) -> bool:
    return isinstance(x, Buffer)


class Module(Namespace):
    r"""Base class for all modules.

    Models should subclass this class. A module is a PyTree whose attributes
    are branches, meaning that you can assign any PyTree-compatible object (:py:`tuple`,
    :py:`list`, :py:`dict`, ...), including other modules, as regular attribute.

    .. code-block:: python

        import jax
        import inox
        import inox.nn as nn

        class Classifier(nn.Module):
            def __init__(self, key, in_features, num_classes):
                keys = jax.random.split(key, 3)

                self.l1 = nn.Linear(keys[0], in_features, 64)
                self.l2 = nn.Linear(keys[1], 64, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(keys[2], p=0.5)
                self.softmax = nn.Softmax()

                self.return_logits = True  # static leaf

            def __call__(self, x):
                x = self.relu(self.l1(x))
                x = self.l2(self.dropout(x))

                if self.return_logits:
                    return x
                else:
                    return self.softmax(x)

        key = jax.random.PRNGKey(0)
        model = Classifier(key)

    Additionally, modules will automatically detect non-array leaves and mark them as
    static. This results in module instances compatible with :func:`jax.jit` and
    :func:`jax.vmap` out of the box.

    .. code-block:: python

        from optax import softmax_cross_entropy

        def loss(model, x, labels):
            return jax.mean(softmax_cross_entropy(jax.vmap(model)(x), labels))

        jax.jit(loss)(model, data, labels)  # works like a charm

    However, in-place modification of attributes does not work as one would expect at
    first sight. The correct way to modify an attribute is to replace it. The only
    exception are sub-modules, whose attributes can also be replaced in-place.

    .. code-block:: python

        model.attr = ['a', 'b']
        model.attr.append('c')  # model.attr is still ['a', 'b']
        model.attr = [*model.attr, 'c']  # model.attr is now ['a', 'b', 'c']
        model.dropout.q = 0.9  # model.dropout has changed
    """

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
        r"""Splits the module into two partitions based on inclusion and exclusion rules.

        By default, all :class:`Buffer` instances in the module tree are excluded from
        the first partition, leaving only array leaves that should be optimized. This is
        especially useful for training with :mod:`optax` optimizers.

        .. code-block:: python

            params, buffers, build = model.partition()
            optimizer = optax.adamw(learning_rate=1e-3)
            opt_state = optimizer.init(params)

            for _ in range(epochs):  # training loop
                ...
                grads = jax.grad(loss)(params, ...)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                ...

            model = build(params, buffers)

        Arguments:
            include: The inclusion rule. If :py:`include=None`, only include arrays.
            exclude: The exclusion rule. If :py:`include=None` and :py:`exclude=None`,
                exclude :class:`Buffer` instances.

        Returns:
            The two partitions and a function to re-construct the module.
        """

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
        r"""Replaces all occurrences of some attributes in the module tree.

        This is primarily useful to switch the mode of submodules that behave
        differently at training and evaluation, such as :class:`inox.nn.dropout.Dropout`
        and :class:`inox.nn.normalization.BatchNorm`.

        .. code-block:: python

            model.replace(training=False)  # turns off dropout

        Arguments:
            kwargs: The attributes to replace and their new values.
        """

        for name, value in kwargs.items():
            if hasattr(self, name):
                setattr(self, name, value)

        leaves = jtu.tree_leaves(self.__dict__, is_leaf=is_module)

        for leaf in leaves:
            if is_module(leaf):
                leaf.replace(**kwargs)


class Buffer(Namespace):
    r"""Subclass of :class:`inox.tree_util.Namespace` intended to contain
    non-optimizable arrays.

    All arrays that do not require gradients in a module, such as constants,
    hyper-parameters, running statistics or RNG keys, should be leaves of a
    :class:`Buffer` instance.
    """

    pass
