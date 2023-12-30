r"""Base modules"""

from __future__ import annotations

__all__ = [
    'Module',
    'ModuleDef',
    'Variable',
    'Parameter',
    'Statistic',
]

import jax
import jax.tree_util as jtu

from jax import Array
from typing import *

from ..tree_util import PyArray, Static, Auto


def is_module(x: Any) -> bool:
    return isinstance(x, Module)


def is_parameter(x: Any) -> bool:
    return isinstance(x, Parameter)


class Module(Auto):
    r"""Base class for all modules.

    A module is a PyTree whose branches are its attributes. A branch can be any
    PyTree-compatible object (:py:`tuple`, :py:`list`, :py:`dict`, ...), including other
    modules. Parametric functions, such as neural networks, should subclass
    :class:`Module` and indicate their parameters with :class:`Parameter`.

    .. code-block:: python

        import jax
        import jax.random as jrd
        import inox
        import inox.nn as nn

        class Linear(nn.Module):
            def __init__(self, key, in_features, out_features):
                keys = jrd.split(key, 2)

                self.weight = Parameter(jrd.normal(keys[0], (in_features, out_features)))
                self.bias = Parameter(jrd.normal(keys[1], (out_features,)))

            def __call__(self, x):
                return x @ self.weight + self.bias

        class Classifier(nn.Module):
            def __init__(self, key, in_features, num_classes):
                keys = jrd.split(key, 3)

                self.l1 = Linear(keys[0], in_features, 64)
                self.l2 = Linear(keys[1], 64, 64)
                self.l3 = Linear(keys[2], 64, num_classes)
                self.relu = nn.ReLU()

                self.return_logits = True  # static leaf

            def __call__(self, x):
                x = self.l1(x)
                x = self.l2(self.relu(x))
                x = self.l3(self.relu(x))

                if self.return_logits:
                    return x
                else:
                    return jax.nn.softmax(x)

        key = jax.random.key(0)
        model = Classifier(key)

    Modules automatically detect non-array leaves and consider them as static. This
    results in module instances compatible with native JAX transformations
    (:func:`jax.jit`, :func:`jax.vmap`, :func:`jax.grad`, ...) out of the box.

    .. code-block:: python

        import optax

        @jax.jit
        def loss_fn(model, x, y):
            logits = jax.vmap(model)(x)
            loss = optax.softmax_cross_entropy(logits, y)

            return jax.numpy.mean(loss)

        grads = jax.grad(loss_fn)(model, x, y)

    However, JAX transformations are designed to work on pure functions. Some neural
    network layers, including batch normalization, are not pure as they hold a state
    which is updated as part of the layer's execution. In this case, using a
    functionally pure definition of the model is safer for training, but also convenient
    as some internal arrays do not require gradients.

    .. code-block:: python

        modef, params, others = model.pure(nn.Parameter)
        optimizer = optax.adamw(learning_rate=1e-3)
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, others, opt_state, x, y):  # gradient descent step
            def loss_fn(params):
                model = modef(params, others)
                logits = jax.vmap(model)(x)
                loss = optax.softmax_cross_entropy(logits, y)
                _, _, others = model.pure(nn.Parameter)

                return jax.numpy.mean(loss), others

            grads, others = jax.grad(loss_fn, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            return params, others, opt_state

        for x, y in trainset:  # training loop
            params, others, opt_state = step(params, others, opt_state, x, y)

        model = modef(params, others)

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

        leaves = jtu.tree_leaves(vars(self), is_leaf=is_module)

        for leaf in leaves:
            if is_module(leaf):
                leaf.train(mode)

        if hasattr(self, 'training'):
            self.training = mode

    def pure(self, *roles: type) -> Tuple[ModuleDef, Dict[str, Array]]:
        r"""Splits the functional definition of the module from its state.

        The state is represented by a path-array mapping split into different
        collections depending on the role (:class:`Parameter`, :class:`Statistic`, ...)
        of arrays. One collection is returned for each provided role plus an additional
        collection for arrays that do not fit any role.

        See also:
            :class:`ModuleDef`

        Arguments:
            roles: A set of :class:`Variable` subclasses.

        Returns:
            The module definition and state collection(s).

        Examples:
            >>> modef, state = model.pure()
            >>> clone = modef(state)

            >>> modef, params, others = model.pure(nn.Parameter)
            >>> params, opt_state = optimizer.update(grads, opt_state, params)
            >>> model = modef(params, others)
        """

        assert all(issubclass(role, Variable) for role in roles)

        state = [{} for _ in roles]
        state.append({})

        def is_leaf(x):
            return any(isinstance(x, role) for role in roles)

        leaves = jtu.tree_leaves_with_path(tree=self, is_leaf=is_leaf)

        for path, leaf in leaves:
            key = jtu.keystr(path)

            for i, role in enumerate(roles):
                if isinstance(leaf, role):
                    state[i][key] = leaf.value
                    break
            else:
                state[-1][key] = leaf

        return ModuleDef(self), *state


class ModuleDef(Static):
    r"""Abstraction for the static definition of a module.

    See also:
        :meth:`Module.pure`

    Arguments:
        module: A module instance.
    """

    def __init__(self, module: Module):
        self.value = jtu.tree_structure(module)

    def __call__(self, *state: Dict[str, Array]) -> Module:
        r"""
        Arguments:
            state: A set of state collections.

        Returns:
            A new instance of the module.
        """

        state = {
            key: leaf
            for collection in state
            for key, leaf in collection.items()
        }

        missing = []

        def f(path, leaf):
            key = jtu.keystr(path)

            if key in state:
                leaf = state.pop(key)
            else:
                missing.append(key)

            return leaf

        leaves = [object()] * self.value.num_leaves
        module = jtu.tree_unflatten(self.value, leaves)
        module = jtu.tree_map_with_path(f=f, tree=module)

        if missing:
            keys = ', '.join(f'"{key}"' for key in missing)

            raise RuntimeError(f"Missing key(s) in state: {keys}.")

        if state:
            keys = ', '.join(f'"{key}"' for key in state)

            raise RuntimeError(f"Unexpected key(s) in state: {keys}.")

        return module


class Variable(PyArray):
    r"""Wrapper to indicate variable arrays.

    Wrapping arrays in :class:`Variable` subclasses allows to split the state of modules
    into sub-collections.

    Arguments:
        value: An array value.
    """

    pass


class Parameter(Variable):
    r"""Wrapper to indicate optimizable arrays.

    All arrays that require gradient updates in a :class:`Module` should be wrapped in a
    :class:`Parameter` instance.

    Arguments:
        value: An array value.
    """

    pass


class Statistic(Variable):
    r"""Wrapper to indicate statistic arrays.

    See also:
        :class:`inox.nn.normalization.BatchNorm`

    Arguments:
        value: An array value.
    """

    pass
