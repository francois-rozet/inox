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

from ..tree_util import Auto


def is_placeholder(x: Any) -> bool:
    return isinstance(x, Placeholder)


def is_module(x: Any) -> bool:
    return isinstance(x, Module)


def is_buffer(x: Any) -> bool:
    return isinstance(x, Buffer)


class Placeholder(Auto):
    r"""Dummy placeholder class."""

    pass


class Module(Auto):
    r"""Base class for all modules.

    A module is a PyTree whose attributes are branches, meaning that you can assign any
    PyTree-compatible object (:py:`tuple`, :py:`list`, :py:`dict`, ...), including other
    modules, as regular attribute. Parametric functions, such as neural networks, should
    subclass :class:`Module`.

    .. code-block:: python

        import jax
        import inox
        import inox.nn as nn

        class Classifier(nn.Module):
            def __init__(self, key, in_features, num_classes):
                keys = jax.random.split(key, 3)

                self.l1 = nn.Linear(keys[0], in_features, 64)
                self.l2 = nn.Linear(keys[1], 64, 64)
                self.l3 = nn.Linear(keys[2], 64, num_classes)
                self.relu = nn.ReLU()
                self.softmax = nn.Softmax()

                self.return_logits = True  # static leaf

            def __call__(self, x):
                x = self.l1(x)
                x = self.l2(self.relu(x))
                x = self.l3(self.relu(x))

                if self.return_logits:
                    return x
                else:
                    return self.softmax(x)

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

    Arguments:
        kwargs: A name-value mapping.
    """

    def pure(self) -> Tuple[Module, Dict[str, Dict[str, Array]]]:
        r"""Returns a functionally pure copy of the module.

        JAX transformations are designed to work on pure functions. Some neural network
        layers, including batch normalization, are not pure as they hold a state which
        is updated as part of the layer's execution.

        The :meth:`Module.pure` method separates the functional definition of the module
        from its state, that is its parameters and buffers, which prevents unnoticed
        state mutations during training.

        .. code-block:: python

            # Impure
            output = model(*args)

            # Pure
            stateless, state = model.pure()
            output, mutations = stateless.apply(state, *args)
            state['buffers'].update(mutations)  # only buffers can mutate

        Using the functional definition of modules is safer but also handy for training
        with Optax optimizers when the model contains buffers.

        .. code-block:: python

            stateless, state = model.pure()
            params, buffers = state['params'], state['buffers']
            optimizer = optax.adamw(learning_rate=1e-3)
            opt_state = optimizer.init(params)

            @jax.jit
            def step(params, buffers, opt_state, x, y):  # gradient descent step
                def ell(params):
                    state = dict(params=params, buffers=buffers)
                    logits, mutations = stateless.apply(state, x)
                    loss = optax.softmax_cross_entropy(logits, y)

                    return jax.numpy.mean(loss), mutations

                grads, mutations = jax.grad(ell, has_aux=True)(params)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                buffers.update(mutations)

                return params, buffers, opt_state

            for x, y in trainset:  # training loop
                params, buffers, opt_state = step(params, buffers, opt_state, x, y)

            model = stateless.impure(dict(params=params, buffers=buffers))

        See also:
            :meth:`Module.impure` and :meth:`Module.apply`

        Returns:
            A stateless copy of the module and the state dictionary.
        """

        state = dict(params={}, buffers={})

        def f(path, leaf):
            if is_buffer(leaf):
                def g(subpath, subleaf):
                    key = jtu.keystr([*path, *subpath])
                    state['buffers'][key] = subleaf

                    return Placeholder()

                return jtu.tree_map_with_path(f=g, tree=leaf)
            else:
                key = jtu.keystr(path)
                state['params'][key] = leaf

                return Placeholder()

        stateless = jtu.tree_map_with_path(f=f, tree=self, is_leaf=is_buffer)

        return stateless, state

    def impure(self, state: Dict[str, Dict[str, Array]]):
        r"""Returns a functionally impure copy of the module.

        Arguments:
            state: The state dictionary.

        Returns:
            A copy of the module where parameters and buffers have been put back in
            place.
        """

        leaves = {
            key: leaf
            for substate in state.values()
            for key, leaf in substate.items()
        }

        def f(path, leaf):
            if is_placeholder(leaf):
                key = jtu.keystr(path)

                if key in leaves:
                    leaf = leaves.pop(key)
                else:
                    raise RuntimeError(f"Missing key \"{key}\" in state.")

            return leaf

        module = jtu.tree_map_with_path(f=f, tree=self, is_leaf=is_placeholder)

        if leaves:
            keys = ', '.join(f'"{key}"' for key in leaves)

            raise RuntimeError(f"Unexpected key(s) in state: {keys}.")

        return module

    def apply(
        self,
        state: Dict[str, Dict[str, Array]],
        *args,
        method: Union[str, Callable] = None,
        **kwargs,
    ) -> Tuple[Any, Dict[str, Array]]:
        r"""Applies a module method for a given state.

        Arguments:
            state: The state dictionary.
            method: The method to apply. Either a name (string) or a callable.
                If :py:`None`, :py:`__call__` is applied instead.
            args: The postitional arguments of the method.
            kwargs: The keyword arguments of the method.

        Returns:
            The method's output and the state mutations.
        """

        # Call
        module = self.impure(state)

        if method is None:
            output = module(*args, **kwargs)
        elif isinstance(method, str):
            output = getattr(module, method)(*args, **kwargs)
        else:
            output = method(module, *args, **kwargs)

        _, new_state = module.pure()

        # Mutations
        mutations = {}

        for key, buffer in new_state['buffers'].items():
            if buffer is not state['buffers'][key]:
                mutations[key] = buffer

        return output, mutations

    def train(self, mode: bool = True):
        r"""Toggles between training and evaluation modes.

        This is primarily useful for (sub)modules that behave differently at training
        and evaluation, such as :class:`inox.nn.dropout.TrainingDropout` and
        :class:`inox.nn.normalization.BatchNorm`.

        .. code-block:: python

            model.train(False)  # turns off dropout

        Arguments:
            mode: Whether to turn training mode on or off.
        """

        leaves = jtu.tree_leaves(self.__dict__, is_leaf=is_module)

        for leaf in leaves:
            if is_module(leaf):
                leaf.train(mode)

        if hasattr(self, 'training'):
            self.training = mode


class Buffer(Auto):
    r"""Container for non-optimizable arrays.

    All arrays that do not require gradient updates in a module, such as constants or
    running statistics should be leaves of a :class:`Buffer` instance.

    Arguments:
        kwargs: A name-value mapping.
    """

    pass
