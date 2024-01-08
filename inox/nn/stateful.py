r"""Stateful modules

In Inox, in-place module mutations are not strictly prohibited, but are not recommended
as they often lead to silent errors around JAX transformations. Instead, it is safer to
externalize the state of modules and handle mutations explicitely.

The :mod:`inox.nn.stateful` module provides a simple interface to declare the state of
modules and define state mutations.

.. code-block:: python

    class Moments(Stateful):
        def __init__(self, features):
            self.first = StateEntry(jax.numpy.zeros(features))
            self.second = StateEntry(jax.numpy.ones(features))

        def __call__(self, x, state):
            first = state[self.first]
            second = state[self.second]

            state = self.update(state, {
                self.first: 0.9 * first + 0.1 * x,
                self.second: 0.9 * second + 0.1 * x**2,
            })

            return state

    class MLP(nn.Module):
        def __init__(self, in_features, num_classes, key):
            keys = jrd.split(key, 3)

            self.in_stats = Moments(in_features)
            self.out_stats = Moments(num_classes)

            self.l1 = nn.Linear(in_features, 64, key=keys[0])
            self.l2 = nn.Linear(64, 64, key=keys[1])
            self.l3 = nn.Linear(64, num_classes, key=keys[2])
            self.relu = nn.ReLU()

        def __call__(self, x, state):
            state = self.in_stats(x, state)

            x = self.l1(x)
            x = self.relu()
            x = self.l2(x)
            x = self.relu()
            x = self.l3(x)

            state = self.out_stats(x, state)

            return x, state

    key = jax.random.key(0)
    model = MLP(16, 3, key)
    model, state = inox.nn.pull_state(model)

    y, state = model(x, state)
"""

__all__ = [
    'Stateful',
    'StateEntry',
    'StateKey',
    'pull_state',
]

import jax
import jax.tree_util as jtu

from typing import *

from .module import Module
from ..tree_util import PyTree


def is_entry(x: Any) -> bool:
    return isinstance(x, StateEntry)


class Stateful(Module):
    r"""Base class for stateful modules.

    See also:
        :class:`pull_state`

    Arguments:
        kwargs: A name-value mapping.
    """

    @staticmethod
    def update(state: Dict, mutation: Dict) -> Dict:
        r"""Creates a copy of the state dictionary and updates it.

        Arguments:
            state: The state dictionary.
            mutation: The update.
        """

        state = state.copy()
        state.update(mutation)

        return state


class StateEntry(NamedTuple):
    r"""Wrapper to indicate a state entry.

    Arguments:
        value: A value.
    """

    value: Any


class StateKey(NamedTuple):
    r"""Wrapper to indicate a state key.

    Arguments:
        key: An hashable key.
    """

    key: Hashable

    def __repr__(self) -> str:
        return self.tree_repr()

    def tree_repr(self, **kwargs) -> str:
        return f'{self.__class__.__name__}({repr(self.key)})'


def pull_state(tree: PyTree) -> Tuple[PyTree, Dict]:
    r"""Pulls the state entries out of a tree.

    Arguments:
        tree: A tree or module.

    Returns:
        The stateless tree and the state dictionary.

    Example:
        >>> tree = {'a': 1, 'b': StateEntry(jax.numpy.zeros(2))}
        >>> tree, state = pull_state(tree)
        >>> tree
        {'a': 1, 'b': StateKey("['b']")}
        >>> state
        {StateKey("['b']"): Array([0., 0.], dtype=float32)}
    """

    nodes, treedef = jtu.tree_flatten_with_path(tree, is_entry)
    leaves = []
    state = {}

    for path, node in nodes:
        if is_entry(node):
            key = StateKey(jtu.keystr(path))
            node, state[key] = key, node.value

        leaves.append(node)

    tree = jtu.tree_unflatten(treedef, leaves)

    return tree, state
