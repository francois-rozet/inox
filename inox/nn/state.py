r"""Stateful modules

In Inox, in-place module mutations are not prohibited, but are not recommended as they
often lead to silent errors around JAX transformations. Instead, it is safer to
externalize the state of modules and handle mutations explicitely.

The :mod:`inox.nn.state` module provides a simple interface to declare the state of
modules and apply state updates. During initialization, mutable arrays are wrapped in
:class:`StateEntry` instances. After initialization, these arrays are pulled out and
replaced with hashable :class:`StateKey` instances using the :func:`export_state`
function. The state is represented by a dictionary which is used and updated during the
module's execution.

.. code-block:: python

    import inox
    import inox.nn as nn
    import jax
    import jax.numpy as jnp

    class Moments(nn.Module):
        def __init__(self, features):
            self.first = nn.StateEntry(jnp.zeros(features))
            self.second = nn.StateEntry(jnp.ones(features))

        def __call__(self, x, state):
            first = state[self.first]
            second = state[self.second]

            state = update_state(state, {
                self.first: 0.9 * first + 0.1 * x,
                self.second: 0.9 * second + 0.1 * x**2,
            })

            return state

    class MLP(nn.Module):
        def __init__(self, in_features, num_classes, key):
            keys = jax.random.split(key, 3)

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
    model, state = nn.export_state(model)

    y, state = model(x, state)
"""

__all__ = [
    'StateEntry',
    'StateKey',
    'update_state',
    'export_state',
]

import jax.tree_util as jtu

from typing import Any, Dict, Hashable, NamedTuple, Tuple

# isort: split
from ..tree_util import PyTree


def is_entry(x: Any) -> bool:
    return isinstance(x, StateEntry)


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


def update_state(state: Dict, mutation: Dict) -> Dict:
    r"""Creates a copy of the state dictionary and updates it.

    Arguments:
        state: The state dictionary.
        mutation: The update.

    Returns:
        The updated state dictionary.
    """

    state = state.copy()
    state.update(mutation)

    return state


def export_state(tree: PyTree) -> Tuple[PyTree, Dict]:
    r"""Pulls the state entries out of a tree.

    State entries are replaced by state keys which can be used to index the state
    dictionary.

    Arguments:
        tree: A tree or module.

    Returns:
        The stateless tree and the state dictionary.

    Example:
        >>> tree = {'a': 1, 'b': StateEntry(jax.numpy.zeros(2))}
        >>> tree, state = export_state(tree)
        >>> tree
        {'a': 1, 'b': StateKey("['b']")}
        >>> state
        {StateKey("['b']"): Array([0., 0.], dtype=float32)}
        >>> state[tree['b']]
        Array([0., 0.], dtype=float32)
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
