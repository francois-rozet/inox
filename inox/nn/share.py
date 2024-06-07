r"""Sharing modules

In a vanilla :class:`inox.nn.module.Module`, shared references to the same layer or
parameter would be treaded as separate copies and their weights would not be tied. The
:class:`Scope` class correctly handles such cases when shared references are explicited
with :class:`Reference`.

.. code-block:: python

    import inox
    import inox.nn
    import jax
    import jax.numpy as jnp

    class WeightSharingMLP(nn.Scope):
        def __init__(self, key):
            keys = jax.random.split(key, 3)

            self.l1 = nn.Linear(in_features=64, out_features=64, key=keys[0])
            self.l3 = nn.Linear(in_features=64, out_features=64, key=keys[1])
            self.l4 = nn.Linear(in_features=64, out_features=64, key=keys[2])

            self.l1 = nn.Reference('l1', self.l1)
            self.l2 = self.l1  # tied layer
            self.l3.weight = nn.Reference('l3.weight', self.l3.weight)
            self.l4.weight = self.l3.weight  # tied parameter

            self.relu = nn.ReLU()

        def __call__(self, x):  # standard __call__
            x = self.l1(x)
            x = self.l2(self.relu(x))
            x = self.l3(self.relu(x))
            x = self.l4(self.relu(x))

            return x

    key = jax.random.key(0)
    model = WeightSharingMLP(key)
    static, params, others = model.partition(nn.Parameter)

    print(inox.tree_repr(params))  # does not contain 'l2' and 'l4.weight'

.. code-block:: text

    {
      '.l1.value.bias.value': float32[64],
      '.l1.value.weight.value': float32[64, 64],
      '.l3.bias.value': float32[64],
      '.l3.weight.value.value': float32[64, 64],
      '.l4.bias.value': float32[64]
    }
"""

__all__ = [
    'Scope',
    'Reference',
]

import jax.tree_util as jtu

from typing import Any, Hashable

# isort: split
from .module import Module
from ..tree_util import PyTreeMeta, tree_repr


class Scope(Module):
    r"""Subclass of :class:`inox.nn.module.Module` which handles shared object
    references within its scope.

    All references with the same identification tag in a scope are considered to be the
    same and all but one copies are pruned during the flattening of the scope tree.
    Cyclic references are allowed, with the exception of a scope referencing itself.

    Warning:
        Shared references and in-place mutations are very hard to combine properly.
        Conversely, :class:`Reference` works seamlessly with :mod:`inox.nn.state` utils.

    Arguments:
        kwargs: A name-value mapping.
    """

    def tree_flatten(self):
        values, names = super().tree_flatten()
        visited = set()

        def prune(tree):
            def f(x):
                if isinstance(x, Reference):
                    if x.tag in visited:
                        x = Reference(x.tag, None)
                    else:
                        visited.add(x.tag)
                        x = Reference(x.tag, prune(x.value))

                return x

            def is_leaf(x):
                return isinstance(x, Reference) or isinstance(x, Scope)

            return jtu.tree_map(f, tree, is_leaf=is_leaf)

        return prune(values), names

    @classmethod
    def tree_unflatten(cls, names, values):
        visited = {}

        def unprune(tree):
            def f(x):
                if isinstance(x, Reference):
                    if x.tag in visited:
                        x = visited[x.tag]
                    else:
                        visited[x.tag] = x
                        x = Reference(x.tag, unprune(x.value))

                return x

            def is_leaf(x):
                return isinstance(x, Reference) or isinstance(x, Scope)

            return jtu.tree_map(f, tree, is_leaf=is_leaf)

        return super().tree_unflatten(names, unprune(values))

    def tree_repr(self, **kwargs) -> str:
        kwargs['references'] = set()
        return super().tree_repr(**kwargs)


class Reference(metaclass=PyTreeMeta):
    r"""Creates a reference to a value.

    A :class:`Reference` instance forwards :py:`__call__`,  :py:`__getattr__`, and
    :py:`__getitem__` operations to the value it references. For arithmetic operations
    (`+`, `*`, ...), use :py:`ref.value` directly instead.

    See also:
        :class:`Scope`

    Arguments:
        tag: An identification tag.
        value: The value to reference.

    Example:
        >>> weight = Reference('my-ref', Parameter(jax.random.ones((3, 5))))
        >>> weight  # repr preceded by an asterisk
        *Parameter(float[3, 5])
        >>> weight.shape
        (3, 5)
        >>> weight()
        Array([[1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.]], dtype=float32)
    """

    value: Any = None

    def __init__(self, tag: Hashable, value: Any):
        self.tag = tag
        self.value = value

    def __call__(self, *args, **kwargs) -> Any:
        return self.value(*args, **kwargs)

    def __delattr__(self, name: str):
        if name in ('tag', 'value'):
            object.__delattr__(self, name)
        else:
            delattr(self.value, name)

    def __delitem__(self, key: Hashable):
        del self.value[key]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.value, name)

    def __getitem__(self, key: Hashable):
        return self.value[key]

    def __setattr__(self, name: str, value: Any):
        if name in ('tag', 'value'):
            object.__setattr__(self, name, value)
        else:
            setattr(self.value, name, value)

    def __setitem__(self, key: Hashable, value: Any):
        self.value[key] = value

    def __repr__(self) -> str:
        return self.tree_repr()

    def tree_repr(self, **kwargs) -> str:
        references = kwargs.setdefault('references', set())

        if self.tag in references:
            return f'@{self.tag}'
        else:
            references.add(self.tag)
            return f'*{tree_repr(self.value, **kwargs)}'

    def tree_flatten(self):
        return [self.value], self.tag

    def tree_flatten_with_keys(self):
        return [(jtu.GetAttrKey('value'), self.value)], self.tag

    @classmethod
    def tree_unflatten(cls, tag, children):
        self = object.__new__(cls)
        self.tag = tag
        self.value = children[0]

        return self
