r"""Sharing modules

In JAX, implicit shared references (objects with the same Python :py:`id`) to a node in
a tree are treated as distinct objects. While this design choice is reasonable, it makes
it difficult to express that two layers in a module are identical, and preserve their
shared identity through transformations.

Inox provides a mechanism to express explicitly such identity. During flattening of a
:class:`Scope` module, if several :class:`Reference` instances in the tree share the
same identification tag, the first occurence (depth-first order) is preserved, while the
following occurences are pruned. During unflattening, the pruned occurences are filled
in with the preserved occurence, which preserves their shared identity.

.. code-block:: python

    import inox
    import inox.nn as nn
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

    print(inox.tree.prepr(params))  # does not contain 'l2' and 'l4.weight'

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
    "Scope",
    "Reference",
]

import jax.tree_util as jtu

from typing import Any, Hashable, Iterator

import inox.nn as nn
import inox.tree


class Scope(nn.Module):
    r"""Subclass of :class:`inox.nn.module.Module` which handles shared object
    references within its scope.

    Arguments:
        kwargs: A name-value mapping.
    """

    def tree_flatten(self):
        children, meta = super().tree_flatten()
        memory = {}

        def prune(tree):
            def f(x):
                if isinstance(x, Reference):
                    if x.tag in memory:
                        y = memory[x.tag]
                    else:
                        y = memory[x.tag] = Reference(x.tag, None)

                    new = Reference(x.tag, None)

                    if x.obj is None:
                        pass
                    elif y.obj is None:
                        y.obj = x.obj
                        new.obj = prune(x.obj)
                    else:
                        assert x.obj is y.obj, f"#{x.tag} has conflicting identities"

                    return new

                return x

            def is_leaf(x):
                return isinstance(x, Reference) or isinstance(x, Scope)

            return jtu.tree_map(f, tree, is_leaf=is_leaf)

        return prune(children), meta

    @classmethod
    def tree_unflatten(cls, meta, children):
        memory = {}

        def unprune(tree):
            def f(x):
                if isinstance(x, Reference):
                    if x.tag in memory:
                        y, new = memory[x.tag]
                    else:
                        y, new = memory[x.tag] = Reference(x.tag, None), Reference(x.tag, None)

                    if x.obj is None:
                        pass
                    elif y.obj is None:
                        y.obj = x.obj
                        new.obj = unprune(x.obj)
                    else:
                        assert x.obj is y.obj, f"#{x.tag} has conflicting identities"

                    return new

                return x

            def is_leaf(x):
                return isinstance(x, Reference) or isinstance(x, Scope)

            return jtu.tree_map(f, tree, is_leaf=is_leaf)

        return super().tree_unflatten(meta, unprune(children))


class Reference(metaclass=inox.tree.PyTreeMeta):
    r"""Creates a reference to an object.

    A :class:`Reference` instance forwards :py:`__call__`, :py:`__iter__`,
    :py:`__getattr__`, and :py:`__getitem__` operations to the object it references. For
    arithmetic operations (`+`, `*`, ...), use :py:`ref.obj` directly instead.

    Arguments:
        tag: An identification tag.
        obj: The object to reference.

    Example:
        >>> dummy = ["zero", nn.Parameter(jax.numpy.ones((2, 3)))]
        >>> dummy = Reference("dummy-list", dummy)
        >>> dummy  # repr preceded by asterisk
        *['zero', Parameter(float32[2, 3])]
        >>> len(dummy)
        2
        >>> "zero" in dummy
        True
        >>> dummy[1].shape
        (2, 3)
    """

    def __init__(self, tag: Hashable, obj: Any):
        self.tag = tag
        self.obj = obj

    def __call__(self, *args, **kwargs) -> Any:
        return self.obj(*args, **kwargs)

    def __delattr__(self, name: str):
        if name in ("tag", "obj"):
            object.__delattr__(self, name)
        else:
            delattr(self.obj, name)

    def __getattr__(self, name: str) -> Any:
        if name in ("tag", "obj"):
            return object.__getattr__(self, name)
        else:
            return getattr(self.obj, name)

    def __setattr__(self, name: str, value: Any):
        if name in ("tag", "obj"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.obj, name, value)

    def __delitem__(self, key: Hashable):
        del self.obj[key]

    def __getitem__(self, key: Hashable) -> Any:
        return self.obj[key]

    def __setitem__(self, key: Hashable, value: Any):
        self.obj[key] = value

    def __contains__(self, key: Hashable) -> bool:
        return key in self.obj

    def __len__(self) -> int:
        return len(self.obj)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.obj)

    def __repr__(self) -> str:
        return self.tree_repr()

    def tree_repr(self, **kwargs) -> str:
        if self.obj is None:
            return f"#{self.tag}"
        else:
            return f"*{inox.tree.prepr(self.obj, **kwargs)}"

    def tree_flatten(self):
        return [self.obj], self.tag

    def tree_flatten_with_keys(self):
        return [(jtu.GetAttrKey("obj"), self.obj)], self.tag

    @classmethod
    def tree_unflatten(cls, tag, children):
        self = object.__new__(cls)
        self.tag = tag
        self.obj = next(iter(children))

        return self
