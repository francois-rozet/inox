![Inox's banner](https://raw.githubusercontent.com/francois-rozet/inox/master/docs/images/banner.svg)

# Stainless neural networks in JAX

Inox is a minimal [JAX](https://github.com/google/jax) library for neural networks with an intuitive PyTorch-like interface. As with [Equinox](https://github.com/patrick-kidger/equinox), modules are represented as PyTrees, which allows to pass networks in and out of JAX transformations, like `jax.jit` or `jax.vmap`. However, Inox modules automatically detect non-array leaves, like hyper-parameters or boolean flags, and consider them as static. Consequently, Inox modules are compatible with native JAX transformations out of the box, and do not require custom lifted transformations.

> Inox means "stainless steel" in French 🔪

## Installation

The `inox` package is available on [PyPI](https://pypi.org/project/inox), which means it is installable via `pip`.

```
pip install inox
```

Alternatively, if you need the latest features, you can install it from the repository.

```
pip install git+https://github.com/francois-rozet/inox
```

## Getting started

Networks are defined with an intuitive PyTorch-like syntax,

```python
import jax
import inox.nn as nn

init_key, data_key = jax.random.split(jax.random.key(0))

class MLP(nn.Module):
    def __init__(self, key):
        keys = jax.random.split(key, 3)

        self.l1 = nn.Linear(keys[0], 3, 64)
        self.l2 = nn.Linear(keys[1], 64, 64)
        self.l3 = nn.Linear(keys[2], 64, 3)
        self.relu = nn.ReLU()

    def __call__(self, x):
        x = self.l1(x)
        x = self.l2(self.relu(x))
        x = self.l3(self.relu(x))

        return x

network = MLP(init_key)
```

and are fully compatible with native JAX transformations.

```python
X = jax.random.normal(data_key, (1024, 3))
Y = jax.numpy.sort(X, axis=-1)

@jax.jit
def loss_fn(network, x, y):
    pred = jax.vmap(network)(x)
    return jax.numpy.mean((y - pred) ** 2)

grads = jax.grad(loss_fn)(network, X, Y)
```

For more information, check out the documentation at [inox.readthedocs.io](https://inox.readthedocs.io).
