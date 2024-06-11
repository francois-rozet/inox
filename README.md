![Inox's banner](https://raw.githubusercontent.com/francois-rozet/inox/master/docs/images/banner.svg)

# Stainless neural networks in JAX

Inox is a minimal [JAX](https://github.com/google/jax) library for neural networks with an intuitive [PyTorch](https://github.com/pytorch/pytorch)-like syntax. As with [Equinox](https://github.com/patrick-kidger/equinox), modules are represented as PyTrees, which enables complex architectures, easy manipulations, and functional transformations.

Inox aims to be a leaner version of Equinox by only retaining its core features: PyTrees and lifted transformations. In addition, Inox takes inspiration from other projects like [NNX](https://github.com/cgarciae/nnx) and [Serket](https://github.com/ASEM000/serket) to provide a versatile interface. Despite the differences, Inox remains compatible with the Equinox ecosystem, and its components (e.g. modules, transformations, ...) are for the most part interchangeable with those of Equinox.

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

Modules are defined with an intuitive PyTorch-like syntax,

```python
import jax
import inox.nn as nn

init_key, data_key = jax.random.split(jax.random.key(0))

class MLP(nn.Module):
    def __init__(self, key):
        keys = jax.random.split(key, 3)

        self.l1 = nn.Linear(3, 64, key=keys[0])
        self.l2 = nn.Linear(64, 64, key=keys[1])
        self.l3 = nn.Linear(64, 3, key=keys[2])
        self.relu = nn.ReLU()

    def __call__(self, x):
        x = self.l1(x)
        x = self.l2(self.relu(x))
        x = self.l3(self.relu(x))

        return x

model = MLP(init_key)
```

and are compatible with JAX transformations.

```python
X = jax.random.normal(data_key, (1024, 3))
Y = jax.numpy.sort(X, axis=-1)

@jax.jit
def loss_fn(model, x, y):
    pred = jax.vmap(model)(x)
    return jax.numpy.mean((y - pred) ** 2)

grads = jax.grad(loss_fn)(model, X, Y)
```

However, if a tree contains strings or boolean flags, it becomes incompatible with JAX transformations. For this reason, Inox provides lifted transformations that consider all non-array leaves as static.

```python
model.name = 'stainless'  # not an array

@inox.jit
def loss_fn(model, x, y):
    pred = inox.vmap(model)(x)
    return jax.numpy.mean((y - pred) ** 2)

grads = inox.grad(loss_fn)(model, X, Y)
```

For more information, check out the documentation at [inox.readthedocs.io](https://inox.readthedocs.io).

## Contributing

If you have a question, an issue or would like to contribute, please read our [contributing guidelines](https://github.com/francois-rozet/inox/blob/master/CONTRIBUTING.md).
