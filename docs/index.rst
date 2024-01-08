.. image:: images/banner.svg
   :class: only-light

.. image:: images/banner_dark.svg
   :class: only-dark

Inox
====

Inox is a minimal `JAX <https://github.com/google/jax>`_ library for neural networks with an intuitive PyTorch-like syntax. As with `Equinox <https://github.com/patrick-kidger/equinox>`_, modules are represented as PyTrees, which enables complex architectures, easy manipulations, and functional transformations.

Inox aims to be a leaner version of `Equinox <https://github.com/patrick-kidger/equinox>`_ by only retaining its core features: PyTrees and lifted transformations. In addition, Inox takes inspiration from other projects like `NNX <https://github.com/cgarciae/nnx>`_ and `Serket <https://github.com/ASEM000/serket>`_ to provide a versatile interface.

Installation
------------

The :mod:`inox` package is available on `PyPI <https://pypi.org/project/inox>`_, which means it is installable via `pip`.

.. code-block:: console

    pip install inox

Alternatively, if you need the latest features, you can install it from the repository.

.. code-block:: console

    pip install git+https://github.com/francois-rozet/inox

Getting started
---------------

Models are defined with an intuitive PyTorch-like syntax,

.. code-block:: python

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

and are compatible with JAX transformations.

.. code-block:: python

    X = jax.random.normal(data_key, (1024, 3))
    Y = jax.numpy.sort(X, axis=-1)

    @jax.jit
    def loss_fn(model, x, y):
        pred = jax.vmap(model)(x)
        return jax.numpy.mean((y - pred) ** 2)

    grads = jax.grad(loss_fn)(model, X, Y)

However, if a module contains strings or flags, it becomes incompatible with JAX transformations. For this reason, Inox provides lifted transformations that consider all non-array leaves as static.

.. code-block:: python

    model.name = 'stainless'

    @inox.jit
    def loss_fn(model, x, y):
        pred = inox.vmap(model)(x)
        return jax.numpy.mean((y - pred) ** 2)

    grads = inox.grad(loss_fn)(model, X, Y)

.. toctree::
    :caption: inox
    :hidden:
    :maxdepth: 2

    tutorials.rst
    api.rst

.. toctree::
    :caption: Development
    :hidden:
    :maxdepth: 1

    Contributing <https://github.com/francois-rozet/inox/blob/master/CONTRIBUTING.md>
    Changelog <https://github.com/francois-rozet/inox/releases>
    License <https://github.com/francois-rozet/inox/blob/master/LICENSE>
