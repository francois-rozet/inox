.. image:: images/banner.svg
   :class: only-light

.. image:: images/banner_dark.svg
   :class: only-dark

Inox
====

Inox is a minimal `JAX <https://github.com/google/jax>`_ library for neural networks with an intuitive PyTorch-like interface. As with `Equinox <https://github.com/patrick-kidger/equinox>`_, modules are represented as PyTrees, which allows to pass networks in and out of JAX transformations, like :func:`jax.jit` or :func:`jax.vmap`. However, Inox modules automatically detect non-array leaves, like hyper-parameters or boolean flags, and consider them as static. Consequently, Inox modules are compatible with native JAX transformations, and do not require custom lifted transformations.

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

Networks are defined with an intuitive PyTorch-like syntax,

.. code-block:: python

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

and are fully compatible with native JAX transformations.

.. code-block:: python

    X = jax.random.normal(data_key, (1024, 3))
    Y = jax.numpy.sort(X, axis=-1)

    @jax.jit
    def loss_fn(network, x, y):
        pred = jax.vmap(network)(x)
        return jax.numpy.mean((y - pred) ** 2)

    grads = jax.grad(loss_fn)(network, X, Y)

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
