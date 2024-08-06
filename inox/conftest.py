r"""Doctests configuration."""

import inox
import jax
import pytest


@pytest.fixture(autouse=True, scope="module")
def doctest_imports(doctest_namespace):
    doctest_namespace["jax"] = jax
    doctest_namespace["inox"] = inox
    doctest_namespace["nn"] = inox.nn
