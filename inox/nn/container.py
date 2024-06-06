r"""Container modules"""

__all__ = [
    'Sequential',
]


from textwrap import indent
from typing import Any

# isort: split
from .module import Module
from ..tree_util import tree_repr


class Sequential(Module):
    r"""Creates a composition of layers.

    .. math:: y = f_n \circ \dots \circ f_2 \circ f_1(x)

    Arguments:
        layers: A sequence of layers :math:`f_1, f_2, \dots, f_n`.
    """

    def __init__(self, *layers: Module):
        self.layers = layers

    def __call__(self, x: Any) -> Any:
        r"""
        Arguments:
            x: The input :math:`x`.

        Returns:
            The output :math:`y`.
        """

        for layer in self.layers:
            x = layer(x)
        return x

    def tree_repr(self, **kwargs) -> str:
        lines = (tree_repr(layer, **kwargs) for layer in self.layers)
        lines = ',\n'.join(lines)

        if lines:
            lines = '\n' + indent(lines, '  ') + '\n'

        return f'{self.__class__.__name__}({lines})'
