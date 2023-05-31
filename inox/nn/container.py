r"""Container modules"""

__all__ = [
    'Sequential',
]

import jax

from textwrap import indent
from typing import *

from .module import *
from ..tree_util import *


class Sequential(Module):
    r""""""

    def __init__(self, *layers: Module):
        self.layers = layers

    def __call__(self, x: Any) -> Any:
        for layer in self.layers:
            x = layer(x)
        return x

    def tree_repr(self, **kwargs) -> str:
        lines = (tree_repr(layer, **kwargs) for layer in self.layers)
        lines = ',\n'.join(lines)

        if lines:
            lines = '\n' + indent(lines, '  ') + '\n'

        return f'{self.__class__.__name__}({lines})'
