import numpy as np
from tensorslow.tensor.core import Tensor

class Input(Tensor):
    """Node with optionally strict shape that is evaluated according to a value that is set in the context dictionary"""

    def evaluate(self, context):
        assert self in context, f"Attempted to evaluate graph without specifying value of Input node: {self}"
        value = context[self]
        if self.shape is not None:
            assert self.shape == value.shape, f"Input node {self} expected value of shape {self.shape} != {value.shape}"

        return value

