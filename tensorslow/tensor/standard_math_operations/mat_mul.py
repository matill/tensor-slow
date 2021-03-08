from tensorslow.tensor.core import AssistedBackPropOperation
from .transpose import Transpose
import numpy as np


# TODO: Make differentiable
# TODO: Check and find output shape
class MatMul(AssistedBackPropOperation):
    """Mutliplies two matrices"""

    def __init__(self, in_a, in_b):
        if in_a.shape is not None and in_b.shape is not None:
            a_left, a_right = in_a.shape
            b_left, b_right = in_b.shape
            assert a_right == b_left, f"shapes: {in_a.shape} {in_b.shape}"
            shape = a_left, b_right
        else:
            shape = None

        super().__init__([in_a, in_b], shape)
        self.in_a = in_a
        self.in_b = in_b

    def compute(self, context):
        in_a = self.in_a.evaluate(context)
        in_b = self.in_b.evaluate(context)
        return np.matmul(in_a, in_b)

    def get_parents_gradient_assisted(self, parent, self_gradient):
        is_a = parent is self.in_a
        is_b = parent is self.in_b
        assert is_a != is_b, "Parent must be a xor b"
        if parent is self.in_a:
            return MatMul(self_gradient, Transpose(self.in_b))
        else:
            return MatMul(Transpose(self.in_a), self_gradient)
