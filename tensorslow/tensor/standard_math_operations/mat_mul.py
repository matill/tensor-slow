from tensorslow.tensor.core import Operation
import numpy as np


# TODO: Make differentiable
# TODO: Check and find output shape
class MatMul(Operation):
    """Mutliplies two matrices"""

    def __init__(self, in_a, in_b):
        super().__init__([in_a, in_b], None)
        self.in_a = in_a
        self.in_b = in_b

    def compute(self, context):
        in_a = self.in_a.evaluate(context)
        in_b = self.in_b.evaluate(context)
        return np.matmul(in_a, in_b)

