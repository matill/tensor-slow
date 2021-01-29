import numpy as np
from tensorslow.tensor.core import Operation


class SoftMax(Operation):
    """Adds N nodes of equal shape"""

    def __init__(self, in_node):
        # Assert that the input is a vector
        assert in_node.shape is not None
        assert len(in_node.shape) == 1
        self.in_node = in_node
        super().__init__([in_node], in_node.shape)

    def compute(self, context):
        # Get input
        in_val = self.in_node.evaluate(context)

        # Normalize so that x'_i = x_i - max_i(x) to reduce numeric over/underflow
        highest = np.max(in_val)
        normalized = in_val - highest
        exponented = np.exp(normalized, out=normalized)
        exp_sum = np.sum(exponented)
        exponented /= exp_sum
        print("in_val", in_val, "exponented", exponented)
        return exponented

