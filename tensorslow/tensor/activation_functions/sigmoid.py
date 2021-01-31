import numpy as np
from tensorslow.tensor.core import Operation


class Sigmoid(Operation):

    def __init__(self, in_node):
        # Assert that the input is a scalar
        assert in_node.shape is not None
        assert in_node.shape == ()
        self.in_node = in_node
        super().__init__([in_node], ())

    def compute(self, context):
        x = self.in_node.evaluate(context)
        return 1 / (1 + np.exp(-x))

