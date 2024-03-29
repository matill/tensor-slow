import numpy as np
from tensorslow.tensor.core import Operation


class Sigmoid(Operation):

    def __init__(self, in_node):
        self.in_node = in_node
        super().__init__([in_node], in_node.shape)

    def compute(self, context):
        x = self.in_node.evaluate(context)
        return 1 / (1 + np.exp(-x))

