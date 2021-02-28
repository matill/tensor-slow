import numpy as np
from tensorslow.tensor.core import Operation


class Tanh(Operation):

    def __init__(self, in_node):
        self.in_node = in_node
        super().__init__([in_node], in_node.shape)

    def compute(self, context):
        x = self.in_node.evaluate(context)
        print(self, x)
        y = x*2
        np.exp(y, out=y)
        np.divide(2, y, out=y)
        y -= 1
        return y
