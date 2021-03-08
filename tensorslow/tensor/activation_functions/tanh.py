import numpy as np
from tensorslow.tensor.core import AssistedBackPropOperation
from tensorslow.tensor.standard_math_operations import ElementwiseMultiply, Subtract
from tensorflow.tensor.root_nodes.constant import Constant


class Tanh(AssistedBackPropOperation):

    def __init__(self, in_node):
        self.in_node = in_node
        super().__init__([in_node], in_node.shape)

    def compute(self, context):
        x = self.in_node.evaluate(context)
        y = x * (-2)
        np.exp(y, out=y)
        np.add(1, y, out=y)
        np.divide(2, y, out=y)
        np.subtract(y, 1, out=y)
        return y

    def get_parents_gradient_assisted(self, parent, self_gradient):
        return Subtract(
            Constant(np.array(1)),
            ElementwiseMultiply([self, self])
        )

