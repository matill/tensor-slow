import numpy as np
from tensorslow.tensor.core import AssistedBackPropOperation, Operation
from tensorslow.tensor.standard_math_operations import ElementwiseMultiply


class Relu(AssistedBackPropOperation):
    """
    RELU activation. y_i = x_i if x_i > 0 else 0.
    """

    def __init__(self, in_node):
        self.in_node = in_node
        super().__init__([in_node], in_node.shape)

    def compute(self, context):
        x = self.in_node.evaluate(context)
        return np.fmax(x, 0.)

    def get_parents_gradient_assisted(self, parent, self_gradient):
        self_over_parent_gradient = ReluDerivative(self.in_node)
        return ElementwiseMultiply([self_over_parent_gradient, self_gradient])


class ReluDerivative(Operation):
    def __init__(self, in_node):
        self.in_node = in_node
        super().__init__([in_node], in_node.shape)

    def compute(self, context):
        x = self.in_node.evaluate(context)

        # y_i is one of {-1, 0, 1}
        y = np.sign(x)
        # y_i is one of {0, 1}
        np.fmax(y, 0, out=y)
        return y
