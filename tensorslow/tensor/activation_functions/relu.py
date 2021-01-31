import numpy as np
from tensorslow.tensor.core import BackPropOperation, Operation
from tensorslow.tensor.standard_math_operations import ElementwiseMultiply


class Relu(BackPropOperation):
    """
    RELU activation. y_i = x_i if x_i > 0 else 0.
    """

    def __init__(self, in_node):
        self.in_node = in_node
        super().__init__([in_node], in_node.shape)

    def compute(self, context):
        x = self.in_node.evaluate(context)
        return np.fmax(x, 0.)

    def get_parents_gradient(self, parent, j):
        # NOTE: Don't handle self == j
        jacobian = self.get_jacobian_operation(parent)
        self_gradient = self.get_gradient(j)
        return ElementwiseMultiply([jacobian, self_gradient])

    def get_jacobian_operation(self, parent):
        return ReluDerivative(self.in_node)


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
