import numpy as np
from tensorslow.tensor.core import AssistedBackPropOperation, Operation
from tensorslow.tensor.standard_math_operations import ElementwiseMultiply


class LRelu(AssistedBackPropOperation):
    """
    RELU activation. y_i = x_i if x_i > 0 else x_i * leaky_scale.
    leaky_scale is a hyperparameter. (a simple scalar, not a node that evaluates to a scalar).
    """

    def __init__(self, in_node, leaky_scale):
        self.in_node = in_node
        self.leaky_scale = leaky_scale
        super().__init__([in_node], in_node.shape)

    def compute(self, context):
        x = self.in_node.evaluate(context)
        negatives = np.fmin(x, 0.)
        positives = np.fmax(x, 0.)
        return positives + negatives * self.leaky_scale

    def get_parents_gradient_assisted(self, parent, self_gradient):
        self_over_parent_gradient = LReluDerivative(self.in_node, self.leaky_scale)
        return ElementwiseMultiply([self_over_parent_gradient, self_gradient])


class LReluDerivative(Operation):
    def __init__(self, in_node, leaky_scale):
        self.in_node = in_node
        self.leaky_scale = leaky_scale
        super().__init__([in_node], in_node.shape)

    def compute(self, context):
        x = self.in_node.evaluate(context)

        # y_i is one of {-1, 0, 1}
        y = np.sign(x)
        # y_i is one of {-1, 0}
        np.fmin(y, 0, out=y)
        # y_i is one of {0, self.leaky_scale-1}
        y *= (self.leaky_scale - 1)
        # y_i is one of {1, self.leaky_scale}
        y += 1
        return y

