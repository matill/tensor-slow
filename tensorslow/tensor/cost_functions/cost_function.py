import numpy as np
from tensorslow.tensor.core import AssistedBackPropOperation
from tensorslow.tensor.standard_math_operations import ScalarTensorMultiply


class CostFunction(AssistedBackPropOperation):
    """
    Base class for cost functions (or any operation that returns a scalar).
    Implements get_parents_gradient_assisted() by multiplying the jacobian with its own derivative.
    Child classes need to implement get_jacobian_operation() and compute().
    """

    def get_parents_gradient_assisted(self, parent, self_gradient):
        jacobian = self.get_jacobian_operation(parent)
        return ScalarTensorMultiply(self_gradient, jacobian)
