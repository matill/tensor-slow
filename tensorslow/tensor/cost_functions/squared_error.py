import numpy as np
from .cost_function import CostFunction
from tensorslow.tensor.standard_math_operations import Subtract


class SquaredError(CostFunction):
    """Evaluates to half of the squared L2 difference between in_a and in_b"""

    def __init__(self, in_a, in_b):
        super().__init__([in_a, in_b], ())
        self.in_a = in_a
        self.in_b = in_b

    def compute(self, context):
        a = self.in_a.evaluate(context)
        b = self.in_b.evaluate(context)
        norm = np.linalg.norm(a - b)
        return norm * norm * 0.5

    def get_jacobian_operation(self, parent):
        if parent == self.in_a and parent == self.in_b:
            raise NotImplementedError
        elif parent == self.in_a:
            jacobian = Subtract(self.in_a, self.in_b)
        elif parent == self.in_b:
            jacobian = Subtract(self.in_b, self.in_a)
        else:
            raise NotImplementedError

        return jacobian

