import numpy as np
from .cost_function import CostFunction
from tensorslow.tensor.standard_math_operations import Subtract
from tensorslow.tensor.activation_functions import Sigmoid
from tensorslow.tensor.standard_math_operations import ScalarTensorMultiply


class SigmoidCrossEntropy(CostFunction):
    """
    Applies sigmoid to an input value, and computes cross entropy of a that
    using a ground truth value.
    """

    def __init__(self, sigmoid_input, ground_truth):
        """
        sigmoid_input: The input node used to compute sigmoid. Not a node that computes sigmoid.
        ground_truth: The value the sigmoid should should approximate.
        """

        # Assert that the inputs are vectors of equal size
        inputs = [sigmoid_input, ground_truth]
        self.get_and_assert_common_shape_in_list(inputs)

        # Store input nodes and initialize
        self.sigmoid_input = sigmoid_input
        self.ground_truth = ground_truth
        shape = ()
        super().__init__(inputs, shape)

    def compute(self, context):
        """
        Computes J = cross_entropy(sigmoid(x), y)
        where x = self.sigmoid_input and y = self.ground_truth.
        """

        # Get input
        x = self.sigmoid_input.evaluate(context)
        y = self.ground_truth.evaluate(context)

        # Compute
        sigmoid = 1 / (1 + np.exp(-x))
        term_a = y * np.log(sigmoid)
        term_b = (1 - y) * np.log(1 - sigmoid)
        return - term_a - term_b

    def get_jacobian_operation(self, parent):
        if parent is self.ground_truth:
            raise NotImplementedError
        elif parent is self.sigmoid_input:
            sigmoid = Sigmoid(self.sigmoid_input)
            return Subtract(sigmoid, self.ground_truth)
        else:
            raise NotImplementedError

