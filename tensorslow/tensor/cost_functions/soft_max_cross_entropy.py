import numpy as np
from .cost_function import CostFunction
from tensorslow.tensor.standard_math_operations import Subtract
from tensorslow.tensor.activation_functions import SoftMax


class SoftMaxCrossEntropy(CostFunction):
    """
    Applies softmax to an input value, and computes cross entropy of a that
    using a ground truth value.
    """

    def __init__(self, softmax_input, ground_truth):
        """
        softmax_input: The input node used to compute softmax. Not a node that computes softmax.
        ground_truth: The value the softmax should should approximate.
        """

        # Assert that the inputs are vectors of equal size
        inputs = [softmax_input, ground_truth]
        self.get_and_assert_common_shape_in_list(inputs)

        # Store input nodes and initialize
        self.softmax_input = softmax_input
        self.ground_truth = ground_truth
        shape = ()
        super().__init__(inputs, shape)

    def compute(self, context):
        """
        Computes J = cross_entropy(softmax(x), y)
        x_max * log(sum_i(e^(x_i - x_max))) * sum_i(y_i) - dot(y, x)
        where x = self.softmax_input and y = self.ground_truth.
        """

        # Get input
        x = self.softmax_input.evaluate(context)
        y = self.ground_truth.evaluate(context)

        # Compute
        dot = np.dot(x, y)
        y_sum = np.sum(y)
        x_max = np.max(x)
        x_normalized = x - x_max
        e_to_x_normalized = np.exp(x_normalized, out=x_normalized)
        sum_e = np.sum(e_to_x_normalized)
        log_of_sum = np.log(sum_e)
        return x_max * log_of_sum * y_sum - dot

    def get_jacobian_operation(self, parent):
        """
        Returns a node that computes (dJ / dself) * (dself / dparent)
        = y - SoftMax(x)
        where x = self.softmax_input and y = self.ground_truth
        """

        if parent is self.ground_truth:
            raise NotImplementedError
        elif parent is self.softmax_input:
            soft_max = SoftMax(self.softmax_input)
            return Subtract(soft_max, self.ground_truth)
        else:
            raise NotImplementedError

