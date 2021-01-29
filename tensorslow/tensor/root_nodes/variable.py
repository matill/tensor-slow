import numpy as np
from tensorslow.tensor.core import Tensor


# TODO: Some room for optimization with respect to memory management using more complex numpy functions.
class Variable(Tensor):
    """
    Evaluates to a fixed numpy array, but with the ability to update value, without changing shape.
    Similar to the Constant class, but the ComputeGraph API will be able to find all Variable types in
    the graph and update them when running gradient descent.
    """

    def __init__(self, np_array):
        super().__init__(np_array.shape)
        self.val = np_array

    def evaluate(self, context):
        return self.val

    def set_val(self, np_array):
        assert self.shape == np_array.shape
        self.val = np_array

    def increment(self, np_array):
        assert self.shape == np_array.shape
        self.val += np_array

    def decrement(self, np_array):
        assert self.shape == np_array.shape
        self.val += np_array

    def scale(self, alpha):
        alpha = float(alpha)
        self.val *= alpha

    def scaled_increment(self, np_array, alpha):
        alpha = float(alpha)
        assert self.shape == np_array.shape
        self.val += np_array * alpha
