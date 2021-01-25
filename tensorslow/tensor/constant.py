from .core import Tensor
import numpy as np


class Constant(Tensor):
    """Evaluates to a constant numpy array"""

    def __init__(self, np_array):
        super().__init__(np_array.shape)
        self.val = np_array

    def evaluate(self, context):
        return self.val

