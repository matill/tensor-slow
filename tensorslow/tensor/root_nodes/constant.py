import numpy as np
from tensorslow.tensor.core import Tensor


class Constant(Tensor):
    """Evaluates to a constant numpy array"""

    def __init__(self, np_array):
        super().__init__(np_array.shape)
        self.val = np_array

    def evaluate(self, context):
        return self.val

