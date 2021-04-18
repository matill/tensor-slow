from tensorslow.tensor.core import Operation
import numpy as np


class Sum(Operation):
    """
    Sum tensor elements over a given axis.
    Wraps np.sum()
    """

    def __init__(self, in_node, axis=None):
        # print("in_node", in_node)
        # print("axis", axis)
        self.in_node = in_node
        self.axis = axis

        # Get output shape
        if axis is None:
            shape = ()
        elif in_node.shape is None:
            shape = None
        else:
            # A little hacky and inefficient way to get the output shape
            # Generate an array with the given input shape, do the sum operation
            # over the given axis, and get the output shape
            shape = np.sum(np.zeros(in_node.shape), axis=axis).shape

        super().__init__([in_node], shape)

    def compute(self, context):
        in_val = self.in_node.evaluate(context)
        return np.sum(in_val, axis=self.axis)
