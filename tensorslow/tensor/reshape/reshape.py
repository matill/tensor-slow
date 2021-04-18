from tensorslow.tensor.core import AssistedBackPropOperation, Operation
import numpy as np


class Squeeze(AssistedBackPropOperation):
    def __init__(self, input_node, axis):
        if input_node.shape is None:
            shape = None
        else:
            shape = np.squeeze(np.zeros(input_node.shape), axis=axis).shape

        self.axis = axis
        self.input_node = input_node
        super().__init__([input_node], shape)

    def compute(self, context):
        input_val = self.input_node.evaluate(context)
        return np.squeeze(input_val, self.axis)

    def get_parents_gradient_assisted(self, parent, self_gradient):
        assert parent.shape is not None
        return Reshape(self_gradient, parent.shape)


class Reshape(Operation):
    def __init__(self, input_node, shape):
        self.input_node = input_node
        super().__init__([input_node], shape)

    def compute(self, context):
        input_val = self.input_node.evaluate(context)
        return np.reshape(input_val, self.shape)

