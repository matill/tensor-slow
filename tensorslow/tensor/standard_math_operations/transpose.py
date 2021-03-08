from tensorslow.tensor.core import AssistedBackPropOperation


class Transpose(AssistedBackPropOperation):
    """
    Returns the transpose of a matrix. Must be a 2 dimensional numpy array
    """

    def __init__(self, in_node):
        assert in_node.shape is None or len(in_node.shape) == 2, "Input to a " +\
                f"Transpose operation must be a 2D array. Got {in_node.shape}"

        if in_node.shape is None:
            shape = None
        else:
            a, b = in_node.shape
            shape = b, a

        self.in_node = in_node
        super().__init__([in_node], shape)

    def compute(self, context):
        return self.in_node.evaluate(context).T

    def get_parents_gradient_assisted(self, parent, self_gradient):
        return Transpose(self_gradient)
