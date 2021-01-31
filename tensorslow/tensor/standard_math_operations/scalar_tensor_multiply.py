from tensorslow.tensor.core import Operation


class ScalarTensorMultiply(Operation):
    """
    Uses a scalar and a tensor as input. Returns the product
    """

    def __init__(self, scalar, tensor):
        self.scalar = scalar
        self.tensor = tensor
        assert self.scalar.shape == ()
        inputs = [scalar, tensor]
        shape = tensor.shape
        super().__init__(inputs, shape)

    def compute(self, context):
        scalar = self.scalar.evaluate(context)
        tensor = self.tensor.evaluate(context)
        return tensor * scalar
