from .core import Operation


class JacobianMultiply(Operation):
    def __init__(self, dJ_over_dY, dY_over_dX):
        shape_self = list(dJ_over_dY.shape)
        shape_jacobian = list(dY_over_dX.shape)
        num_axes = len(shape_jacobian) - 1
        self.dJ_over_dY = dJ_over_dY
        self.dY_over_dX = dY_over_dX
        super().__init__()
        raise NotImplementedError

    def compute(self, context):
        num_axes = len(y_gradient.shape) - 1
        return np.tensordot(y_gradient, y_jacobian, axes=num_axes)

