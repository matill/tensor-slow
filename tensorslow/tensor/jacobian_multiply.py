from .core import Operation


class JacobianMultiply(Operation):
    def __init__(self, dJ_over_dY, dY_over_dX):
        # remember transpose
        raise NotImplementedError

    def compute(self, context):
        raise NotImplementedError

