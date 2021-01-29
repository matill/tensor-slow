from tensorslow.tensor.core import BackPropOperation


class Subtract(BackPropOperation):
    """Evaluates to in_a - in_b"""

    def __init__(self, in_a, in_b):
        inputs = [in_a, in_b]
        shape = self.get_and_assert_common_shape_in_list(inputs)
        super().__init__(inputs, shape)
        self.in_a = in_a
        self.in_b = in_b

    def compute(self, context):
        a = self.in_a.evaluate(context)
        b = self.in_b.evaluate(context)
        return a - b

    def get_parents_gradient(self, parent, j):
        self_gradient = self.get_gradient(j)
        if parent == self.in_a and parent == self.in_b:
            raise NotImplementedError
        elif parent == self.in_a:
            parents_gradient = self_gradient
        elif parent == self.in_b:
            parents_gradient = StaticMultiply(self_gradient, -1.0)
        else:
            raise NotImplementedError

        return parents_gradient