import numpy as np
import json
import tensorslow as ts

# TODO:
# Write test cases for extending with gradients and evaluation
# Make Sum operations properly differentiable
# More operation types
# Prune / simplify rules?
# Add Zero-Tensor type and add where necessary?
# Tag gradients automatically if J and variable have names?
# Implement JacobianMultiply
# When finding the gradient of a node, the shape of the gradient and the original node should have equal shape. Add assertions for this.
# Add __init__ file in tensor folder
# evaluate() should check the shape
# Add readme


class Network(ts.ComputeGraph):
    def __init__(self):
        self.w1 = ts.Variable(np.array([[1., 2., 3.], [1., 2., 3.]])).tag("w1")
        self.w2 = ts.Constant(np.arange(6.0).reshape((2, 3))).tag("w2")
        # self.w2 = ts.Input((2, 3))
        self.j = ts.SquaredError(self.w1, self.w2).tag("j")

    def train_step(self, step_size):
        self.sgd(step_size, self.j, {})


nn = Network()
print(nn.w1.evaluate({}))
for _ in range(200):
    nn.train_step(0.02)
    print(nn.w1.evaluate({}))
    # nn.w1.print_json({})
