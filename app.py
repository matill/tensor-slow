import numpy as np
import json
import tensorslow as ts

# TODO:
# Write test cases for extending with gradients and evaluation
# Add Input tensor type
# Add Variable tensor type
# Add SGD utility function to ComputeGraph
# Make Sum operations properly differentiable
# More operation types
# Prune / simplify rules?
# Add Zero-Tensor type and add where necessary?
# Tag gradients automatically if J and variable have names?
# Implement JacobianMultiply
# Add readme


class Network(ts.ComputeGraph):
    def __init__(self):
        self.w1 = ts.Constant(np.array([1., 2., 3.])).tag("w1")
        self.w2 = ts.Constant(np.array([4., 5., 6.])).tag("w2")
        self.w3 = ts.Add2(self.w1, self.w2).tag("w3")
        self.j = ts.SquaredError(self.w2, self.w3).tag("j")
        self.w1_d = self.w1.get_gradient(self.j).tag("w1_d")
        self.w2_d = self.w2.get_gradient(self.j).tag("w2_d")
        self.w3_d = self.w3.get_gradient(self.j).tag("w3_d")

    def get_w2_d(self):
        return self.evaluate([self.w3_d])[self.w3_d]


nn = Network()
w2_d_val = nn.get_w2_d()
print(w2_d_val)
