import numpy as np
import json
import tensorslow as ts

# TODO:
# Write test cases for extending with gradients and evaluation
# Make Sum operations properly differentiable
# More operation types (Sigmoid, Relu, CrossEntropy)
# Make SoftMax operation differentiable if parent is CrossEntropy
# Make SoftMaxCrossEntropy differentiable with respect to ground_truth
# Make SoftMaxCrossEntropy able to back-propagate properly
# Create a common way to check if the same node already exists when creating it, and return the existing one????
# Make context an optional argument to Tensor.evaluate
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
        self.w1 = ts.Variable(np.array((-3), dtype=float)).tag("w1")
        self.w2 = ts.Constant(np.array((1), dtype=float)).tag("w2")
        # self.w2 = ts.Input((2, 3))
        # self.j = ts.SquaredError(self.w1, self.w2).tag("j")
        self.j = ts.SigmoidCrossEntropy(self.w1, self.w2)
        self.soft = ts.Sigmoid(self.w1)
        self.w3 = ts.Constant(np.array(0.69)).tag("w3")
        self.j2 = ts.SquaredError(self.j, self.w3)

    def train_step(self, step_size):
        self.sgd(step_size, self.j2, {})

    def get_soft(self):
        return self.soft.evaluate({})

    # def get_w1(self):
    #     return self.w1.evaluate({})


nn = Network()
print(nn.w1.evaluate({}))
for _ in range(2000):
    nn.train_step(0.02)
    print("w1", nn.w1.evaluate({}))
    print("softmax here", nn.get_soft())
    # nn.w1.print_json({})
