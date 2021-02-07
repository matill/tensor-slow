import numpy as np
import json
import tensorslow as ts


# TODO:
# Core stuff
# When finding the gradient of a node, the shape of the gradient and the original node should have equal shape. Add assertions for this.
# Create a common way to check if the same node already exists when creating it, and return the existing one????
# Make context an optional argument to Tensor.evaluate
# Prune / simplify rules?
# evaluate() should check the shape
# Special support for when the node we compute derivatives of has trivial gradients (sum, static multiply, etc...)
# General way to handle if self == j when creating a derivative node (self.shape == (), simplified subgraph (without JacobianMultiply, etc...)) 
# Tag gradients automatically if J and variable have names?
# Add support for momentum at training

# Other stuff
# Write test cases for extending with gradients and evaluation
# Make Sum operations properly differentiable
# Make SoftMax operation differentiable if parent is CrossEntropy
# Add Zero-Tensor type and add where appropriate?
# Implement JacobianMultiply
# Sigmoid should not require that in_node.shape == ()
# Add CostFunction super-class
# Add readme


class Network(ts.ComputeGraph):
    def __init__(self):
        self.w1 = ts.Variable(np.array([1, 2, 3, 4, -1], dtype=float)).tag("w1")
        self.w2 = ts.Constant(np.array([0, -1, 10, .5, 10], dtype=float)).tag("w2")
        # self.w2 = ts.Input((2, 3))
        self.w3 = ts.LRelu(self.w1, 0.001)
        # self.j = ts.SquaredError(self.w1, self.w2).tag("j")
        self.j = ts.SquaredError(self.w3, self.w2)

    def train_step(self, step_size):
        self.sgd(step_size, self.j, {})

    def get_soft(self):
        return self.soft.evaluate({})

    # def get_w1(self):
    #     return self.w1.evaluate({})


nn = Network()
print(nn.w1.evaluate({}))
for _ in range(100000):
    nn.train_step(0.04)
    # print("w1", nn.w1.evaluate({}))
    print("w3", nn.w3.evaluate({}))
    # print("softmax here", nn.get_soft())
    # nn.w1.print_json({})
