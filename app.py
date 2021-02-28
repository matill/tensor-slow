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


class IsSmaller(ts.Operation):
    def __init__(self, input_node, val):
        assert type(val) == float
        super().__init__([input_node], None)
        self.val = val
        self.input = input_node

    def compute(self, context):
        in_val = self.input.evaluate(context)
        return in_val > self.val


# class RecurrentNetwork(ts.ComputeGraph):
    # def __init__(self):


enter_loop = ts.EnterLoop()
queue = ts.FIFOQueue([
    np.array([1, 2, 3]),
    np.array([1, 1, 1]),
    np.array([0, 0, 0]),
])
dequeue = ts.Dequeue(queue)
zero = ts.Constant(np.array([0.1, 0.0, 0.0]))
cost = ts.SquaredError(dequeue, zero)
loop_end_condition = IsSmaller(cost, 0.2)

next_iteration = ts.NextIteration(enter_loop, {}).tag("next_iteration")
exit_loop = ts.ExitLoop(next_iteration, loop_end_condition, [cost, dequeue]).tag("exit_loop")
cost_out = ts.LoopOutput(exit_loop, cost).tag("cost_out")
dequeue_out = ts.LoopOutput(exit_loop, dequeue).tag("dequeue_out")

context = {}
cost_out = cost_out.evaluate(context)
print("cost_out: ", cost_out, type(cost_out))
dequeue_out = dequeue_out.evaluate(context)
print("dequeue_out: ", dequeue_out, type(dequeue_out))

# context = {}
# exit_loop.evaluate(context)
# for key, val in context.items():
#     print(type(key), val)
# print("context", json.dumps(context, indent=2))




# class Network(ts.ComputeGraph):
#     def __init__(self):
#         self.w1 = ts.Variable(np.array([1, 2, 3, 4, -1], dtype=float)).tag("w1")
#         self.w2 = ts.Constant(np.array([0, -1, 10, .5, 10], dtype=float)).tag("w2")
#         # self.w2 = ts.Input((2, 3))
#         self.w3 = ts.LRelu(self.w1, 0.001)
#         # self.j = ts.SquaredError(self.w1, self.w2).tag("j")
#         self.j = ts.SquaredError(self.w3, self.w2)

#     def train_step(self, step_size):
#         self.sgd(step_size, self.j, {})

#     def get_soft(self):
#         return self.soft.evaluate({})

#     # def get_w1(self):
#     #     return self.w1.evaluate({})


# nn = Network()
# print(nn.w1.evaluate({}))
# for _ in range(100000):
#     nn.train_step(0.04)
#     # print("w1", nn.w1.evaluate({}))
#     print("w3", nn.w3.evaluate({}))
#     # print("softmax here", nn.get_soft())
#     # nn.w1.print_json({})
