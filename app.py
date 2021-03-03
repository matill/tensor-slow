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
# Replace self.is_j_dependent_on(j) with self.is_dependent_on(other)
# Instead? Make other.get_parents_gradient(self, j) return None if j does not depend on j?
# |--> This can open for new possibilities with constant(0) being equivalent with None.

# Other stuff
# LSTM builder utility
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


class Not(ts.Operation):
    def __init__(self, input_node):
        super().__init__([input_node], None)
        self.input = input_node

    def compute(self, context):
        return not self.input.evaluate(context)

# Queues, variables, and initialization
zero = ts.Constant(np.array([.0, .0, .0]))
queue = ts.FIFOQueue([
    # np.array([1, 2]),
    # np.array([1, 1]),
    np.array([0, 0]),
])

dim_in = 2
dim_hid = 3
wx = ts.Variable(np.arange(dim_in * 4 * dim_hid).reshape((4 * dim_hid, dim_in)) * 0.001)
wh = ts.Variable(np.arange(dim_hid * 4 * dim_hid).reshape((4 * dim_hid, dim_hid)) * 0.001)
b = ts.Variable(np.arange(4 * dim_hid).reshape((4 * dim_hid)))

# Loop start and recurrence relations
enter_loop = ts.EnterLoop()
h = ts.RecurrenceRelation(enter_loop, zero)
c = ts.RecurrenceRelation(enter_loop, zero)
x = ts.Dequeue(queue)

# Checks if loop has ended
loop_end_condition = Not(ts.IsQueueEmpty(queue))

# Computes output of sigmoid and tanh
L = ts.AddN([
    ts.MatMul(wx, x),
    ts.MatMul(wh, h),
    b
]).tag("L")
f =   ts.Sigmoid(L[0 * dim_hid : 1 * dim_hid])
i =   ts.Sigmoid(L[1 * dim_hid : 2 * dim_hid])
cprime = ts.Tanh(L[2 * dim_hid : 3 * dim_hid]).tag("cprime")
o =   ts.Sigmoid(L[3 * dim_hid : 4 * dim_hid]).tag("o")

# Computes next c
c_next = ts.Add2(
    ts.ElementwiseMultiply([c, f]),
    ts.ElementwiseMultiply([i, cprime])
)

# Computes next h
h_next = ts.ElementwiseMultiply([
    o,
    ts.Tanh(c_next).tag("cnext_tanh")
]).tag("h_next")

# Feed c_next and h_next to next iteration
next_iteration = ts.NextIteration(enter_loop, {c:c_next, h:h_next})

# End loop when queue is empty. Output h
exit_loop = ts.ExitLoop(next_iteration, loop_end_condition, [h])

# Get the last h value
h_out = ts.LoopOutput(exit_loop, h)


# cost = ts.SquaredError(dequeue, zero)
# loop_end_condition = IsSmaller(cost, 0.2)

# next_iteration = ts.NextIteration(enter_loop, {}).tag("next_iteration")
# exit_loop = ts.ExitLoop(next_iteration, loop_end_condition, [cost, dequeue]).tag("exit_loop")
# cost_out = ts.LoopOutput(exit_loop, cost).tag("cost_out")
# dequeue_out = ts.LoopOutput(exit_loop, dequeue).tag("dequeue_out")

context = {}
h_out = h_out.evaluate(context)
print("h_out: ", h_out, type(h_out))
# dequeue_out = dequeue_out.evaluate(context)
# print("dequeue_out: ", dequeue_out, type(dequeue_out))



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
