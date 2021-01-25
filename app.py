# from mod import Constant, Add2, AddN, ComputeGraph, SquaredError
# from mod.tensor.core import AddN
import numpy as np
import json


from mod import AddN, Add2, StaticMultiply, Constant, SquaredError
from mod.compute_graph import ComputeGraph

# TODO:
# Redefine interface to ComputeGraph
# Write test cases
# Make Sum operations properly differentiable
# More operation types
# Split into multiple files
# Prune / simplify rules?
# Add SGD utility function to ComputeGraph
# Add readme


w1 = Constant(np.array([1., 2., 3.]))
w2 = Constant(np.array([4., 5., 6.]))
w3 = Add2(w1, w2)
j = SquaredError(w2, w3)
j.name = "j"
w1.name_tag = "w1"
w2.name_tag = "w2"
w3.name_tag = "w3"

graph = ComputeGraph()
graph, gradient_map = graph.extend_with_gradients(j, [w1, w2, w3])
w1_d = gradient_map[w1]
w2_d = gradient_map[w2]
w3_d = gradient_map[w3]
w1_d.name_tag = "w1_d"
w2_d.name_tag = "w2_d"
w3_d.name_tag = "w3_d"


# print("w1", w1.direct_dependent_nodes)
# print("w2", w2.direct_dependent_nodes)
# print("w3", w3.direct_dependent_nodes)
# print("gradient_map", gradient_map)

output = graph.evaluate(required_outputs = [w3, j, w3_d, w2_d])
print("\nout")
for node, val in output.items():
    node.print_json(output)
    print("\n\n")
# print(json.dumps(output, indent=2, default=str))
# print("w1_d", output[w1_d])
# print("w2_d", output[w2_d], w2_d)
# print("w3_d", output[w3_d], w3_d)