class Tensor:
    """Abstract base class for nodes in a graph that can be evaluated to return a numpy array."""

    def __init__(self):
        self.direct_dependent_nodes = []

    def add_dependent_node(self, node):
        self.direct_dependent_nodes.append(node)

    def evaluate(self, context):
        raise NotImplementedError

    def get_derivative_operation(self):
        raise NotImplementedError



class Operation(Tensor):
    """Abstract base class for operations in a graph that are evaluated using other input Tensors""""

    def evaluate(self, context):
        if self in context:
            return context[self]
        else:
            evaluated = self.compute(context)
            context[self] = evaluated
            return evaluated

    def compute(self, context):
        raise NotImplementedError


class DifferentiableOperation(Operation):
    """Abstract base class for operations in a graph that can be used to automatically generate a graph to compute gradients"""

    def get_jacobian_operation_of_input_node(self):
        raise NotImplementedError


class ComputeGraph:
    """Class to represent a collection of nodes that create a compute graph"""

    def __init__(self, node_in_graph):
        """Creates a ComputeGraph object that finds all nodes in the graph by traversing the graph."""
        raise NotImplementedError

    def prune(self, required_outputs):
        """Creates a new compute graph that only contains the nodes required to compute the nodes in required_outputs"""
        raise NotImplementedError

    def with_gradients(self, output_scalar_node, gradient_targets):
        """
        Returns a (ComputeGraph, dict) tuple where the ComputeGraph is a new graph that contains gradients.
        output_scalar_node is a node that evaluates to a scalar value, which is the value being differentiated.
        gradient_targets is a list of nodes that output_scalar_node is beind differentiated with respect to.
        The dict in the return value is a map from gradient_target: gradient_node, for each gradient_target in gradient_targets.
        """
        raise NotImplementedError

    def evaluate(self, context=None, required_outputs=None):
        """Evaluates the the graph. required_outputs can be specified as a subset of nodes that are required to be evaluated"""
        raise NotImplementedError


class Add2(DifferentiableOperation):
    """Differentiable operation of adding two tensors"""
    def __init__(self, in_a, in_b):
        super().__init__(self)
        in_a.add_dependent_node(self)
        in_b.add_dependent_node(self)
        self.in_a = in_a
        self.in_b = in_b

    def compute(self, context):
        a = self.in_a.evaluate(context)
        b = self.in_a.evaluate(context)
        return a + b

    def get_derivative_operation(self):
        