class Tensor:
    """Abstract base class for nodes in a graph that can be evaluated to return a numpy array."""

    def __init__(self):
        self.direct_dependent_nodes = []

    def add_dependent_node(self, node):
        self.direct_dependent_nodes.append(node)

    def evaluate(self, context):
        raise NotImplementedError


class DifferentiableTensor(Tensor):
    """Abstract base class for tensors that have defined an Operation to compute the derivative using back-propagation"""

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


class DifferentiableOperation(Operation, DifferentiableTensor):
    """Abstract base class for operations in a graph that can be used to automatically generate a graph to compute gradients"""
    pass

