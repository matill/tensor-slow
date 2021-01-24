import numpy as np
import json


class Tensor:
    """Abstract base class for nodes in a graph that can be evaluated to return a numpy array."""

    def __init__(self):
        self.direct_dependent_nodes = []
        self.derivative_operations = {}
        self.name_tag = None

    def add_dependent_node(self, node):
        self.direct_dependent_nodes.append(node)

    def evaluate(self, context):
        raise NotImplementedError

    def get_gradient(self, j):
        # Make sure there exists a node that evaluates to the gradient
        if j not in self.derivative_operations:

            # Get a list of nodes that adds up to the gradient of j with respect to self
            gradient_subterms = []
            for node in self.direct_dependent_nodes:
                if node.is_j_dependent_on(j):
                    assert isinstance(node, BackPropOperation), "Could not extend compute graph with back-propagation when an operation ({node.class}) the cost function depends on is not differentiable."
                    gradient_through_node = node.get_parents_gradient(self, j)
                    gradient_subterms.append(gradient_through_node)

            # Reduce the list of nodes to a single node
            if len(gradient_subterms) == 0:
                raise NotImplementedError
            elif len(gradient_subterms) == 1:
                derivative_operation = gradient_subterms[0]
            else:
                derivative_operation = AddN(gradient_subterms)

            # Store result
            self.derivative_operations[j] = derivative_operation

        # Return the gradient node
        return self.derivative_operations[j]

    def is_j_dependent_on(self, j):
        if self is j:
            return True
        else:
            for node in self.direct_dependent_nodes:
                if node.is_j_dependent_on(j):
                    return True

        return False

    def __str__(self):
        nodes = "\n".join([node.__str__().replace('\n', '\n\t') for node in self.inputs])
        if len(nodes) > 0:
            nodes = "\n" + nodes
        
        if self.name_tag is None:
            return f"{type(self)}{nodes}"
        else:
            return f"{type(self)} ({self.name_tag}){nodes}"

    def to_dict(self, context):
        as_dict = {
            "type": str(type(self)),
            "val": str(self.evaluate(context))
        }

        if self.name_tag is not None:
            as_dict["name"] = self.name_tag


        if self.inputs != []:
            as_dict["inputs"] = [node.to_dict(context) for node in self.inputs]

        return as_dict

    def print_json(self, context):
        print(json.dumps(self.to_dict(context), indent=2))


class Operation(Tensor):
    """Abstract base class for operations in a graph that are evaluated using other input Tensors"""

    def evaluate(self, context):
        if self in context:
            return context[self]
        else:
            evaluated = self.compute(context)
            context[self] = evaluated
            return evaluated

    def compute(self, context):
        raise NotImplementedError


class BackPropOperation(Operation):
    def get_parents_gradient(self, parent, j):
        """
        Returns a node that computes d(J) / d(self) * d(self) / d(parent).
        If self == J, it returns a node that computes the d(self) / d(parent).
        """
        raise NotImplementedError

    def get_jacobian_operation(self, parent):
        """Returns a node that computes d(self) / d(parent)"""
        raise NotImplementedError


class Add2(BackPropOperation):
    """Differentiable operation of adding two tensors"""

    def __init__(self, in_a, in_b):
        super().__init__()
        in_a.add_dependent_node(self)
        in_b.add_dependent_node(self)
        self.in_a = in_a
        self.in_b = in_b
        self.inputs = [in_a, in_b]

    def compute(self, context):
        a = self.in_a.evaluate(context)
        b = self.in_b.evaluate(context)
        print("w3 computing")
        print("a", a)
        print("b", b)
        print("a+b", a+b)
        return a + b

    def get_parents_gradient(self, parent, j):
        self_gradient = self.get_gradient(j)
        num_inputs = 0
        num_inputs += int(self.in_a is parent)
        num_inputs += int(self.in_b is parent)
        assert num_inputs != 0, f"get_parents_gradient called with parent argument that is not a parent"
        if num_inputs == 1:
            return self_gradient
        else:
            parents_gradient = StaticMutiply(self_gradient, 2)
            return parents_gradient


class Subtract(BackPropOperation):
    def __init__(self, in_a, in_b):
        super().__init__()
        in_a.add_dependent_node(self)
        in_b.add_dependent_node(self)
        self.in_a = in_a
        self.in_b = in_b
        self.inputs = [in_a, in_b]

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
            parents_gradient = StaticMutiply(self_gradient, -1.0)
        else:
            raise NotImplementedError

        return parents_gradient


class AddN(BackPropOperation):
    """Differentiable operation of adding two tensors"""

    def __init__(self, inputs):
        super().__init__()
        for node in inputs:
            node.add_dependent_node(self)

        self.inputs = inputs

    def compute(self, context):
        return np.sum(x.evaluate(context) for x in self.inputs)

    def get_parents_gradient(self, parent, j):
        self_gradient = self.get_gradient(j)
        num_inputs = len(node for node in self.inputs if node is parent)
        assert num_inputs != 0, f"get_parents_gradient called with parent argument that is not a parent"
        if num_inputs == 1:
            return self_gradient
        else:
            parents_gradient = StaticMutiply(self_gradient, num_inputs)
            return parents_gradient


class StaticMutiply(Operation):
    def __init__(self, in_node, constant):
        super().__init__()
        self.in_node = in_node
        self.constant = constant
        in_node.add_dependent_node(self)
        self.inputs = [in_node]

    def compute(self, context):
        return self.in_node.evaluate(context) * self.constant


class SquaredError(BackPropOperation):
    def __init__(self, in_a, in_b):
        super().__init__()
        in_a.add_dependent_node(self)
        in_b.add_dependent_node(self)
        self.in_a = in_a
        self.in_b = in_b
        self.inputs = [in_a, in_b]

    def compute(self, context):
        a = self.in_a.evaluate(context)
        b = self.in_b.evaluate(context)
        print("a", a)
        print("b", b)
        print("a - b", a -b)
        norm = np.linalg.norm(a - b)
        return norm * norm * 0.5

    def get_parents_gradient(self, parent, j):
        jacobian = self.get_jacobian_operation(parent)
        if self is j:
            return jacobian
        else:
            return JacobianMultiply(self_gradient, jacobian)

    def get_jacobian_operation(self, parent):
        if parent == self.in_a and parent == self.in_b:
            raise NotImplementedError
        elif parent == self.in_a:
            jacobian = Subtract(self.in_a, self.in_b)
        elif parent == self.in_b:
            jacobian = Subtract(self.in_b, self.in_a)
        else:
            raise NotImplementedError

        return jacobian


class JacobianMultiply(Operation):
    def __init__(self, dJ_over_dY, dY_over_dX):
        # remember transpose
        raise NotImplementedError

    def compute(self, context):
        raise NotImplementedError


class Constant(Tensor):
    def __init__(self, np_array):
        super().__init__()
        self.val = np_array
        self.inputs = []

    def evaluate(self, context):
        return self.val


class ComputeGraph:
    """Class to represent a collection of nodes that create a compute graph"""

    def __init__(self, node_in_graph):
        """Creates a ComputeGraph object that finds all nodes in the graph by traversing the graph."""
        self.node_in_graph = node_in_graph
        self.nodes = set()

    def add_node(self, node):
        self.nodes.add(node)

    def prune(self, required_outputs):
        """Creates a new compute graph that only contains the nodes required to compute the nodes in required_outputs"""
        raise NotImplementedError

    def extend_with_gradients(self, j, gradient_targets):
        """
        Returns a (ComputeGraph, dict) tuple where the ComputeGraph is a new graph that contains gradients.
        j is a node that evaluates to a scalar value, which is the value being differentiated.
        gradient_targets is a list of nodes that j is beind differentiated with respect to.
        The dict in the return value is a map from node: gradient_node, for each node in gradient_targets.
        """
        gradient_map = {}
        for node in gradient_targets:
            gradient = node.get_gradient(j)
            gradient_map[node] = gradient

        return self, gradient_map

    def evaluate(self, context=None, required_outputs=None):
        """
        Evaluates the the graph. required_outputs can be specified as a subset
        of nodes that are required to be evaluated.
        """
        if context is None:
            context = {}

        if required_outputs is None:
            required_outputs = self.node_in_graph

        for node in required_outputs:
            node.evaluate(context)

        # return {node: context[node] for node in required_outputs}
        return context


