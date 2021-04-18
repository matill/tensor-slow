import numpy as np
import json

"""
Contains the base classes for Tensor child classes, and some simple Operation classes
that are required to extend compute graphs with nodes to compute gradients, namely AddN and StaticMultiply.
"""


class Tensor:
    """Abstract base class for nodes in a graph that can be evaluated to return a numpy array."""

    def __init__(self, shape):
        self.direct_dependent_nodes = []
        self.derivative_operations = {}
        self.name_tag = None
        self.shape = shape

    def add_dependent_node(self, node):
        self.direct_dependent_nodes.append(node)

    def get_loop_tag(self):
        return None

    def recursively_add_nodes_to_set(self, node_set):
        """Adds this node to the set, and recursively adds all other nodes in the graph to the set"""
        if self in node_set:
            return

        node_set.add(self)
        for node in self.get_directly_related_nodes():
            node.recursively_add_nodes_to_set(node_set)

    def get_directly_related_nodes(self):
        """Returns all nodes that depend on this node"""
        return self.direct_dependent_nodes

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
        if self.name_tag is None:
            return f"{type(self)} with shape {self.shape}"
        else:
            return f"{type(self)} with shape {self.shape}: ({self.name_tag})"

    def to_dict(self, context):
        as_dict = {
            "type": str(type(self)),
            "val": str(self.evaluate(context)),
            "shape": self.shape
        }

        if self.name_tag is not None:
            as_dict["name"] = self.name_tag

        return as_dict

    def print_json(self, context):
        print(json.dumps(self.to_dict(context), indent=2))

    def tag(self, name_tag):
        self.name_tag = name_tag
        return self

    def __getitem__(self, key):
        return TensorIndex(self, key)


class Operation(Tensor):
    """Abstract base class for operations in a graph that are evaluated using other input Tensors"""

    def __init__(self, inputs, shape, find_loop_tag=True):
        super().__init__(shape)
        self.inputs = inputs

        if find_loop_tag:
            self.find_loop_tag_from_inputs()

        # Let the input nodes to know that this node depends on them as input
        for node in inputs:
            node.add_dependent_node(self)

    def get_loop_tag(self):
        return self.loop_tag

    def find_loop_tag_from_inputs(self):
        # Find the loop tags of the inputs and make sure they all have the same
        if len(self.inputs) == 0:
            loop_tag = None
            print("WARNING: Operation with no inputs:", self)
        else:
            loop_tag = self.inputs[0].get_loop_tag()
            for node in self.inputs:
                assert node.get_loop_tag() is loop_tag, "All inputs to the same \
                        Operation must be members of the same loop"

        # Notify the Loop of the new node that was added
        self.loop_tag = loop_tag
        if self.loop_tag is not None:
            self.loop_tag._add_operation(self)

    def get_directly_related_nodes(self):
        """Returns all nodes that depend on this node, and the ones this node depends on"""
        return self.direct_dependent_nodes + self.inputs

    def get_dependencies(self):
        """Returns all nodes this specific operaiton depends on"""

        dependencies = set(self.inputs)
        for node in self.inputs:
            if isinstance(node, Operation):
                dependencies |= node.get_dependencies()
        return dependencies


    def get_and_assert_common_shape_in_list(self, nodes):
        common_shape = None
        for node in nodes:
            if common_shape == None:
                common_shape = node.shape
            elif node.shape is not None:
                assert node.shape == common_shape, f"Different shapes {node.shape} and {common_shape}"

        return common_shape

    def to_dict(self, context):
        as_dict = super().to_dict(context)
        as_dict["inputs"] = [node.to_dict(context) for node in self.inputs]
        return as_dict

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
    """
    Base class for operations that have an extended interface to back propagate 
    gradients.
    """

    def get_parents_gradient(self, parent, j):
        """
        Returns a node that computes d(J) / d(self) * d(self) / d(parent).
        If self == J, it returns a node that computes the d(self) / d(parent).
        """
        raise NotImplementedError

    def get_jacobian_operation(self, parent):
        """Returns a node that computes d(self) / d(parent)"""
        raise NotImplementedError


class AssistedBackPropOperation(BackPropOperation):
    """
    Base class for operations that have an extended interface to back propagate
    gradients, but implements some more functionality that are common for most operation.
    """

    def get_parents_gradient(self, parent, j):
        assert parent in self.inputs, "get_parents_gradient called with parent argument that is not an input node"
        if self is j:
            assert self.shape == (), f"Tried to differentiate with j ({self}) not being a scalar"
            return self.get_jacobian_operation(parent)
        else:
            self_gradient = self.get_gradient(j)
            return self.get_parents_gradient_assisted(parent, self_gradient)

    def get_parents_gradient_assisted(self, parent, self_gradient):
        raise NotImplementedError


class AddN(BackPropOperation):
    """Adds N nodes of equal shape"""

    def __init__(self, inputs):
        shape = self.get_and_assert_common_shape_in_list(inputs)
        super().__init__(inputs, shape)

    def compute(self, context):
        return np.sum(x.evaluate(context) for x in self.inputs)

    def get_parents_gradient(self, parent, j):
        self_gradient = self.get_gradient(j)
        num_inputs = len([node for node in self.inputs if node is parent])
        assert num_inputs != 0, f"get_parents_gradient called with parent argument that is not a parent"
        if num_inputs == 1:
            return self_gradient
        else:
            parents_gradient = StaticMultiply(self_gradient, num_inputs)
            return parents_gradient


class StaticMultiply(Operation):
    """Scales the input node by a constant value"""

    def __init__(self, in_node, constant):
        super().__init__([in_node], in_node.shape)
        self.in_node = in_node
        self.constant = constant

    def compute(self, context):
        return self.in_node.evaluate(context) * self.constant


# TODO: Make differentiable
# Check shape
class TensorIndex(Operation):
    """
    Operation that indexes numpy arrays equivalently to the [] operator.
    Not meant to be used directly by users. Use Tensor[<args>] (which is
    implemented by Tensor.__getitem__(key)) to construct a TensorIndex
    object in a simple and correct way.
    """

    def __init__(self, in_node, key):
        super().__init__([in_node], None)
        self.in_node = in_node
        self.key = key

    def compute(self, context):
        in_val = self.in_node.evaluate(context)
        return in_val.__getitem__(self.key)
