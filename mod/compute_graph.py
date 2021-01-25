
class ComputeGraph:
    """Class to represent a collection of nodes that create a compute graph"""

    def get_all_nodes(self, any_node):
        raise NotImplementedError

    def find_variable_nodes(self):
        raise NotImplementedError

    def sgd_on_variables(self, step_size, batch_size, cost_func_node):
        raise NotImplementedError

    def sgd(self, parameter_nodes, step_size, batch_size, cost_func_node):
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

        return gradient_map

    def evaluate(self, required_outputs, context=None):
        """
        Evaluates the the graph.
        required_outputs must be specified as a subset of nodes that are required to be evaluated.
        context must be specified if the graph contains Input nodes.
        """
        if context is None:
            context = {}

        for node in required_outputs:
            node.evaluate(context)

        # return {node: context[node] for node in required_outputs}
        return context
