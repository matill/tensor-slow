from .tensor import Variable


class ComputeGraph:
    """Class to represent a collection of nodes that create a compute graph"""

    def get_all_nodes(self, any_node, force_update=False):
        """
        Returns all nodes in the same graph as the given node.
        Caches the result from when it's called since this function has quite high complexity.
        Use force_update=True if the graph has changes since the last time it was called.
        """
        _all_nodes = getattr(self, '_all_nodes', None)
        if _all_nodes is None or force_update:
            _all_nodes = set()
            any_node.recursively_add_nodes_to_set(_all_nodes)
            self._all_nodes = _all_nodes

        return self._all_nodes

    def get_variables(self, any_node, force_update=False):
        """
        Returns all Variable nodes in the same graph as the given node.
        Caches the result from when it's called since this function has quite high complexity.
        Use force_update=True if the graph has changes since the last time it was called.
        """
        _variables = getattr(self, '_variables', None)
        if _variables is None or force_update:
            all_nodes = self.get_all_nodes(any_node, force_update=force_update)
            self._variables = set(node for node in all_nodes if isinstance(node, Variable))

        return self._variables

    def sgd(self, step_size, cost_func_node, input_map=None):
        """
        Runs a gradient descent step with the given cost function and input variables.
        input_map (optional): A dictionary {key: val} where key is an Input node and val is the value the Input node evaluates to.
        """
        input_map = {} if input_map is None else input_map

        # Find all nodes to be updated by traversing the graph, and store for next run
        variables = self.get_variables(cost_func_node)

        # Make sure the variables have gradient nodes
        gradient_map = self.extend_with_gradients(cost_func_node, variables)
        gradients = [val for key, val in gradient_map.items()]

        # Evaluate the gradients
        gradients = self.evaluate(gradients, context=input_map)

        # Increment all variables
        for variable in variables:
            # Get the variable's gradient node, get the (cached) evaluated value, and increment.
            gradient = variable.get_gradient(cost_func_node).evaluate(gradients)
            variable.scaled_increment(gradient, -step_size)

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

        return {node: context[node] for node in required_outputs}
        # return context
