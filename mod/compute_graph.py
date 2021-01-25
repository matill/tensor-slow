
class ComputeGraph:
    """Class to represent a collection of nodes that create a compute graph"""

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
