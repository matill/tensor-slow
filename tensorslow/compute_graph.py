from .tensor import Variable
import numpy as np
import random


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

    def compute_minibatch_gradient(self, minibatch, gradient_nodes, cost_node):
        # Add the gradients at each input set
        batch_size = 0
        cost = 0
        for input_map in minibatch:
            batch_size += 1
            context = {key: val for (key, val) in input_map.items()}
            cost += cost_node.evaluate(context)
            if batch_size == 1:
                gradient_vals = [node.evaluate(context) for node in gradient_nodes]
            else:
                for gradient_node, gradient_val in zip(gradient_nodes, gradient_vals):
                    gradient_val += gradient_node.evaluate(context)

        # Divide the gradient and the cost by the batch size
        assert batch_size > 0, "Minibatch was empty"
        cost /= batch_size
        for gradient_val in gradient_vals:
            gradient_val /= batch_size

        return gradient_vals, cost

    def _get_minibatch_iterator(self, training_set, batch_size):
        """Returns a nested iterator of equally sized mini-batches"""

        num_datapoints = len(training_set)
        num_batches = int(np.ceil(num_datapoints / batch_size))
        return (
            (
                training_set[i] for i in 
                range(
                    batch_index*batch_size,
                    int(np.fmin(
                        num_datapoints,
                        (batch_index+1)*batch_size
                    ))
                )
            )
            for batch_index in range(num_batches)
        )

    def _momentum_sgd_store_get(self, cost_node, target_parameters):
        """
        Internal function that returns a tuple given the cost node:
            parameters: The list of trainable parameters (Variable node) that the
                        cost node depends on.
            gradient_nodes: A list of nodes that computes the gradient of the cost
                            node with respect to each of the trainable parameters.
            momentum: A list of np arrays representing the momentum of each 
                        parameter
        """

        # Get the momentum_data dict.
        momentum_data = getattr(self, 'momentum_data', None)
        momentum_data = {} if momentum_data is None else momentum_data
        self.momentum_data = momentum_data

        # Find and cache relevant nodes and state for this cost function
        if not cost_node in momentum_data:
            # Find all trainable parameters (variables), get their gradient
            # node and initialize their momentum to zero
            cost_node_dependencies = cost_node.get_dependencies()
            parameters = [
                x for x in cost_node_dependencies if type(x) is Variable
            ]

            gradient_nodes = [x.get_gradient(cost_node) for x in parameters]
            momentum = [np.zeros(x.shape) for x in parameters]

            # Store the data
            momentum_data[cost_node] = {
                'parameters': parameters,
                'gradient_nodes': gradient_nodes,
                'momentum': momentum,
            }

        # Retrieve cached data
        else:
            cost_node_data = momentum_data[cost_node]
            parameters = cost_node_data['parameters']
            gradient_nodes = cost_node_data['gradient_nodes']
            momentum = cost_node_data['momentum']

        return parameters, gradient_nodes, momentum

    def momentum_sgd_epoch(self, step_size, cost_node, batch_size, momentum_constant, training_set, parameters=None):
        # Get the set of trainable parameters, their gradients, and their momentum
        parameters, gradient_nodes, momentum = \
                self._momentum_sgd_store_get(cost_node, parameters)

        # Initialize the batches
        random.shuffle(training_set)
        minibatches = self._get_minibatch_iterator(training_set, batch_size)

        cost = 0
        for minibatch in minibatches:

            # Compute the gradient using the minibatch
            minibatch_gradient, cost_ = self.compute_minibatch_gradient(
                minibatch,
                gradient_nodes,
                cost_node
            )
            cost += cost_

            # Update momentum and update parameters
            for param_i, momentum_i, gradient_i in zip(parameters, \
                                    momentum, minibatch_gradient):
                momentum_i *= momentum_constant
                momentum_i += gradient_i
                param_i.scaled_increment(momentum_i, -step_size)

        print("cost:", cost)
        return cost

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
