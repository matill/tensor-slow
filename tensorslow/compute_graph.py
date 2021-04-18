from .tensor import Variable
import numpy as np
import random


class ComputeGraph:
    """
    Class to represent a collection of nodes that form a graph.
    Provide methods to subclasses that help optimize cost functions:
        - momentum_sgd_epoch
    """

    def _compute_minibatch_gradient(self, minibatch, gradient_nodes, cost_node):
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
        """
        Performs a sgd epoch with momentum.
        Splits the dataset into minibatches.
        Caches state so that momentum is available between method calls.
        Parameters:
            step_size: Step size / learning rate
            cost_node: The node that computes the cost function
            batch_size: The size of the minibatches
            momentum_constant: The momentum constant, known from the momentum
                        algorithm.
            training_set: The training set, as a list of {ts.Input: np.ndarray}
                        dictionaries per datapoint
            parameters: (optional) A subset of the ts.Variable nodes the cost_node
                        depends on. If specified, only those parameters are updated
        """

        # Get the set of trainable parameters, their gradients, and their momentum
        parameters, gradient_nodes, momentum = \
                self._momentum_sgd_store_get(cost_node, parameters)

        # Initialize the batches
        random.shuffle(training_set)
        minibatches = self._get_minibatch_iterator(training_set, batch_size)

        cost = 0
        for minibatch in minibatches:

            # Compute the gradient using the minibatch
            minibatch_gradient, cost_ = self._compute_minibatch_gradient(
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
