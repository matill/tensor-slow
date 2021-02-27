
import numpy as np
from tensorslow.tensor.core import Tensor, Operation


class LoopInput(Operation):
    """
    Used to access Tensor objects that are evaluated before the loop starts,
    and are therefore not part of the loop. This makes sure that the source
    operation is not re-evaluated at each operation.
    """
    def __init__(self, enter_loop, source):
        self.enter_loop = enter_loop
        self.enter_loop.add_loop_input(self)
        self.source = source

    def compute(self, context):
        self.enter_loop.evaluate()
        value = self.source.evaluate(context)
        context[self] = value
        return value


class RecurrenceRelation(Tensor):
    """
    Special kind of loop input that gets an initial value at the first operation
    and gets a new value for each iteration.
    """
    # TODO: Remove self from context at loop end?

    def __init__(self, enter_loop, initial):
        self.enter_loop = enter_loop
        self.enter_loop.add_recurrence_relation(self)
        self.initial = initial

    def fetch_initial(self, context):
        context[self] = self.initial.evaluate(context)

    def update(self, context, value):
        context[self] = value

    def evaluate(self, context):
        if not self in context:
            self.enter_loop.evaluate(context)
            assert self in context
        return context[self]


class EnterLoop(Operation):
    def __init__(self):
        super().__init__([], None)
        self.recurrence_relations = []
        self.loop_inputs = []

    # NOTE: May not be useful at all
    def add_recurrence_relation(self, node):
        self.recurrence_relations.append(node)

    def add_loop_input(self, node):
        self.loop_inputs.append(node)

    def compute(self, context):
        # Only called in the first iteration.
        return None


class BooleanOperation(Operation):
    """Class of operations that evaluates to a single boolean"""
    pass


class NextIteration(Operation):
    def __init__(self, enter_loop, recurrence_relations):
        """
        enter_loop: The EnterLoop operation in the same loop.
        recurrence_relations: A dictionary of:
            key: A RecurrenceRelation object that is input to the same loop.
            value: A node that computes the next input to the loop.
        """
        inputs = set(recurrence_relations.values())
        super().__init__(inputs, None)
        self.enter_loop = enter_loop
        assert inputs == enter_loop.loop_inputs, "ERROR: NextIteration object did " \
            + "not get a node that computes a new value for all recurrence relations"
        self.recurrence_relations = recurrence_relations

    def compute(self, context):
        """
        Returns a dictionary where:
            key: A RecurrenceRelation object.
            value: The new value for the RecurrenceRelation object.
        """
        return {
            key: val.evaluate(context) for (key, val) \
            in self.recurrence_relations.items()
        }


class ExitLoop(Operation):
    def __init__(self, next_iteration, loop_end_condition, outputs):
        """
        next_iteration: The NextIteration operation in the same loop.
        loop_end_condition: A BooleanOperation that tells if the loop
        should end or not.
        outputs: A list of nodes in the loop that this ExitLoop operation
        is able to output using a LoopOutput operation.
        """
        super().__init__(outputs, None)
        self.next_iteration = next_iteration
        self.enter_loop = next_iteration.enter_loop
        self.loop_end_condition = loop_end_condition
        self.outputs = outputs

        # Search for all nodes in the loop
        all_loop_nodes = self.find_loop_nodes()

        # Split set of nodes into groups that are handled differently
        # TODO: check that recurrences and loop_nodes fit the content of EnterLoop.
        self.loop_inputs = {x for in all_loop_nodes if isinstance(x, LoopInput)}
        self.recurrences = {x for in all_loop_nodes if isinstance(x, RecurrenceRelation)}
        self.loop_nodes = (all_loop_nodes - self.loop_inputs) - self.recurrences

    def find_loop_nodes(self):
        """
        Searches through all nodes that are input to this ExitLoop operation, or
        to the NextIteration operation, and the loop-end-condition operation.
        These nodes are used to clear the cached internal state of the loop between
        iterations to make sure that operations are re-computed with the new
        input that is provided by recurrence relations or new output from
        stateful operations (Eg. FIFO-Queues or variables that have been updated).
        """

        # The set of nodes in the loop are added to this set
        settled_nodes = set()

        # A temporary set containing nodes in the loop.
        # Nodes are moved from this set to settled_nodes
        # when their parent/input nodes are added to unsettled_nodes.
        unsettled_nodes = self.inputs
        unsettled_nodes += self.next_iteration.inputs
        unsettled_nodes.append(self.loop_end_condition)
        unsettled_nodes = set(unsettled_nodes)

        while len(unsettled_nodes) > 0:
            next_unsettled = set()
            for node in unsettled_nodes:

                # Check if the node's parents/inputs should be added to the set.
                # Eg. The LoopInput and RecurrenceRelation operations are the interface
                # from the outside to the inside of the loop, so their parents do not
                # belong in this set.
                # Similarly, it can find the ExitLoop operation of a nested loop. While
                # the outputs of the nested loop (which are the inputs to the nested
                # ExitLoop operation) are part of this (outer) loop, the nested
                # loop is responsible for cleaning up its own internal state,
                # so these nodes are not added to the set.
                if isinstance(node, RecurrenceRelation):
                    # TODO: Assert that it belongs to the right loop.
                    check_parents = False
                elif isinstance(node, LoopInput):
                    check_parents = False
                elif isinstance(node, EnterLoop):
                    assert False, "WARNING: Nodes cannot have EnterLoop as input"
                elif isinstance(node, NextIteration):
                    assert False, "WARNING: Nodes cannot have NextIteration as input"
                elif isinstance(node, ExitLoop):
                    check_parents = False
                elif isinstance(node, LoopOutput):
                    check_parents = True
                elif isinstance(node, Operation):
                    check_parents = True
                elif isinstance(node, Tensor):
                    check_parents = False
                else:
                    assert False, f"ExitLoop: Found node in loop-graph that is is not" \
                            + " a tensor subclass: {node}"

                # Add this node to the set of settled nodes, and add its parents/
                # inputs to the new set of nodes to search.
                settled_nodes.add(node)
                if check_parents:
                    next_unsettled += set(node.inputs)

            # Initialize the next iteration by updating unsettled_nodes.
            # The nested loop above may add nodes to next_unsettled that
            # may already be in settled_nodes, so no need to repeat the
            # procedure for those.
            unsettled_nodes = next_unsettled - settled_nodes

        return settled_nodes

    def clear_context_cache(self, context, nodes):
        for node in nodes:
            if node in context:
                del context[node]

    def compute(self, context):
        """
        Returns a dictionary where:
            key: An element in self.outputs
            value: The value returned by the element in self.outputs
        """
        while True:
            do_new_iteration = self.loop_end_condition.evaluate(context)
            if do_new_iteration:
                # Compute the input to the next iteration
                new_input_vals = self.next_iteration.evaluate(context)

                # Clear cached computations within the loop
                self.clear_context_cache(context, self.loop_nodes)

                # Set the new state of recurrence relations
                for node, val in new_input_vals.items():
                    node.update(context, val)

            else:
                # Evaluate the nodes that are returned from the loop
                loop_outputs = {node: node.evaluate(context)}

                # Clean up internal state and cached computations in the loop
                # that are no longer useful.
                # (loop-inputs, recurrence relations and internal operations)
                for node_set in [self.loop_nodes, self.recurrences, self.loop_inputs]:
                    self.clear_context_cache(context, node_set)

                # Return results
                return loop_outputs


class LoopOutput(Operation):
    """
    A special kind of operation that enables loops to produce output that can be
    accessed after the loop has finished.
    Operations outside the loop that only needs the value after the loop has terminated should
    access it through the LoopOutput operation to not be included in the loop.
    """
    def __init__(self, exit_loop, output):
        """
        exit_loop: The ExitLoop operation of the given loop.
        output: An operation within exit_loop.outputs. LoopOut.evaluate
        returns the same value as output.evaluate.
        """
        self.exit_loop = exit_loop
        self.key = key
        shape = exit_loop.outputs[key].shape
        super().__init__([exit_loop], shape)

    def compute(self, context):
        output_vals = self.exit_loop.evaluate()
        return output_vals[self.key]

