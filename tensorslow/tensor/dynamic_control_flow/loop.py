
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


class LoopEndCondition(Operation):
    def compute(self, context):
        """
        Returns a boolean:
            True --> Finish the iteration and start again
            False --> Go to ExitLoop
        """
        raise NotImplementedError


class NextIteration(Operation):
    def __init__(self, enter_loop, recurrence_relations):
        """
        enter_loop: The EnterLoop operation in the same loop.
        recurrence_relations: A dictionary of:
            key: A RecurrenceRelation object that is input to the same loop.
            value: A node that computes the next input to the loop.
        """
        # Check that recurrence_relations includes all from enter_loop.
        self.recurrence_relations = recurrence_relations
        inputs = set(recurrence_relations.values())
        super().__init__(inputs, None)

    def compute(self, context):
        """
        Returns a dictionary where:
            key: A RecurrenceRelation object.
            value: The new value for the RecurrenceRelation object.
        """
        raise NotImplementedError


class ExitLoop(Operation):
    def __init__(self, next_iteration, loop_end_condition, outputs):
        """
        next_iteration: The NextIteration operation in the same loop.
        loop_end_condition: A BooleanOperation that tells if the loop
        should end or not.
        outputs: A list of nodes in the loop that this ExitLoop operation
        is able to output using a LoopOutput operation.
        """
        self.next_iteration = next_iteration
        self.loop_end_condition = loop_end_condition
        self.outputs = outputs

        # Search for all nodes in the loop, except for 
        # LoopInput and RecurrenceRelation nodes.
        root_loop_nodes = self.outputs + [loop_end_condition] \
                    + next_iteration.recurrence_relations

        loop_nodes = []
        for node in root_loop_nodes:
        super().__init__(outputs, None)

    def compute(self, context):
        """
        Returns a dictionary where:
            key: An element in self.outputs
            value: The value returned by the element in self.outputs
        """
        while True:
            new_iteration = self.loop_end_condition.evaluate(context)
            if new_iteration:
                new_input_vals = self.next_iteration.evaluate(context)




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

