
import numpy as np
from tensorslow.tensor.core import Tensor, Operation



# New things to support:
# Remove NextIteration
# Add RecRelNew
# Add new LoopOutputs to a loop after EndLoop is defined
# Add new LoopInputs to a loop after EndLoop is defined
# A way to add shadow dependencies to LoopEnd after LoopEnd is defined
# A way to replace dependencies to a node?
# Get gradient through specific paths?
# Normal nodes with inputs that are in a loop should automatically be flagged
#   as a loop member, and the EndLoop should add them (and all their dependencies)
#   to the set of members (and tag all their dependencies).
# Normal nodes with input from two different loops should raise an exception.
# LoopOut and LoopEnd nodes should not be flagged as loop members, unless nested.
# LoopInput and RecRel nodes should or should be flagged to make it simple
#   for the nodes that depend on them to flag temselves as loop members.
# RecRelOut nodes should or should not be flagged???? I dont think it matters
#   No nodes depend on RecRelOut nodes, so they can only be found in the inverse
#   -dependency search. Here it can be usefull for them to be tagged.



# Workflow:
# 1. Create all RecRel objects
# 2. Create EnterLoop object
# 3:
#   a: Create all RecRelNew objects
#   c: Create loop end condition (BooleanOperation)
#   d: Create some LoopInput objects.
# 4: Create EndLoop (Check dependencies of all RecRelNews, 
    # and the loop-end-condition:
        # * They should be flagged to be contained in THIS loop.
        # * Inverse dependencies also need to be flagged to be contained in the loop.
        # * Terminate the dependency search at RecRel and LoopInput nodes.
        # * When the dependency search finds LoopOut nodes it should continue searching
        #   through the corresponding RecRel and LoopIn nodes.
        # * Terminate the inverse dependency search at RecRelOut, LoopOut.
        # * When the inverse-dependency search finds LoopIn or RecRel nodes it should
        #   continue searching throgh the corresponding LoopOut nodes.
        # * All RecRel nodes should have a single corresponding RecRelNew object,
        #   and that RecRelNew SHOULD BE provided to the LoopEnd.
        # * Only Operation and Constant nodes are allowed in the loop 
        #   (no Variable, Queue, or Input nodes.)
        # * The provided RecRelNews' corresponsing RecRels' should have the right EnterLoop (both ways)
# 5 (optional, repeated):
    # a: Add new LoopOutput to the LoopEnd:
        # * Check this LoopOutput's dependencies as described above
    # b: Add a new LoopInput to the loop.
    # c: Add a new "withinin-loop" node to the loop.
        # * This node needs to be aware of it's membership in the node
        #   without the user manually making it happen (through dependencies)
        # * Either all it's dependencies belong to the loop, or none of them do.
        # * Its inputs must be an Operation sub-class or Constant.
    # d: Add shadow-dependencies to EndLoop. This must be contained in the loop.


class EnterLoop(Operation):
    """Does not really have Operation semantics at all????"""

    def __init__(self):
        super().__init__([], None)
        self.loop_inputs = set()
        self.rec_rels = set()
        self.loop_end = None

    def add_loop_input(self, node):
        assert type(loop_end) is LoopInput
        self.loop_inputs.add(node)

    def add_rec_rel(self, node):
        assert type(node) is RecurrenceRelation
        assert self.loop_end is None, "ERROR: Added a RecurrenceRelation to a loop \
                that has already defined the fixed set of recurrence relations."

        self.rec_rels.add(node)

    def set_loop_end(self, loop_end):
        assert type(loop_end) is LoopEnd
        assert self.loop_end is None, "EnterLoop object already has a corresponding LoopEnd object"
        self.loop_end = loop_end

    def compute(self, context):
        # If this function is called it means the EnterLoop object was not "activated"
        # in the context, which means a node in the loop's "evaluate" function
        # was called directly, and not through a LoopOutput node.
        assert False


class RecurrenceRelation(Operation):
    """
    Special kind of loop input that gets an initial value at the first operation
    and gets a new value for each iteration.
    """

    def __init__(self, initial, enter_loop):
        super().__init__([initial], initial.shape)
        self.rec_rel_out = None
        self.initial = initial
        assert type(enter_loop) is EnterLoop, "Expected enter_loop to be of type EnterLoop"
        self.enter_loop = enter_loop
        enter_loop.add_rec_rel(self)

    def set_rec_rel_out(self, rec_rel_out):
        assert type(rec_rel_out) is RecurrenceRelationOut, \
                    "ERROR: RecurrenceRelation.set_rec_rel_out called with \
                    argument that is not of RecurrenceRelationOut type"

        assert self.rec_rel_out is None, "RecurrenceRelation object was  \
                    added to more than one RecurrenceRelationOut object"

        self.rec_rel_out = rec_rel_out

    def update(self, context, value):
        context[self] = value

    def compute(self, context):
        self.enter_loop.evaluate(context)
        return self.initial.evaluate(context)


class RecurrenceRelationOut(Operation):
    """
    Input to a node of this type becomes the output of the corresponding 
    RecurrenceRelation object in the next loop iteration.
    """

    def __init__(self, rec_rel, source):
        super().__init__([source], source.shape)
        self.source = source
        self.rec_rel = rec_rel
        rec_rel.set_rec_rel_out(self)

    def add_dependent_node(self, node):
        super().add_dependent_node(self, node)
        print("WARNING: A node uses a RecurrenceRelationOut as input")

    def compute(self, context):
        return self.source.evaluate(context)


class LoopInput(Operation):
    """
    Used to access Tensor objects that are evaluated before the loop starts,
    and are therefore not part of the loop. This makes sure that the source
    operation is not re-evaluated at each operation.
    """

    def __init__(self, enter_loop, source):
        super().__init__([source], source.shape)
        self.enter_loop = enter_loop
        self.enter_loop.add_loop_input(self)
        self.source = source

    def compute(self, context):
        self.enter_loop.evaluate(context)
        return self.source.evaluate(context)


class LoopOutput(Operation):
    """
    A special kind of operation that enables loops to produce output that can be
    accessed after the loop has finished.
    Operations outside the loop that only needs the value after the loop has terminated should
    access it through the LoopOutput operation to not be included in the loop.
    """

    def __init__(self, source, exit_loop):
        """
        source: A node within the loop. The LoopOutput returns the same value as source.
        """

        super().__init__([source], source.shape)
        self.source = source
        self.exit_loop = exit_loop
        exit_loop.add_loop_output(self)

    def add_dependent_node(self, node):
        super().add_dependent_node(self, node)
        print("TODO: Make this function check that the dependent node is not in the same loop.")

    def compute(self, context):
        self.exit_loop.evaluate(context)
        return self.source.evaluate(context)


class EndLoop(Operation):
    def __init__(self, enter_loop, rec_rel_outs, loop_end_condition, loop_outputs):
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
        self.loop_inputs = {x for x in all_loop_nodes if isinstance(x, LoopInput)}
        self.recurrences = {x for x in all_loop_nodes if isinstance(x, RecurrenceRelation)}
        self.loop_nodes = (all_loop_nodes - self.loop_inputs) - self.recurrences

        # Assert correctnes of the recurrences and loop_inputs sets
        assert self.recurrences == self.enter_loop.recurrence_relations, \
            "ERROR: ExitLoop was not able to find the same set of recurrence relation \
            objects that are registered in the EnterLoop object."

        assert self.loop_inputs == self.enter_loop.loop_inputs, \
            "ERROR: ExitLoop was not able to find the same set of loop inputs \
            objects that are registered in the EnterLoop object."

    def add_loop_output(self, loop_output):
        raise NotImplementedError

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
        unsettled_nodes = [x for x in self.inputs]
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
                    next_unsettled |= set(node.inputs)

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
                self.clear_context_cache(context, [self.next_iteration])

                # Set the new state of recurrence relations
                for node, val in new_input_vals.items():
                    node.update(context, val)

            else:
                # Evaluate the nodes that are returned from the loop
                loop_outputs = {
                    node: node.evaluate(context) for node in self.outputs
                }

                # Clean up internal state and cached computations in the loop
                # that are no longer useful.
                # (loop-inputs, recurrence relations and internal operations, EnterLoop)
                for node_set in [self.loop_nodes, self.recurrences, \
                                self.loop_inputs, [self.enter_loop]]:
                    self.clear_context_cache(context, node_set)

                # Return results
                return loop_outputs

