
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
#   -dependency search. Here it can be useful for them to be tagged.
# "Hide" methods that are not meant to be globally available
# More intuitive/ easy to use scripting interface

# LoopTagged: LoopInput, RecRel, and RecRelOut should be flagged as members of the same loop
# Not tagged: LoopOut and EndLoop should not be tagged, but if they are nested
#             then they should be tagged by the outer loop.
#             They should be able to notice the membership to a nested loop if the input to
#             LoopInput or RecRel belongs to another loop. In that case they should all
#             belong to the same outer loop.
# EnterLoop: This is the looptag itself?

# Making loop tags convenient:
# Operations get their loop tag from their parent.
# Tensors without inputs (not operations, but constants, variables and queues)
#   are not allowed in loops, and can therefore have a hard coded "None" tag.
# Therefore we have an easily enforced scheme where LoopInput and RecRel
#   operations get their loop-tag as a parameter, and all "normal" operations
#   in the loop "inherit" their loop-tag from all parents (where all of them
#   need to have the same tag).
# Since all "normal" operations/nodes in the loop have a loop-tag it is easy
#   for LoopOutput and RecRelOut nodes to check that a node is a member of 
#   the loop, and it's also simple for EndLoop to check that the condition
#   node is a member of the loop.

# The different loop operations and their loop tags.
# EnterLoop: Get the outer loop tag from RecRel nodes.
# RecurrenceRelation and LoopInput: The corresponding EnterLoop, so dependent nodes
#   can easily inherit it.
# RecurrenceRelationOut: Don't care?
# EndLoop: Don't really care, but use the outer loop.
# LoopOutput: The tag of the outer loop, so dependent nodes can easily inherit it.

# Additional notes:
# LoopOutput nodes can find their loop tag by checking RecRel/EndLoop nodes.
# RecRel nodes should all have the same "outer-loop-tag".
#   Enforce this by setting EnterLoop's outer loop tag when the first
#   RecRel is initialized, and make sure the others have the same.
# 



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


class Loop(Operation):
    def __init__(self):
        super().__init__([], None, find_loop_tag=False)
        self.loop_input_map = {}
        self.loop_inputs = set()
        self.rec_rels = set()
        self.rec_rel_outs = []
        self.operations = set()
        self.loop_end_condition = None
        self.loop_outputs = {}
        self.shadow_dependencies = set()
        self.is_activated = False

        # Mark this as False since it's determined by the RecurrenceRelation
        # and LoopInput nodes that are added at a later point.
        # This variable can be set to None at a later stage but we need a way
        # to distinguish between "not set" and "None".
        self.loop_tag = False

    def input(self, source):
        # If already created return the old LoopInput
        if source in self.loop_input_map:
            return self.loop_input_map[source]

        # Create, store, and return a new LoopInput for the given source
        self._set_loop_tag(source.get_loop_tag())
        loop_input = LoopInput(self, source)
        self.loop_inputs.add(loop_input)
        self.loop_input_map[source] = loop_input
        return loop_input

    def recurrence_relation(self, initial_source):
        # Create, store, and return a new RecurrenceRelation for the given source
        self._set_loop_tag(initial_source.get_loop_tag())
        rec_rel = RecurrenceRelation(initial_source, self)
        self.rec_rels.add(rec_rel)
        return rec_rel

    def output(self, source):
        assert source.get_loop_tag() is self, "Loop.output expects " +\
                            "'source' to be a node within the loop."

        if not source in self.loop_outputs:
            output = LoopOutput(source, self)
            self.loop_outputs[source] = output
            return output
        else:
            return self.loop_outputs[source]

    def set_ending_condition(self, condition):
        self.loop_end_condition = condition
        assert condition.get_loop_tag() is self, "A loop's ending condition must " +\
                                                        "be a member of the loop."

    def add_shadow_dependency(self, node):
        self.shadow_dependencies.add(node)
        assert node.get_loop_tag() is self, "Loop cannot have a shadow-dependency " +\
                                            "that is contained in another loop."

    def _add_operation(self, node):
        self.operations.add(node)

    def _set_loop_tag(self, tag):
        if self.loop_tag == False:
            self.loop_tag = tag
        else:
            assert self.loop_tag == tag, "All inputs to a loop must come \
                                                from the same outer loop."

    def _assert_has_all_recrel_nexts(self):
        # Make sure the entire set of rec_rel_outs is known.
        if len(self.rec_rels) > len(self.rec_rel_outs):
            self.rec_rel_outs = [
                rec_rel.rec_rel_out
                for rec_rel in self.rec_rels
                if rec_rel.rec_rel_out is not None
            ]

        assert len(self.rec_rels) == len(self.rec_rel_outs), "Tried to evaluate " +\
                "loop before all recurrence relations have been given a source " +\
                "node to compute their value in the next iteration."

    def _assert_has_loop_end_condition(self):
        assert self.loop_end_condition is not None, "Tried to evaluate loop " +\
                                "before it was given a loop end condition."

    def _assert_activated(self):
        assert self.is_activated

    def _activate(self):
        self.is_activated = True

    def _deactivate(self):
        self.is_activated = False

    def _clear_context_cache(self, context, nodes):
        for node in nodes:
            if node in context:
                del context[node]

    def compute(self, context):
        """
        Returns a dictionary where:
            key: An element in self.outputs
            value: The value returned by the element in self.outputs
        """

        self._assert_has_all_recrel_nexts()
        self._assert_has_loop_end_condition()

        # "Activate" the loop to signalize that the nodes within the loop are
        # evaluated through the Loop node.
        self._activate()

        loop_nodes = self.operations
        rec_rels = self.rec_rels
        while True:
            do_new_iteration = self.loop_end_condition.evaluate(context)
            if do_new_iteration:

                # Compute the next value for the recurrence relations
                # Store results as (RecRelOut, val) tuples
                next_vals = []
                for rec_rel in rec_rels:
                    val = rec_rel.rec_rel_out.evaluate(context)
                    next_vals.append((rec_rel, val))

                # Execute shadow dependencies
                for node in self.shadow_dependencies:
                    node.evaluate(context)

                # Clear cached computations within the loop
                self._clear_context_cache(context, loop_nodes)
                self._clear_context_cache(context, self.rec_rel_outs)

                # Set the new state of recurrence relations
                for rec_rel, val in next_vals:
                    rec_rel._update(context, val)

            else:
                # Evaluate the nodes that are returned from the loop
                loop_outputs = {
                    loop_output: source.evaluate(context) \
                    for (source, loop_output) in self.loop_outputs.items()
                }

                # Clean up internal state and cached computations in the loop
                # that are no longer useful.
                node_lists = [
                    loop_nodes,
                    rec_rels,
                    self.loop_inputs,
                ]

                for node_set in node_lists:
                    self._clear_context_cache(context, node_set)

                # "Deactivate" the loop
                self._deactivate()

                # Return results
                return loop_outputs


class RecurrenceRelation(Operation):
    """
    Special kind of loop input that gets an initial value at the first operation
    and gets a new value for each iteration.
    """

    def __init__(self, initial, loop):
        super().__init__([initial], initial.shape, find_loop_tag=False)
        self.loop_tag = loop
        self.rec_rel_out = None
        self.initial = initial
        assert type(loop) is Loop, f"Expected loop to be of type " + \
                                    f"Loop. Got {type(loop)}"
        self.loop = loop

    def next_iteration(self, source):
        assert self.rec_rel_out is None, "RecurrenceRelation's next_iteration " +\
                                        "source has already been set"

        assert source.get_loop_tag() is self.get_loop_tag(), "Expected source " +\
                "to be a node within the same loop as the RecurrenceRelation node"

        self.rec_rel_out = RecurrenceRelationOut(self, source)

    def _update(self, context, value):
        context[self] = value

    def compute(self, context):
        self.loop._assert_activated()
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

    def __init__(self, loop, source):
        super().__init__([source], source.shape, find_loop_tag=False)
        self.loop_tag = loop
        self.source = source
        self.loop = loop

    def compute(self, context):
        self.loop._assert_activated()
        return self.source.evaluate(context)


class LoopOutput(Operation):
    """
    A special kind of operation that enables loops to produce output that can be
    accessed after the loop has finished.
    Operations outside the loop that only needs the value after the loop has terminated should
    access it through the LoopOutput operation to not be included in the loop.
    """

    # TODO: THis and EndLoop should make sure to not mark themselves as loop members
    # This one needs super().__init__(inputs, shape), which notifies the source
    # node that it depends on this one. However, this results in calling
    # self.notify_loop_membership with the source's loop-tag, which we don't want since
    # LoopOutput objects are not meant to be tagged as loop members (unless nested), since
    # that would make Operations that use a LoopOutput as input "believe" they are also
    # part of the loop.
    def __init__(self, source, loop):
        """
        source: A node within the loop. The LoopOutput returns the same value as source.
        """

        super().__init__([source], source.shape, find_loop_tag=False)
        self.source = source
        self.loop = loop
        self.loop_tag = loop.loop_tag

    def add_dependent_node(self, node):
        super().add_dependent_node(self, node)
        print("TODO: Make this function check that the dependent node is not in the same loop.")

    def compute(self, context):
        outputs = self.loop.evaluate(context)
        return outputs[self]

