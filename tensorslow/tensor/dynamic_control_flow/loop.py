
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


class EnterLoop(Operation):
    """Does not really have Operation semantics at all????"""

    def __init__(self):
        super().__init__([], None, find_loop_tag=False)
        self.loop_tag = self
        self.loop_input_map = {}
        self.loop_inputs = set()
        self.rec_rels = set()
        self.operations = set()
        self.end_loop = None

        # Mark this as False since it's determined by the RecurrenceRelation
        # and LoopInput nodes that are added at a later point.
        # This variable can be set to None at a later stage but we need a way
        # to distinguish between "not set" and "None".
        self.outer_loop_tag = False

    def input(self, source):
        # If already created return the old LoopInput
        if source in self.loop_input_map:
            return self.loop_input_map[source]

        # Create, store, and return a new LoopInput for the given source
        self.set_outer_loop_tag(source.get_loop_tag())
        loop_input = LoopInput(self, source)
        self.loop_inputs.add(loop_input)
        self.loop_input_map[source] = loop_input
        return loop_input

    def recurrence_relation(self, initial_source):
        assert self.end_loop is None, "ERROR: Added a RecurrenceRelation to a loop \
                that has already defined the fixed set of recurrence relations."

        # Create, store, and return a new RecurrenceRelation for the given source
        self.set_outer_loop_tag(initial_source.get_loop_tag())
        rec_rel = RecurrenceRelation(initial_source, self)
        self.rec_rels.add(rec_rel)
        return rec_rel

    def add_operation(self, node):
        self.operations.add(node)

    def set_outer_loop_tag(self, tag):
        if self.outer_loop_tag == False:
            self.outer_loop_tag = tag
        else:
            assert self.outer_loop_tag == tag, "All inputs to a loop must come \
                                                from the same outer loop."

    def set_end_loop(self, end_loop):
        assert type(end_loop) is EndLoop
        assert self.end_loop is None, "EnterLoop object already has a corresponding EndLoop object"
        self.end_loop = end_loop

    def activate(self, context):
        context[self] = 'ACTIVATED'

    def deactivate(self, context):
        del context[self]

    def evaluate(self, context):
        assert context[self] == 'ACTIVATED', "A node within a loop was evaluated \
                    directly by a user without going through a LoopOutput node"


class RecurrenceRelation(Operation):
    """
    Special kind of loop input that gets an initial value at the first operation
    and gets a new value for each iteration.
    """

    def __init__(self, initial, enter_loop):
        super().__init__([initial], initial.shape, find_loop_tag=False)
        self.loop_tag = enter_loop
        self.rec_rel_out = None
        self.initial = initial
        assert type(enter_loop) is EnterLoop, f"Expected enter_loop to be of type " + \
                                                f"EnterLoop. Got {type(enter_loop)}"
        self.enter_loop = enter_loop

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
        assert self.loop_tag is rec_rel.loop_tag, "RecurrenceRelationOut node was \
                constructed where the source and the corresponding RecurrenceRelation \
                node are members of different loops."

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
        super().__init__([source], source.shape, find_loop_tag=False)
        self.loop_tag = enter_loop
        self.source = source
        self.enter_loop = enter_loop

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

    # TODO: THis and EndLoop should make sure to not mark themselves as loop members
    # This one needs super().__init__(inputs, shape), which notifies the source
    # node that it depends on this one. However, this results in calling
    # self.notify_loop_membership with the source's loop-tag, which we don't want since
    # LoopOutput objects are not meant to be tagged as loop members (unless nested), since
    # that would make Operations that use a LoopOutput as input "believe" they are also
    # part of the loop.
    def __init__(self, source, end_loop):
        """
        source: A node within the loop. The LoopOutput returns the same value as source.
        """

        super().__init__([source], source.shape, find_loop_tag=False)
        self.loop_tag = end_loop.loop_tag
        self.source = source
        self.end_loop = end_loop

    def add_dependent_node(self, node):
        super().add_dependent_node(self, node)
        print("TODO: Make this function check that the dependent node is not in the same loop.")

    def compute(self, context):
        outputs = self.end_loop.evaluate(context)
        return outputs[self]


class EndLoop(Operation):
    def __init__(self, enter_loop, rec_rel_outs, loop_end_condition):
        """
        enter_loop: The EnterLoop object that starts the loop.
        rec_rel_outs: A list of RecurrenceRelationOut objects that belong in the loop
        loop_end_condition: A BooleanOperation that tells if the loop
        should end or not.
        """

        # Check that everything corresponds to the right loop.
        assert loop_end_condition.get_loop_tag() is enter_loop
        for node in rec_rel_outs:
            assert node.get_loop_tag() is enter_loop

        # Initialize
        super().__init__([], None, find_loop_tag=False)
        self.loop_tag = enter_loop.outer_loop_tag
        self.enter_loop = enter_loop
        enter_loop.set_end_loop(self)
        self.loop_end_condition = loop_end_condition
        self.rec_rel_outs = rec_rel_outs
        self.shadow_dependencies = set()
        self.loop_outputs = {}

    def output(self, source):
        assert source.get_loop_tag() is self.enter_loop, "EndLoop.output exects " +\
                                            "'source' to be a node within the loop."

        if not source in self.loop_outputs:
            self.loop_outputs[source] = LoopOutput(source, self)

        return self.loop_outputs[source]

    def add_shadow_dependency(self, node):
        self.shadow_dependencies.add(node)
        assert node.get_loop_tag() is self.enter_loop, "EndLoop cannot depend on a node \
                                                that is contained in another loop."

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

        # "Activate" the loop to signalize that the nodes within the loop are
        # evaluated through the EnterLoop node.
        self.enter_loop.activate(context)

        loop_nodes = self.enter_loop.operations
        rec_rel_outs = self.rec_rel_outs
        while True:
            do_new_iteration = self.loop_end_condition.evaluate(context)
            if do_new_iteration:

                # Compute the next value for the recurrence relations
                # Store results as (RecRelOut, val) tuples
                next_vals = []
                for rec_rel_out in rec_rel_outs:
                    val = rec_rel_out.evaluate(context)
                    next_vals.append((rec_rel_out, val))

                # Execute shadow dependencies
                for node in self.shadow_dependencies:
                    node.evaluate(context)

                # Clear cached computations within the loop
                self._clear_context_cache(context, loop_nodes)
                self._clear_context_cache(context, rec_rel_outs)

                # Set the new state of recurrence relations
                for rec_rel_out, val in next_vals:
                    rec_rel_out.rec_rel.update(context, val)

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
                    self.enter_loop.rec_rels,
                    self.enter_loop.loop_inputs,
                ]

                for node_set in node_lists:
                    self._clear_context_cache(context, node_set)

                # "Deactivate" the loop
                self.enter_loop.deactivate(context)

                # Return results
                return loop_outputs

