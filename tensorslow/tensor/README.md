# tensorslow/tensor

## Short description of the different folders and files
* core.py: Defines the Tensor, Operation, BackPropOperation classes + some more. All classes within tensorslow/tensor inherit from these classes
* dynamic_control_flow: Mostly experimental. Enables having loops / recurrence relations as a part of a computational graph. May also introduce conditional branching in some form in the future.
* stateful_operations: Just meant to be used in loops. So far only includes FIFO queues which can be pushed and popped from.
* root_nodes: Defines some special Tensor subclasses that don't depend on other Tensor objects as input (in other words, all Tensor subclasses except for Operations).
* The rest of the folders don't have a central role in core system, but includes quite common mathematical operations like matrix multiplication, and different kinds of activation functions and cost functions.

