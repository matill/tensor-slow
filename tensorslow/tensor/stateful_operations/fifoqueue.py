import numpy as np
from tensorslow.tensor.core import Operation, Tensor


# NOTE: Execution order of these operations may be reversed with a
# recursive execution engine? Might need to re-do everything later. idk.
# Could maybe solve this by using "shadow dependencies" (TODO: find another
# name for this that doesn't sound super edgy) that force Enqueue and Dequeue
# operations to be qxecuted in the right order?
# TODO: Support strict shapes
class FIFOQueue(Tensor):
    def __init__(self, queue=None):
        super().__init__(None)
        self.queue = [] if queue is None else queue

    def evaluate(self, context):
        return self

    def dequeue(self):
        return self.queue.pop(0)

    def enqueue(self, obj):
        self.append(obj)

    def set(self, queue):
        self.queue = queue

    def drain(self):
        queue = self.queue
        self.queue = []
        return queue

    def get_all(self):
        return self.queue

    def is_empty(self):
        return len(self.queue) == 0


# TODO: Move Queue operations to seperate file
# Create abstract Queue base class
class Dequeue(Operation):
    def __init__(self, queue):
        super().__init__([queue], None)
        self.queue = queue

    def compute(self, context):
        return self.queue.evaluate(context).dequeue()


class Enqueue(Operation):
    def __init__(self, queue, source):
        super().__init__([queue, source], None)
        self.queue = queue
        self.source = source

    def compute(self, context):
        queue = self.queue.evaluate(context)
        source = self.source.evaluate(context)
        queue.enqueue(source)
        return None


class IsQueueEmpty(Operation):
    def __init__(self, queue):
        super().__init__([queue], None)
        self.queue = queue

    def compute(self, context):
        return self.queue.evaluate(context).is_empty()


# class SetQueue(Operation):
#     def __init__(self, queue):
#         super().__init__([queue], None)
#         self.queue = queue

#     def compute(self, context):
#         queue = self.queue.evaluate(context)
