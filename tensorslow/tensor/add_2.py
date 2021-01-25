from .core import AddN


class Add2(AddN):
    """Adds two nodes of same shape"""

    def __init__(self, in_a, in_b):
        super().__init__([in_a, in_b])
