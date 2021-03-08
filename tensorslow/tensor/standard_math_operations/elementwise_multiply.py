import numpy as np
from tensorslow.tensor.core import AssistedBackPropOperation


class ElementwiseMultiply(AssistedBackPropOperation):
    """
    Returns a tensor of equal shape to the input, where elements at the same
    indexes are multiplied
    """

    def __init__(self, inputs):
        if len(inputs) == 0:
            raise Exception("Error. ElementwiseMultiply node with empty list of nodes as input")
        elif len(inputs) == 1:
            print("Warning. ElementwiseMultiply node with a single node as input is redundant")

        shape = self.get_and_assert_common_shape_in_list(inputs)
        super().__init__(inputs, shape)

    def compute(self, context):
        for i, node in enumerate(self.inputs):
            # Evaluate the current node in the list
            val = node.evaluate(context)

            # Update the "product" variable to be the current product.
            # Three cases to consider:
            # 1. If this is the first variable in the list, then the product should
            # currently just be that element
            # 2. If this is the second element, then we just multiply the product
            # with the current value.
            # 3. If this is the third or higher index, there has already been allocated an
            # array that can be reused. This is not done in the i==1 case, because that would
            # modify the first input array, which could potentially be input to another
            # operation as well.
            if i == 0:
                product = val
            elif i == 1:
                product = np.multiply(product, val)
            else:
                np.multiply(product, val, out=product)

        return product

    def get_parents_gradient_assisted(self, parent, self_gradient):
        n_occurences_in_inputs = len([x for x in self.inputs if x is parent])
        assert n_occurences_in_inputs == 1, "Not implemented for other cases"
        product_nodes = [x for x in self.inputs if x is not parent]
        product_nodes.append(self_gradient)
        return ElementwiseMultiply(product_nodes)
