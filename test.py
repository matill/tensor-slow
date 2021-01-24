import numpy as np



def jacobian_multiply(y_gradient, y_jacobian):
    num_axes = len(y_gradient.shape) - 1
    return np.tensordot(y_gradient, y_jacobian, axes=num_axes)

# x = [
#     [[0, 1], [2, 3], [4, 5]],
#     [[6, 7], [8, 9], [10, 11]],
#     [[12, 13], [14, 15], [16, 17]],
#     [[18, 19], [20, 21], [22, 23]],
# ]

# x = np.arange(60.0).reshape(3, 4, 5)
# y = np.arange(120.0).reshape(3, 4, 5, 2, 1)
# print(x.shape)
# print(y.shape)
# z = np.tensordot(x, y, axes=3)
# print(z)

x = np.array([[1, 2], [3, 4]])
y = np.array([[1, 2], [3, 4]])
z = jacobian_multiply(x, y)
print(z)


# x = np.array(x)
# print(x.shape)
# print(x[0, 1].shape)

