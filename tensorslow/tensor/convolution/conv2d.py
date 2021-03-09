import numpy as np
from tensorslow.tensor.core import AssistedBackPropOperation


class Conv2D(AssistedBackPropOperation):
    """
    Applies a convolution to a 2D image with c_in channels. Assumes stride equal to
    the dimensions of the filter and no padding.
    """

    def __init__(self, image, filter):
        """
        image: Tensor object with shape (c_in, h_in, w_in)
        filter: Tensor object with shape (c_out, c_in, f_h, f_w)
        """
        c_in, h_in, w_in = image.shape
        c_out, _c_in, f_h, f_w = filter.shape
        assert _c_in == c_in, "Filter and image must have equal number of input " +\
                            f"channels. Got {c_in} and {_c_in}"

        assert h_in % f_h == 0, "Image height must be divisible by filter height"
        assert w_in % f_w == 0, "Image width must be divisible by filter width"
        h_out = int(h_in / f_h)
        w_out = int(w_in / f_w)

        shape = (c_out, h_out, w_out)
        inputs = [image, filter]
        self.image = image
        self.filter = filter
        super().__init__(inputs, shape)

    def compute(self, context):
        image = self.image.evaluate(context)
        filter = self.filter.evaluate(context)

        c_out, c_in, f_h, f_w = filter.shape
        c_out, h_out, w_out = self.shape
        out_image = np.ndarray(self.shape, dtype=float)
        for i_out in range(h_out):
            i_in = i_out * f_h
            for j_out in range(w_out):
                j_in = j_out * f_w
                in_image_part = image[:,i_in:i_in+f_h, j_in:j_in+f_w]
                out_image[:, i_out, j_out] = np.tensordot(
                    filter,
                    in_image_part,
                    axes=([1,2,3], [0,1,2])
                )

        return out_image

    def get_parents_gradient_assisted(self, parent, self_gradient):
        is_image = parent is self.image
        is_filter = parent is self.filter

        assert is_image != is_filter, "Cannot get the derivative of Conv2D if the " +\
                                    "input is both the filter and the image."

        if is_image:
            return DImageConv2D(self_gradient, self.filter)
        else:
            return DFilterConv2D(self_gradient, self.image)


class DImageConv2D(AssistedBackPropOperation):
    def __init__(self, out_derivative, filter):
        # Find shapes
        c_out, h_out, w_out = out_derivative.shape
        _c_out, c_in, f_h, f_w = filter.shape
        h_in = h_out * f_h
        w_in = w_out * f_w

        # 
        inputs = [out_derivative, filter]
        shape = (c_in, h_in, w_in)
        self.out_derivative = out_derivative
        self.filter = filter
        super().__init__(inputs, shape)

    def compute(self, context):
        out_derivative = self.out_derivative.evaluate(context)
        filter = self.filter.evaluate(context)

        c_out, h_out, w_out = out_derivative.shape
        c_out, c_in, f_h, f_w = filter.shape
        self_derivative = np.ndarray(self.shape, dtype=float)
        for i_out in range(h_out):
            i_in = i_out * f_h
            for j_out in range(w_out):
                j_in = j_out * f_w
                self_derivative_part = self_derivative[:,i_in:i_in+f_h, j_in:j_in+f_w]
                out_derivative = out_derivative[:, i_out, j_out]
                np.tensordot(
                    out_derivative,
                    filter,
                    axes=([0], [0]),
                    out=self_derivative_part
                )

        return self_derivative


class DFilterConv2D(AssistedBackPropOperation):
    def __init__(self, out_derivative, in_image):
        # Find shapes
        c_out, h_out, w_out = out_derivative.shape
        c_in, h_in, w_in = in_image.shape
        f_h = int(h_in / h_out)
        f_w = int(w_in / w_out)

        inputs = [out_derivative, in_image]
        shape = (c_out, c_in, f_h, f_w)
        self.out_derivative = out_derivative
        self.in_image = in_image
        super().__init__(inputs, shape)

    def compute(self, context):
        out_derivative = self.out_derivative.evaluate(context)
        in_image = self.in_image.evaluate(context)

        c_out, h_out, w_out = out_derivative.shape
        c_out, c_in, f_h, f_w = self.shape
        self_derivative = np.zeros(self.shape, dtype=float)
        buff = np.ndarray(self.shape, dtype=float)
        for i_out in range(h_out):
            i_in = i_out * f_h
            for j_out in range(w_out):
                j_in = j_out * f_w
                in_image_part = in_image[:,i_in:i_in+f_h, j_in:j_in+f_w]
                out_derivative_part = out_derivative[:, i_out, j_out]
                self_derivative += np.tensordot(
                    out_derivative_part,
                    in_image_part,
                    axes=([], []),
                    out=buff
                )

        return self_derivative
