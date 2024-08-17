
from torch.autograd import Function


class ReverseLayerF(Function):
    """The gradient reversal layer (GRL)

    This is defined in the DANN paper http://jmlr.org/papers/volume17/15-239/15-239.pdf

    Forward pass: identity transformation.
    Backward propagation: flip the sign of the gradient.

    From https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/models/layers.py
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
