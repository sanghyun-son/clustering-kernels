from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    module = import_module('model.' + args.base.lower())
    
    class BWN(getattr(module, args.base)):
        def __init__(self, args):
            super(BWN, self).__init__(args, conv=binary_conv)

    return BWN(args)

def binary_conv(
    in_channels, out_channels, kernel_size,
    stride=1, padding=None, bias=False):

    if padding is None:
        padding = kernel_size // 2

    return BinaryConv(
        in_channels, out_channels, kernel_size,
        padding=padding, stride=stride
    )

class Binarize(torch.autograd.Function):
    '''
        Differentiable Sign function
    '''
    @staticmethod
    def forward(ctx, weight):
        ctx.save_for_backward(weight)
        output = weight.sign()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_variables
        slope = torch.eq(weight.gt(-1), weight.lt(1)).float()
        grad_weight = grad_output.mul(slope)

        return grad_weight

class BinaryConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BinaryConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = None
        self.kwargs = {
            'stride': (stride, stride),
            'padding': (padding, padding)
        }
        nn.modules.conv._ConvNd.reset_parameters(self)
    
    def __repr__(self):
        s = 'Binary Convolution({}, {}, kernel_size={}, stride={}, padding={})'.format(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.kwargs['stride'],
                self.kwargs['padding']
            )

        return s

    def forward(self, x):
        a = self.weight.abs()
        a = a.mean(dim=1, keepdim=True)
        a = a.mean(dim=2, keepdim=True)
        a = a.mean(dim=3, keepdim=True)
        b = Binarize.apply(self.weight)
        binary_weight = b.mul(a)

        output = F.conv2d(x, binary_weight, **self.kwargs)

        return output

