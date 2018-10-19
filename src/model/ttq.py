from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    module = import_module('model.' + args.base.lower())
    
    class TTQ(getattr(module, args.base)):
        def __init__(self, args):
            super(TTQ, self).__init__(args, conv=trained_ternary_conv)

    return TTQ(args)

def trained_ternary_conv(
    in_channels, out_channels, kernel_size,
    stride=1, padding=None, bias=False):

    if padding is None:
        padding = kernel_size // 2

    return TrainedTernaryConv(
        in_channels, out_channels, kernel_size,
        padding=padding, stride=stride
    )

class TrainedTernarize(torch.autograd.Function):
    t_delta = 0.05
    @staticmethod
    def forward(ctx, weight, wp, wn):
        ctx.save_for_backward(weight, wp, wn)
        threshold = TrainedTernarize.t_delta * weight.abs().max()
        gt = weight.gt(threshold)
        lt = weight.lt(-threshold)
        output = gt.float().mul(wp) - lt.float().mul(wn)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight, wp, wn = ctx.saved_variables
        threshold = TrainedTernarize.t_delta * weight.abs().max()
        gt = weight.gt(threshold)
        lt = weight.lt(-threshold)
        btwn = torch.eq(gt, lt)
        grad_wp = grad_output.masked_select(gt).sum()
        grad_wn = -grad_output.masked_select(lt).sum()
        grad_mask = gt.float().mul(wp) + lt.float().mul(wn) + btwn.float()
        grad_weight = grad_output.mul(grad_mask)

        return grad_weight, grad_wp, grad_wn

class TrainedTernaryConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(TrainedTernaryConv, self).__init__()

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
        w_initial = self.weight.abs().mean().data[0]
        self.wp = nn.Parameter(torch.Tensor(1).fill_(w_initial))
        self.wn = nn.Parameter(torch.Tensor(1).fill_(w_initial))
    
    def __repr__(self):
        s = 'Trained Ternary Convolution({}, {}, kernel_size={}, stride={}, padding={})'.format(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.kwargs['stride'],
                self.kwargs['padding']
            )

        return s

    def forward(self, x):
        ternary_weight = TrainedTernarize.apply(self.weight, self.wp, self.wn)
        output = F.conv2d(x, ternary_weight, **self.kwargs)

        return output

