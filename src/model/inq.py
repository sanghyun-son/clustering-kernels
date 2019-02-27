from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

bit_width = 5
def make_model(args, parent=False):
    bit_width = args.inq_bits
    module = import_module('model.' + args.base.lower())
    class INQ(getattr(module, args.base)):
        def __init__(self, args):
            super(INQ, self).__init__(args, conv3x3=inq_conv, conv1x1=inq_conv)

    return INQ(args)

def inq_conv(
    in_channels, out_channels, kernel_size,
    stride=1, padding=None, bias=True):

    if padding is None:
        padding = kernel_size // 2

    return INQConv(
        in_channels, out_channels, kernel_size,
        padding=padding, stride=stride, bias=bias
    )

class INQConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, bias=True):

        super(INQConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.kwargs = {
            'stride': (stride, stride),
            'padding': (padding, padding)
        }
        self.n1, self.n2 = None, None
        nn.modules.conv._ConvNd.reset_parameters(self)
    
    def __repr__(self):
        s = 'INQ-{}(2^{}~2^{}) convolution({}, {}, kernel_size={}, stride={}, padding={})'.format(
                bit_width,
                self.n2,
                self.n1,
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.kwargs['stride'],
                self.kwargs['padding']
            )

        return s

    def forward(self, x):
        output = F.conv2d(x, self.weight, bias=self.bias, **self.kwargs)

        return output

    def quantize(self, percent, debug=False):
        with torch.no_grad():
            _, idx = self.weight.data.view(-1).abs().topk(
                int(percent * self.weight.nelement()), largest=False
            )
            selected_abs = self.weight.data.view(-1).abs()[idx]
            selected_sign = self.weight.data.view(-1).sign()[idx]
            quantized = selected_abs.clamp(1e-8).mul(4 / 3).log2().floor()
            quantized_pow = torch.pow(2, quantized)
            quantized_pow.clamp_(min=2**self.n2, max=2**self.n1)
            keep = selected_abs.ge(2**(self.n2 - 1)).float()
            selected = selected_sign * keep * quantized_pow
            self.weight.data.view(-1).scatter_(0, idx, selected)
            self.weight.grad_mask = self.weight.new_ones(self.weight.size())
            self.weight.grad_mask.view(-1)[idx] = 0

    def set_range(self):
        s = self.weight.abs().max()
        self.n1 = int(s.mul(4/3).log2().floor().item())
        self.n2 = int(self.n1 + 1 - 2**(bit_width - 2))

