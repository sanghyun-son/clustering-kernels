import math

import torch
import torch.nn as nn

def default_conv(
        in_channels, out_channels, kernel_size, stride=1, bias=True):

    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias
    )

def nopad_conv(
        in_channels, out_channels, kernel_size, stride=1, bias=True):

    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=0, stride=stride, bias=bias)

def default_linear(in_channels, out_channels, bias=True):
    return nn.Linear(in_channels, out_channels, bias=bias)

def default_norm(in_channels):
    return nn.BatchNorm2d(in_channels)

def default_act():
    return nn.ReLU(True)

def init_vgg(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            c_out, _, kh, kw = m.weight.size()
            n = kh * kw * c_out
            m.weight.data.normal_(0, math.sqrt(2 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

def init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            _, c_in, kh, kw = m.weight.size()
            n = kh * kw * c_in
            m.weight.data.normal_(0, math.sqrt(2 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            _, c_in = m.weight.size()
            m.weight.data.normal_(0, math.sqrt(1 / c_in))
            m.bias.data.zero_()

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=True,
        conv3x3=default_conv, norm=default_norm, act=default_act):

        modules = []
        modules.append(conv3x3(
            in_channels, out_channels, kernel_size, stride=stride, bias=bias
        ))
        if norm is not None: modules.append(norm(out_channels))
        if act is not None: modules.append(act())

        super(BasicBlock, self).__init__(*modules)
