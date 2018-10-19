from model import clustering

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import qnn

def cluster_conv(
    in_channels, out_channels, kernel_size, stride=1, bias=True):
    
    return CConv()

def gen_cconv(target):
    return (m for m in target.modules() if isinstance(m, CConv))

class CConv(nn.Module):
    n_bits = 16
    def __init__(self):
        super(CConv, self).__init__()
        self.k = -1
        self.channels = ('-', '-')

    def __repr__(self):
        s = '{}-{}-{}: {} -> {}'.format(
            self.__class__.__name__, self.k, CConv.n_bits, *self.channels
        )
        return s

    def set_params(
        self, source,
        centroids=None, idx=None, scales=None, debug=False):

        self.centroids = centroids
        self.k, self.kh, self.kw = centroids.size()
        self.channels = (source.in_channels, source.out_channels)
        self.kwargs = {'stride': source.stride, 'padding': source.padding}

        if hasattr(source, 'bias'):
            self.register_parameter('bias', source.bias)

        if hasattr(source, 'cut'):
            self.register_buffer('cut', source.cut)

        self.register_buffer('idx', idx)
        self.register_parameter('scales', scales)

        self.debug = debug
        self.flops = 0

    def get_parameters(self):
        weight = self.centroids[self.idx]
        weight = weight.view(*self.channels[::-1], self.kh, self.kw)
        weight = weight * self.get_scales()
        if hasattr(self, 'cut'): weight = weight * self.cut
        bias = getattr(self, 'bias', None)

        return weight, bias

    def get_scales(self):
        if CConv.n_bits == 16:
            scales = self.scales.half().float()
        elif CConv.n_bits < 16:
            scales = qnn.QuantizeParams.apply(self.scales)

        return scales

    def forward(self, x):
        if self.debug: self.calc_flops(x)

        weight, bias = self.get_parameters()
        x = F.conv2d(x, weight, bias, **self.kwargs)

        return x

    def calc_flops(self, x):
        _, _, h, w = x.size()
        indices = self.idx.view(*self.channels[::-1])
        cuts = self.cut.view(*self.channels[::-1])
        flops_in, flops_out = 0, 0
        ccount = x.new(self.k)
        for index, cut in zip(indices, cuts):
            sel = index.masked_select(cut.byte())
            if len(sel) > 0:
                ccount.fill_(0)
                ccount.index_add_(0, sel, x.new_ones(len(sel))).long()
                feature_conv = self.kh * self.kw * ccount.gt(0).sum()
                flops_in += h * w * feature_conv

        for index, cut in zip(indices.t(), cuts.t()):
            sel = index.masked_select(cut.byte())
            if len(sel) > 0:
                ccount.fill_(0)
                ccount.index_add_(0, sel, x.new_ones(len(sel))).long()
                feature_conv = self.kh * self.kw * ccount.gt(0).sum()
                flops_out += h * w * feature_conv

        self.flops_original \
            = h * w \
            * self.channels[0] * self.channels[1] \
            * self.kh * self.kw

        if flops_in < flops_out:
            print('In\tIn: {:.2e}\tOut: {:.2e}\tOriginal: {:.2e}'.format(
                flops_in, flops_out, self.flops_original
            ))
        else:
            print('Out\tIn: {:.2e}\tOut: {:.2e}\tOriginal: {:.2e}'.format(
                flops_in, flops_out, self.flops_original
            ))
        self.flops = min(flops_in, flops_out)

def gscluster_conv(
    in_channels, out_channels, kernel_size, stride=1, bias=True):

    return GSCConv()

class GSCConv(CConv):
    def __init__(self):
        super(GSCConv, self).__init__()

    def set_params(self, *args, **kwargs):
        super(GSCConv, self).set_params(*args, **kwargs)

        n = self.channels[0] * self.channels[1]
        tfs = clustering.get_grid(self.kh * self.kw)
        offsets = (self.kh * self.kw) * torch.arange(n).view(-1, 1).long()

        self.register_buffer('t', self.idx / self.k)
        self.register_buffer('sampler', (tfs[self.t] + offsets).view(-1))
        self.idx %= self.k

    def get_parameters(self):
        weight = self.centroids[self.idx].view(-1)[self.sampler]
        weight = weight.view(
            *self.channels[::-1],
            self.kh,
            self.kw
        )
        weight = weight * self.get_scales()
        if hasattr(self, 'cut'): weight = weight * self.cut
        bias = getattr(self, 'bias', None)

        return weight, bias

    def forward(self, x):
        weight, bias = self.get_parameters()
        x = F.conv2d(x, weight, bias, **self.kwargs)

        return x

