import torch
import torch.nn as nn
from torch.autograd import Variable

from model import common
from model import vgg

def make_model(args, parent=False):
    return Efficient(args)

class Efficient(vgg.VGG):
    def __init__(self, args, conv=common.default_conv):
        super(Efficient, self).__init__(args, conv)
        if args.extend:
            self.load_state_dict(torch.load(args.extend))

        self.percent = [
            50,     # Conv_1
            0,      # Conv_2
            0,      # Conv_3
            0,      # Conv_4
            0,      # Conv_5
            0,      # Conv_6
            0,      # Conv_7
            50,     # Conv_8
            50,     # Conv_9
            50,     # Conv_10
            50,     # Conv_11
            50,     # Conv_12
            50,     # Conv_13
            0]      # Linear
        self.res = [32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
        self.prune_layer = 0

        self.convs = [
            m for m in self.modules() if isinstance(m, nn.Conv2d)]
        self.convs.extend([
            m for m in self.modules() if isinstance(m, nn.Linear)])
        self.bns = [
            m for m in self.modules() if isinstance(m, nn.BatchNorm2d)]

        self.flops_full = self._calculate_flops()
        self.params_full = sum([p.nelement() for p in self.parameters()])

        if args.iterative_pruning == 0:
            prune_total = len(self.percent) - self.percent.count(0)
            for _ in range(prune_total):
                print(self.prune())

    def prune(self, threshold=0):
        for i, c in enumerate(self.convs):
            if i == self.prune_layer:
                c_out = c.weight.size(0)
                weight = c.weight.data.view(c_out, -1)
                criterion = weight.abs().sum(dim=1)
                _, order = criterion.sort(descending=True)
                order = order[:(c_out * (100 - self.percent[i])) // 100]
                _, order = order.sort(descending=False)
                
                self._reduce_dim(c, 0, order)
                if len(self.bns) > i:
                    self._reduce_dim(self.bns[i], 0, order)
                if i + 1 < len(self.convs):
                    self._reduce_dim(self.convs[i + 1], 1, order)
                
                break

        flops_reduced = self._calculate_flops()
        params_reduced = sum([p.nelement() for p in self.parameters()])

        log_list = ['Pruning layer {}...'.format(self.prune_layer + 1)]
        log_list.extend([
            '{}: {} -> {} ({:.2f}%) reduction'.format(t, f, r, 100 * r / f) \
                for t, f, r in (
                    ('Parameters', self.params_full, params_reduced),
                    ('FLOPs', self.flops_full, flops_reduced))])

        res = '\n'.join(log_list)

        while True:
            self.prune_layer += 1
            if self.prune_layer >= len(self.percent) \
            or self.percent[self.prune_layer] != 0:
                break

        return res

    def _reduce_dim(self, m, dim, order):
        for p in m.parameters():
            if dim < p.dim():
                p.data = p.data.index_select(dim, order)
                p.grad = None

        if isinstance(m, nn.BatchNorm2d):
            m.out_channels = len(order)
            m.in_channels = len(order)
            m.running_mean = m.running_mean.index_select(0, order)
            m.running_var = m.running_var.index_select(0, order)
        else:
            if dim == 0:
                m.out_channels = len(order)
            elif dim == 1:
                m.in_channels = len(order)

    def _calculate_flops(self):
        ret = 0
        for i, c in enumerate(self.convs):
            weight = c.weight.data
            if weight.dim() == 4:
                c_out, c_in, kh, kw = weight.size()
                flops = c_out * c_in * kh * kw * (self.res[i]**2)
            elif weight.dim() == 2:
                c_out, c_in = weight.size()
                flops = c_out * c_in

            ret += flops
            print('Conv {:0>2}: {: >3} -> {: >3} - {:.4f} FLOPs'.format(
                i + 1, c_in, c_out, flops / 10**6))

        return ret

