import os

import torch
import torch.nn as nn
import torch.utils.model_zoo

from model import common

def make_model(args, parent=False):
    return VGG(args)

# reference: torchvision
class VGG(nn.Module):
    def __init__(self, args, conv3x3=common.default_conv, conv1x1=None):
        super(VGG, self).__init__()

        # we use batch noramlization for VGG
        norm = common.default_norm
        bias = not args.no_bias

        configs = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            'ef': [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M']
        }

        body_list = []
        in_channels = args.n_colors
        for v in configs[args.vgg_type]:
            if v == 'M':
                body_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                body_list.append(common.BasicBlock(
                    in_channels, v, args.kernel_size,
                    bias=bias, conv3x3=conv3x3, norm=norm
                ))
                in_channels = v

        # for CIFAR10 and CIFAR100 only
        assert(args.data_train.find('CIFAR') >= 0)
        n_classes = int(args.data_train[5:])

        self.features = nn.Sequential(*body_list)
        self.classifier = nn.Linear(in_channels, n_classes)

        if conv3x3 == common.default_conv:
            if args.pretrained == 'download' or args.extend == 'download':
                url = (
                    'https://cv.snu.ac.kr/'
                    'research/clustering_kernels/models/vgg16-89711a85.pt'
                )
                model_dir = os.path.join('..', 'models')
                os.makedirs(model_dir, exist_ok=True)
                state = torch.utils.model_zoo.load_url(url, model_dir=model_dir)
            elif args.extend:
                state = torch.load(args.extend)
            else:
                common.init_vgg(self)
                return

            self.load_state_dict(state)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.squeeze())

        return x

