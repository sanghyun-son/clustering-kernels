import os 
from importlib import import_module

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.args = args
        self.crop = args.crop
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.precision = args.precision
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        if args.model.find('DeepComp') >= 0:
            dc_type = args.model.split('-')[-1]
            module = import_module('model.deepcomp')
            self.model = module.make_model(args, dc_type)
        else:
            module = import_module('model.' + args.model.lower())
            self.model = module.make_model(args)

        self.model = self.model.to(self.device)
        if args.precision == 'half': self.model = self.model.half()

        if not args.cpu:
            print('CUDA is ready!')
            torch.cuda.manual_seed(self.args.seed)
            if args.n_GPUs > 1:
                self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        
        self.load(
            ckp.dir,
            pretrained=args.pretrained,
            load=args.load,
            resume=args.resume,
            cpu=args.cpu
        )
        for m in self.modules():
            if hasattr(m, 'set_range'): m.set_range()

        print(self.get_model(), file=ckp.log_file)
        self.summarize(ckp)

    def forward(self, x):
        if self.crop > 1:
            b, n_crops, c, h, w = x.size()
            x = x.view(-1, c, h, w)

        x = self.model(x)

        if self.crop > 1: x = x.view(b, n_crops, -1).mean(1)

        return x

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        return self.get_model().state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model().state_dict()
        
        conditions = (True, is_best, self.save_models)
        names = ('latest', 'best', '{}'.format(epoch))

        for c, n in zip(conditions, names):
            if c: 
                torch.save(
                    target,
                    os.path.join(apath, 'model', 'model_{}.pt'.format(n))
                )

    def load(self, apath, pretrained='', load='', resume=-1, cpu=False):
        f = None
        if pretrained:
            if pretrained != 'download':
                print('Load pre-trained model from {}'.format(pretrained))
                f = pretrained
        else:
            if load:
                if resume == -1:
                    print('Load model after the last epoch')
                    resume = 'latest'
                else:
                    print('Load model after epoch {}'.format(resume))

                f = os.path.join(apath, 'model', 'model_{}.pt'.format(resume))
        
        if f:
            kwargs = {}
            if cpu:
                kwargs = {'map_location': lambda storage, loc: storage}

            state = torch.load(f, **kwargs)
            self.get_model().load_state_dict(state, strict=False)
    
    def begin(self, epoch, ckp):
        self.train()
        m = self.get_model()
        if hasattr(m, 'begin'): m.begin(epoch, ckp)

    def log(self, ckp):
        m = self.get_model()
        if hasattr(m, 'log'): m.log(ckp)

    def summarize(self, ckp):
        ckp.write_log('# parameters: {:,}'.format(
            sum([p.nelement() for p in self.model.parameters()])
        ))

        kernels_1x1 = 0
        kernels_3x3 = 0
        kernels_others = 0
        gen = (c for c in self.model.modules() if isinstance(c, nn.Conv2d))
        for m in gen:
            kh, kw = m.kernel_size
            n_kernels = m.in_channels * m.out_channels
            if kh == 1 and kw == 1:
                kernels_1x1 += n_kernels
            elif kh == 3 and kw == 3:
                kernels_3x3 += n_kernels
            else:
                kernels_others += n_kernels
        
        linear = sum([
            l.weight.nelement() for l in self.model.modules() \
            if isinstance(l, nn.Linear)
        ])
            
        ckp.write_log(
            '1x1: {:,}\n3x3: {:,}\nOthers: {:,}\nLinear:{:,}\n'.format(
                kernels_1x1, kernels_3x3, kernels_others, linear
            ),
            refresh=True
        )

        if self.args.debug:
            def _get_flops(conv, x, y):
                _, _, h, w = y.size()
                kh, kw = conv.kernel_size
                conv.flops \
                    = h * w \
                    *conv.in_channels * conv.out_channels * kh * kw
                conv.flops_original = conv.flops
            
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    m.register_forward_hook(_get_flops)
