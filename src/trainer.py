import math
import random

import utility

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as tu
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss

        self.optimizer = utility.make_optimizer(args, self.model, ckp=ckp)
        self.scheduler = utility.make_scheduler(
            args,
            self.optimizer,
            resume=len(self.loss.log_test)
        )

        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if args.model.find('INQ') >= 0:
            self.inq_steps = args.inq_steps
        else:
            self.inq_steps = None

    def train(self):
        epoch, _ = self.start_epoch()
        self.model.begin(epoch, self.ckp)
        self.loss.start_log()

        timer_data, timer_model = utility.timer(), utility.timer()
        n_samples = 0
        for batch, (img, label) in enumerate(self.loader_train):
            img, label = self.prepare(img, label)
            n_samples += img.size(0)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            prediction = self.model(img)
            loss, _ = self.loss(prediction, label)
            loss.backward()
            self.optimizer.step()

            timer_model.hold()
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log(
                    '{}/{} ({:.0f}%)\t'
                    'NLL: {:.3f}\t'
                    'Top1: {:.2f} / Top5: {:.2f}\t'
                    'Time: {:.1f}+{:.1f}s'.format(
                        n_samples,
                        len(self.loader_train.dataset),
                        100.0 * n_samples / len(self.loader_train.dataset),
                        *(self.loss.log_train[-1, :] / n_samples),
                        timer_model.release(),
                        timer_data.release()
                    )
                )

            timer_data.tic()
        
        self.model.log(self.ckp)
        self.loss.end_log(len(self.loader_train.dataset))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.loss.start_log(train=False)
        self.model.eval()

        timer_test = utility.timer()
        timer_test.tic()
        with torch.no_grad():
            for img, label in tqdm(self.loader_test, ncols=80):
                img, label = self.prepare(img, label)
                prediction = self.model(img)
                self.loss(prediction, label, train=False)

                if self.args.debug: self._analysis()

        self.loss.end_log(len(self.loader_test.dataset), train=False)

        # Lower is better
        best = self.loss.log_test.min(0)
        for i, measure in enumerate(('Loss', 'Top1 error', 'Top5 error')):
            self.ckp.write_log(
                '{}: {:.3f} (Best: {:.3f} from epoch {})'.format(
                    measure,
                    self.loss.log_test[-1, i],
                    best[0][i],
                    best[1][i] + 1
                )
            )

        self.ckp.write_log(
            'Time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        is_best = self.loss.log_test[-1, -1] <= best[0][-1]
        self.ckp.save(self, epoch, is_best=is_best)
        self.ckp.save_results(epoch, self.model)

    def prepare(self, *args):
        def _prepare(x):
            x = x.to(self.device)
            if self.args.precision == 'half': x = x.half()
            return x

        return [_prepare(a) for a in args]

    def start_epoch(self):
        self.scheduler.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2}'.format(epoch, lr))
        
        return epoch, lr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
    
    def _analysis(self):
        flops = torch.Tensor([
            getattr(m, 'flops', 0) for m in self.model.modules()
        ])
        flops_conv = torch.Tensor([
            getattr(m, 'flops', 0) for m in self.model.modules() if isinstance(m, nn.Conv2d)
        ])
        flops_ori = torch.Tensor([
            getattr(m, 'flops_original', 0) for m in self.model.modules()
        ])

        print('')
        print('FLOPs: {:.2f} x 10^8'.format(flops.sum() / 10**8))
        print('Compressed: {:.2f} x 10^8 / Others: {:.2f} x 10^8'.format(
            (flops.sum() - flops_conv.sum()) / 10**8 , flops_conv.sum() / 10**8
        ))
        print('Accel - Total original: {:.2f} x 10^8 ({:.2f}x)'.format(
            flops_ori.sum() / 10**8, flops_ori.sum() / flops.sum()
        ))
        print('Accel - 3x3 original: {:.2f} x 10^8 ({:.2f}x)'.format(
            (flops_ori.sum() - flops_conv.sum()) / 10**8,
            (flops_ori.sum() - flops_conv.sum()) / (flops.sum() - flops_conv.sum())
        ))
        input()

