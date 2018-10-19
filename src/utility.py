import os
import math
import time
import datetime
from decimal import Decimal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchvision.utils as tu

import misc.warm_multi_step_lr as misc_wms
import misc.custom_sgd as misc_cs

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now

            self.dir = os.path.join('..', 'experiment', args.save)
            if args.reset:
                os.system('rm -rf ' + self.dir)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if not os.path.exists(self.dir):
                args.load = ''

        os.makedirs(os.path.join(self.dir, 'model'), exist_ok=True)
        os.makedirs(os.path.join(self.dir, 'results'), exist_ok=True)

        log_dir = os.path.join(self.dir, 'log.txt')
        open_type = 'a' if os.path.exists(log_dir) else 'w'
        self.log_file = open(log_dir, open_type)
        with open(os.path.join(self.dir, 'config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.compare = []
        if len(args.compare) > 0:
            if args.compare == 'same':
                no_postfix = '_'.join(args.save.split('_')[:-1])
                for d in os.listdir('../experiment'):
                    if d.find(no_postfix) >= 0 and d != args.save:
                        self.compare.append(d)
            else:
                self.compare = args.compare.split('+')

        self.write_log('Batch size: {} = {} x {}'.format(
            args.batch_size, args.linear, args.batch_size // args.linear
        ))

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir)

        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot(self, epoch):
        pass

    def save_results(self, epoch, m):
        m = m.get_model()
        if hasattr(m, 'split'):
            from model import clustering
            for i in set(v for v in m.split['map'].values()):
                centroids = getattr(m, 'centroids_{}'.format(i), None)
                if centroids is not None:
                    clustering.save_kernels(
                        centroids,
                        '{}/results/iter{:0>3}_c{:0>2}.png'.format(
                            self.dir, epoch, i
                        )
                    )

def make_optimizer(args, target, ckp=None, lr=None):
    trainable = filter(lambda x: x.requires_grad, target.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum, 'nesterov': args.nesterov}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': args.betas,
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}
    elif args.optimizer == 'CustomSGD':
        optimizer_function = misc_cs.CustomSGD
        kwargs = {'momentum': args.momentum, 'nesterov': args.nesterov}

    kwargs['lr'] = args.lr if lr is None else lr
    kwargs['weight_decay'] = args.weight_decay
    
    optimizer = optimizer_function(trainable, **kwargs)

    if args.load != '' and ckp is not None:
        print('Loading the optimizer from the checkpoint...')
        optimizer.load_state_dict(
            torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
        )

    return optimizer

def make_scheduler(args, target, resume=-1):
    if args.decay.find('step') >= 0:
        milestones = list(map(lambda x: int(x), args.decay.split('-')[1:]))
        kwargs = {'milestones': milestones, 'gamma': args.gamma}
        if args.decay.find('warm') >= 0:
            scheduler_function = misc_wms.WarmMultiStepLR
            kwargs['scale'] = args.linear
        else:
            scheduler_function = lrs.MultiStepLR

        scheduler = scheduler_function(target, **kwargs)

    if args.load != '' and resume > 0:
        for _ in range(resume): scheduler.step()
        
    return scheduler

