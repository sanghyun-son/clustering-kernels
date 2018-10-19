def set_template(args):
    if args.template.find('CIFAR100') >= 0:
        args.data_train = 'CIFAR100'
        args.data_test = 'CIFAR100'

    if args.template.find('ImageNet') >= 0:
        args.data_train = 'ImageNet'
        args.data_test = 'ImageNet'

    if args.template.find('VGG') >= 0:
        args.base = 'VGG'
        args.base_p = 'VGG'
        args.weight_decay = 5e-4
        args.batch_size = 256
        args.print_every = 50

    if args.template.find('ResNet') >= 0:
        args.base = 'ResNet'
        args.base_p = 'ResNet'

        if args.template.find('ResNet18') >= 0:
            args.depth = 18
        elif args.template.find('ResNet50') >= 0:
            args.depth = 50
            args.bottleneck = True
        elif args.template.find('ResNet101') >= 0:
            args.depth = 101
            args.bottleneck = True

    if args.template.find('DenseNet') >= 0:
        if args.template.find('BC') >= 0:
            args.bottleneck = True
            args.reduction = 0.5
        else:
            args.bottleneck = False
            args.reduction = 1
        
        args.nesterov = True

    if args.template.find('efficient') >= 0:
        args.lr /= 100
        args.decay_type = 'step_200'
        args.epochs = 300
        args.iterative_pruning = 20

    if args.linear > 1:
        args.batch_size *= args.linear
        args.lr *= args.linear
        if args.decay.find('warm') < 0:
            args.decay = 'warm' + args.decay

        args.print_every /= args.linear

