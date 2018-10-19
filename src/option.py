import argparse
import template

parser = argparse.ArgumentParser(description='Deep Kernel Clustering')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='',
                    help='You can set various templates in template.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=10,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='disable CUDA training')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', default='../../../dataset',
                    help='dataset directory')
parser.add_argument('--data_train', default='CIFAR10',
                    help='train dataset name')
parser.add_argument('--data_test', default='CIFAR10',
                    help='test dataset name')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_flip', action='store_true',
                    help='disable flip augmentation')
parser.add_argument('--crop', type=int, default=1,
                    help='enables crop evaluation')

# Model specifications
parser.add_argument('--model', default='DenseNet',
                    help='model name')
parser.add_argument('--vgg_type', type=str, default='16',
                    help='VGG type')
parser.add_argument('--download', action='store_true',
                    help='download pre-trained model')
parser.add_argument('--base', default='',
                    help='base model')
parser.add_argument('--base_p', default='',
                    help='base model for parent')

parser.add_argument('--act', default='relu',
                    help='activation function')
parser.add_argument('--pretrained', default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', default='',
                    help='pre-trained model directory')

parser.add_argument('--depth', type=int, default=100,
                    help='number of convolution modules')
parser.add_argument('--in_channels', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--k', type=int, default=12,
                    help='DenseNet grownth rate')
parser.add_argument('--reduction', type=float, default=1,
                    help='DenseNet reduction rate')
parser.add_argument('--bottleneck', action='store_true',
                    help='ResNet/DenseNet bottleneck')

parser.add_argument('--kernel_size', type=int, default=3,
                    help='kernel size')
parser.add_argument('--no_bias', action='store_true',
                    help='do not use bias term for conv layer')
parser.add_argument('--precision', default='single',
                    help='model and data precision')

parser.add_argument('--multi', type=str, default='full-256',
                    help='multi clustering')
parser.add_argument('--n_init', type=int, default=1,
                    help='number of differnt k-means initialization')
parser.add_argument('--max_iter', type=int, default=4500,
                    help='maximum iterations for kernel clustering')
parser.add_argument('--symmetry', type=str, default='i',
                    help='clustering algorithm')
parser.add_argument('--init_seeds', type=str, default='random',
                    help='kmeans initialization method')
parser.add_argument('--scale_type', type=str, default='kernel_norm_train',
                    help='scale parameter configurations')
parser.add_argument('--n_bits', type=int, default=16,
                    help='number of bits for scale parameters')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--resume', type=int, default=-1,
                    help='load the model from the specified epoch')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training')

# Optimization specifications
parser.add_argument('--linear', type=int, default=1,
                    help='linear scaling rule')
parser.add_argument('--lr', type=float, default=1e-1,
                    help='learning rate')
parser.add_argument('--decay', default='step-150-225',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay factor')

parser.add_argument('--optimizer', type=str, default='SGD',
                    help='optimizer to use')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--nesterov', action='store_true',
                    help='enable nesterov momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM betas')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay parameter')

# Loss specifications
parser.add_argument('--loss', default='1*CE',
                    help='loss function configuration')

# Log specifications
parser.add_argument('--save', default='test',
                    help='file name to save')
parser.add_argument('--load', default='',
                    help='file name to load')
parser.add_argument('--print_every', type=int, default=100,
                    help='print intermediate status per N batches')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--compare', type=str, default='',
                    help='experiments to compare with')

args = parser.parse_args()
template.set_template(args)

if args.epochs == 0:
    args.epochs = 1e8

if args.pretrained and args.pretrained != 'download':
    args.n_init = 1
    args.max_iter = 1

