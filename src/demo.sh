# training VGG16-like baseline on CIFAR10
#python main.py --template VGG --model VGG --vgg_type 16 --pretrained download --test_only

# VGG16-C256
#python main.py --template VGG --vgg_type 16 --model ClusterNet --multi full-256 --lr 5e-3 --decay step-150-225 --extend download --save vgg16_c256 --reset

# VGG16-C128-TIC2
#python main.py --template VGG --vgg_type 16 --model ClusterNet --multi full-128 --symmetry ih --lr 5e-3 --decay step-150-225 --extend download --save vgg16_c128_tic2_test --reset

# VGG16-C32-TIC8
#python main.py --template VGG --vgg_type 16 --model ClusterNet --multi full-32 --symmetry ihvoIHVO --lr 5e-3 --decay step-150-225 --extend download --save vgg16_c32_tic8_test --reset

# Baseline ResNet18
#python main.py --template ImageNet-ResNet18 --model ResNet --pretrained download --test_only

# ResNet18-C256
#python main.py --template ImageNet-ResNet18 --model ClusterNet --lr 5e-3 --decay step-20-40 --epochs 60 --extend download --multi 0-0=64,1-16=256 --save resnet18_c256 --reset

# ResNet18-C1024
#python main.py --template ImageNet-ResNet18 --model ClusterNet --lr 5e-3 --decay step-20-40 --epochs 60 --extend download --multi 0-0=64,1-16=1024 --save resnet18_c1024 --reset
