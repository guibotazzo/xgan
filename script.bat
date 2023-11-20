rmdir /s /q weights
rmdir /s /q runs

python train.py --gan WGAN-GP --xai deeplift
mkdir weights\WGAN-GP\cifar10\deeplift\run1
move weights\WGAN-GP\cifar10\deeplift\disc_epoch_100.pth weights\WGAN-GP\cifar10\deeplift\run1\
move weights\WGAN-GP\cifar10\deeplift\gen_epoch_100.pth weights\WGAN-GP\cifar10\deeplift\run1\
move runs\* weights\WGAN-GP\cifar10\deeplift\run1\
del weights\WGAN-GP\cifar10\deeplift\*.pth
