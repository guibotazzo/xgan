#!/bin/bash

rm -r weights
rm -r runs

# python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
# mkdir weights/xwgangp/mnist/gradcam/run7/
# mv weights/xwgangp/mnist/gradcam/disc_epoch_100.pth weights/xwgangp/mnist/gradcam/run7/
# mv weights/xwgangp/mnist/gradcam/gen_epoch_100.pth weights/xwgangp/mnist/gradcam/run7/
# mv runs/* weights/xwgangp/mnist/gradcam/run7/
# rm weights/xwgangp/mnist/gradcam/*.pth

# Generate benign
#mkdir datasets/CR128/Benign/one_class/
#mv datasets/CR128/Benign/*.png datasets/CR128/Benign/one_class/
#python xwgangp.py -x gradcam -d cr -s 128 -c 3 -e 100 --feature_maps 64
#mv weights/xwgangp/cr/gradcam/disc_epoch_100.pth weights/xwgangp/cr/gradcam/run1/
#mv weights/xwgangp/cr/gradcam/gen_epoch_100.pth weights/xwgangp/cr/gradcam/run1/
#mv runs/* weights/xwgangp/cr/gradcam/run1/

python teste.py --loss_D WGAN-GP -d cifar10 -s 32 -c 3
mkdir weights/WGAN-GP/cifar10/run1/
mv weights/WGAN-GP/cifar10/disc_epoch_100.pth weights/WGAN-GP/cifar10/run1/
mv weights/WGAN-GP/cifar10/gen_epoch_100.pth weights/WGAN-GP/cifar10/run1/
mv runs/* weights/WGAN-GP/cifar10/run1/
rm weights/WGAN-GP/cifar10/*.pth

python teste.py --loss_D WGAN-GP -d cifar10 -s 32 -c 3
mkdir weights/WGAN-GP/cifar10/run2/
mv weights/WGAN-GP/cifar10/disc_epoch_100.pth weights/WGAN-GP/cifar10/run2/
mv weights/WGAN-GP/cifar10/gen_epoch_100.pth weights/WGAN-GP/cifar10/run2/
mv runs/* weights/WGAN-GP/cifar10/run2/
rm weights/WGAN-GP/cifar10/*.pth

python teste.py --loss_D WGAN-GP -d cifar10 -s 32 -c 3
mkdir weights/WGAN-GP/cifar10/run3/
mv weights/WGAN-GP/cifar10/disc_epoch_100.pth weights/WGAN-GP/cifar10/run3/
mv weights/WGAN-GP/cifar10/gen_epoch_100.pth weights/WGAN-GP/cifar10/run3/
mv runs/* weights/WGAN-GP/cifar10/run3/
rm weights/WGAN-GP/cifar10/*.pth

python teste.py --loss_D WGAN-GP -d cifar10 -s 32 -c 3
mkdir weights/WGAN-GP/cifar10/run4/
mv weights/WGAN-GP/cifar10/disc_epoch_100.pth weights/WGAN-GP/cifar10/run4/
mv weights/WGAN-GP/cifar10/gen_epoch_100.pth weights/WGAN-GP/cifar10/run4/
mv runs/* weights/WGAN-GP/cifar10/run4/
rm weights/WGAN-GP/cifar10/*.pth

python teste.py --loss_D WGAN-GP -d cifar10 -s 32 -c 3
mkdir weights/WGAN-GP/cifar10/run5/
mv weights/WGAN-GP/cifar10/disc_epoch_100.pth weights/WGAN-GP/cifar10/run5/
mv weights/WGAN-GP/cifar10/gen_epoch_100.pth weights/WGAN-GP/cifar10/run5/
mv runs/* weights/WGAN-GP/cifar10/run5/
rm weights/WGAN-GP/cifar10/*.pth

python teste.py --loss_D WGAN-GP -d cifar10 -s 32 -c 3
mkdir weights/WGAN-GP/cifar10/run6/
mv weights/WGAN-GP/cifar10/disc_epoch_100.pth weights/WGAN-GP/cifar10/run6/
mv weights/WGAN-GP/cifar10/gen_epoch_100.pth weights/WGAN-GP/cifar10/run6/
mv runs/* weights/WGAN-GP/cifar10/run6/
rm weights/WGAN-GP/cifar10/*.pth

python teste.py --loss_D WGAN-GP -d cifar10 -s 32 -c 3
mkdir weights/WGAN-GP/cifar10/run7/
mv weights/WGAN-GP/cifar10/disc_epoch_100.pth weights/WGAN-GP/cifar10/run7/
mv weights/WGAN-GP/cifar10/gen_epoch_100.pth weights/WGAN-GP/cifar10/run7/
mv runs/* weights/WGAN-GP/cifar10/run7/
rm weights/WGAN-GP/cifar10/*.pth

python teste.py --loss_D WGAN-GP -d cifar10 -s 32 -c 3
mkdir weights/WGAN-GP/cifar10/run8/
mv weights/WGAN-GP/cifar10/disc_epoch_100.pth weights/WGAN-GP/cifar10/run8/
mv weights/WGAN-GP/cifar10/gen_epoch_100.pth weights/WGAN-GP/cifar10/run8/
mv runs/* weights/WGAN-GP/cifar10/run8/
rm weights/WGAN-GP/cifar10/*.pth

python teste.py --loss_D WGAN-GP -d cifar10 -s 32 -c 3
mkdir weights/WGAN-GP/cifar10/run9/
mv weights/WGAN-GP/cifar10/disc_epoch_100.pth weights/WGAN-GP/cifar10/run9/
mv weights/WGAN-GP/cifar10/gen_epoch_100.pth weights/WGAN-GP/cifar10/run9/
mv runs/* weights/WGAN-GP/cifar10/run9/
rm weights/WGAN-GP/cifar10/*.pth

python teste.py --loss_D WGAN-GP -d cifar10 -s 32 -c 3
mkdir weights/WGAN-GP/cifar10/run10/
mv weights/WGAN-GP/cifar10/disc_epoch_100.pth weights/WGAN-GP/cifar10/run10/
mv weights/WGAN-GP/cifar10/gen_epoch_100.pth weights/WGAN-GP/cifar10/run10/
mv runs/* weights/WGAN-GP/cifar10/run10/
rm weights/WGAN-GP/cifar10/*.pth