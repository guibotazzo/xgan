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

python train.py --gan WGAN-GP --xai deeplift
mkdir weights/WGAN-GP/cifar10/deeplift/run4/
mv weights/WGAN-GP/cifar10/deeplift/disc_epoch_100.pth weights/WGAN-GP/cifar10/deeplift/run4/
mv weights/WGAN-GP/cifar10/deeplift/gen_epoch_100.pth weights/WGAN-GP/cifar10/deeplift/run4/
mv runs/* weights/WGAN-GP/cifar10/deeplift/run4/
rm weights/WGAN-GP/cifar10/deeplift/*.pth

python train.py --gan WGAN-GP --xai deeplift
mkdir weights/WGAN-GP/cifar10/deeplift/run5/
mv weights/WGAN-GP/cifar10/deeplift/disc_epoch_100.pth weights/WGAN-GP/cifar10/deeplift/run5/
mv weights/WGAN-GP/cifar10/deeplift/gen_epoch_100.pth weights/WGAN-GP/cifar10/deeplift/run5/
mv runs/* weights/WGAN-GP/cifar10/deeplift/run5/
rm weights/WGAN-GP/cifar10/deeplift/*.pth

python train.py --gan WGAN-GP --xai saliency
mkdir weights/WGAN-GP/cifar10/saliency/run2/
mv weights/WGAN-GP/cifar10/saliency/disc_epoch_100.pth weights/WGAN-GP/cifar10/deeplift/run2/
mv weights/WGAN-GP/cifar10/saliency/gen_epoch_100.pth weights/WGAN-GP/cifar10/deeplift/run2/
mv runs/* weights/WGAN-GP/cifar10/saliency/run2/
rm weights/WGAN-GP/cifar10/saliency/*.pth

python train.py --gan WGAN-GP --xai saliency
mkdir weights/WGAN-GP/cifar10/saliency/run3/
mv weights/WGAN-GP/cifar10/saliency/disc_epoch_100.pth weights/WGAN-GP/cifar10/deeplift/run3/
mv weights/WGAN-GP/cifar10/saliency/gen_epoch_100.pth weights/WGAN-GP/cifar10/deeplift/run3/
mv runs/* weights/WGAN-GP/cifar10/saliency/run3/
rm weights/WGAN-GP/cifar10/saliency/*.pth

python train.py --gan WGAN-GP --xai saliency
mkdir weights/WGAN-GP/cifar10/saliency/run4/
mv weights/WGAN-GP/cifar10/saliency/disc_epoch_100.pth weights/WGAN-GP/cifar10/deeplift/run4/
mv weights/WGAN-GP/cifar10/saliency/gen_epoch_100.pth weights/WGAN-GP/cifar10/deeplift/run4/
mv runs/* weights/WGAN-GP/cifar10/saliency/run4/
rm weights/WGAN-GP/cifar10/saliency/*.pth

python train.py --gan WGAN-GP --xai saliency
mkdir weights/WGAN-GP/cifar10/saliency/run5/
mv weights/WGAN-GP/cifar10/saliency/disc_epoch_100.pth weights/WGAN-GP/cifar10/deeplift/run5/
mv weights/WGAN-GP/cifar10/saliency/gen_epoch_100.pth weights/WGAN-GP/cifar10/deeplift/run5/
mv runs/* weights/WGAN-GP/cifar10/saliency/run5/
rm weights/WGAN-GP/cifar10/saliency/*.pth