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
mkdir datasets/CR128/Benign/one_class/
mv datasets/CR128/Benign/*.png datasets/CR128/Benign/one_class/
python xwgangp.py -x gradcam -d cr -s 128 -c 3 -e 100 --feature_maps 64
mv weights/xwgangp/cr/gradcam/disc_epoch_100.pth weights/xwgangp/mnist/gradcam/run1/
mv weights/xwgangp/cr/gradcam/gen_epoch_100.pth weights/xwgangp/mnist/gradcam/run1/
mv runs/* weights/xwgangp/cr/gradcam/run1/