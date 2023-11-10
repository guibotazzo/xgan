#!/bin/bash

rm -r weights
rm -r runs

# python xdcgan.py -x gradcam -d mnist -s 28 -c 1 -e 100
# mkdir weights/xdcgan/mnist/gradcam/run1/
# mv weights/xdcgan/mnist/gradcam/disc_epoch_100.pth weights/xdcgan/mnist/gradcam/run1/
# mv weights/xdcgan/mnist/gradcam/gen_epoch_100.pth weights/xdcgan/mnist/gradcam/run1/
# mv runs/* weights/xdcgan/mnist/gradcam/run1/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/mnist/gradcam/run1/
mv weights/xwgangp/mnist/gradcam/disc_epoch_100.pth weights/xwgangp/mnist/gradcam/run1/
mv weights/xwgangp/mnist/gradcam/gen_epoch_100.pth weights/xwgangp/mnist/gradcam/run1/
mv runs/* weights/xwgangp/mnist/gradcam/run1/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/mnist/gradcam/run2/
mv weights/xwgangp/mnist/gradcam/disc_epoch_100.pth weights/xwgangp/mnist/gradcam/run2/
mv weights/xwgangp/mnist/gradcam/gen_epoch_100.pth weights/xwgangp/mnist/gradcam/run2/
mv runs/* weights/xwgangp/mnist/gradcam/run2/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/mnist/gradcam/run3/
mv weights/xwgangp/mnist/gradcam/disc_epoch_100.pth weights/xwgangp/mnist/gradcam/run3/
mv weights/xwgangp/mnist/gradcam/gen_epoch_100.pth weights/xwgangp/mnist/gradcam/run3/
mv runs/* weights/xwgangp/mnist/gradcam/run3/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/mnist/gradcam/run4/
mv weights/xwgangp/mnist/gradcam/disc_epoch_100.pth weights/xwgangp/mnist/gradcam/run4/
mv weights/xwgangp/mnist/gradcam/gen_epoch_100.pth weights/xwgangp/mnist/gradcam/run4/
mv runs/* weights/xwgangp/mnist/gradcam/run4/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/mnist/gradcam/run5/
mv weights/xwgangp/mnist/gradcam/disc_epoch_100.pth weights/xwgangp/mnist/gradcam/run5/
mv weights/xwgangp/mnist/gradcam/gen_epoch_100.pth weights/xwgangp/mnist/gradcam/run5/
mv runs/* weights/xwgangp/mnist/gradcam/run5/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/mnist/gradcam/run6/
mv weights/xwgangp/mnist/gradcam/disc_epoch_100.pth weights/xwgangp/mnist/gradcam/run6/
mv weights/xwgangp/mnist/gradcam/gen_epoch_100.pth weights/xwgangp/mnist/gradcam/run6/
mv runs/* weights/xwgangp/mnist/gradcam/run6/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/mnist/gradcam/run7/
mv weights/xwgangp/mnist/gradcam/disc_epoch_100.pth weights/xwgangp/mnist/gradcam/run7/
mv weights/xwgangp/mnist/gradcam/gen_epoch_100.pth weights/xwgangp/mnist/gradcam/run7/
mv runs/* weights/xwgangp/mnist/gradcam/run7/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/mnist/gradcam/run8/
mv weights/xwgangp/mnist/gradcam/disc_epoch_100.pth weights/xwgangp/mnist/gradcam/run8/
mv weights/xwgangp/mnist/gradcam/gen_epoch_100.pth weights/xwgangp/mnist/gradcam/run8/
mv runs/* weights/xwgangp/mnist/gradcam/run8/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/mnist/gradcam/run9/
mv weights/xwgangp/mnist/gradcam/disc_epoch_100.pth weights/xwgangp/mnist/gradcam/run9/
mv weights/xwgangp/mnist/gradcam/gen_epoch_100.pth weights/xwgangp/mnist/gradcam/run9/
mv runs/* weights/xwgangp/mnist/gradcam/run9/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/mnist/gradcam/run10/
mv weights/xwgangp/mnist/gradcam/disc_epoch_100.pth weights/xwgangp/mnist/gradcam/run10/
mv weights/xwgangp/mnist/gradcam/gen_epoch_100.pth weights/xwgangp/mnist/gradcam/run10/
mv runs/* weights/xwgangp/mnist/gradcam/run10/